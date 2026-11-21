"""Train a student with CE, vanilla KD, or KD + logit standardization (KD+LS).

Examples
  python train.py --preset c100_kd_ls                 # one of lskd/experiments.py
  python train.py --preset c100_kd --epochs 120       # reduced schedule
  python train.py --preset c100_kd_ls --smoke         # 2 tiny epochs, checks everything runs
  python train.py --preset c100_kd_ls --resume        # continue after a Colab disconnect

Outputs (in runs/<name>/): config.json, history.csv, last.pth, best.pth, final.pth, summary.json

ATTRIBUTION: IMPLEMENTED BY OUR TEAM. The training recipe (optimizer, schedule,
loss weights) follows mdistiller / the official logit-standardization-KD code,
but this script is our own.
"""
import argparse
import csv
import json
import os
import time

import torch

from lskd.data import CLASS_NAMES, NUM_CLASSES, STATS, TensorLoader, load_arrays
from lskd.experiments import PRESETS
from lskd.losses import total_loss
from lskd.models import build_model, count_parameters, load_model_weights
from lskd.utils import get_device, save_json, set_seed, topk_correct, torch_load


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default=None, help="name from lskd/experiments.py")
    p.add_argument("--name", default=None, help="run name (default: preset name)")
    p.add_argument("--dataset", default="cifar100", choices=["cifar100", "cifar10"])
    p.add_argument("--arch", default="resnet20")
    p.add_argument("--method", default="kd_ls", choices=["ce", "kd", "kd_ls"])
    p.add_argument("--teacher-arch", dest="teacher_arch", default="resnet56")
    p.add_argument("--teacher-ckpt", dest="teacher_ckpt", default=None)
    p.add_argument("--temperature", type=float, default=2.0, help="T for KD, base tau for KD+LS")
    p.add_argument("--ce-weight", dest="ce_weight", type=float, default=0.1)
    p.add_argument("--kd-weight", dest="kd_weight", type=float, default=9.0)
    p.add_argument("--epochs", type=int, default=240)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", dest="data_dir", default="./data")
    p.add_argument("--out-dir", dest="out_dir", default="./runs")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="disable mixed precision")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true", help="2 epochs on 512 images: pipeline check only")
    args = p.parse_args()

    # preset values override the defaults, but explicit CLI flags override the preset
    if args.preset:
        if args.preset not in PRESETS:
            raise SystemExit("Unknown preset %s. Available: %s" % (args.preset, ", ".join(PRESETS)))
        defaults = vars(p.parse_args([]))
        explicit = {k for k, v in vars(args).items() if v != defaults.get(k)}
        for k, v in PRESETS[args.preset].items():
            if k not in explicit:
                setattr(args, k, v)
        args.name = args.name or args.preset
    args.name = args.name or "%s_%s_%s" % (args.dataset, args.method, args.arch)
    if args.smoke:
        args.epochs = 2
        args.name = args.name + "_smoke"
    if args.method != "ce" and not args.teacher_ckpt:
        raise SystemExit("--teacher-ckpt is required for method %s" % args.method)
    return args


def lr_at(epoch, args):
    """Step schedule: x0.1 at 150/240, 180/240, 210/240 of training (scaled if --epochs changes)."""
    milestones = [round(args.epochs * f) for f in (150 / 240, 180 / 240, 210 / 240)]
    return args.lr * (0.1 ** sum(epoch >= m for m in milestones))


@torch.no_grad()
def evaluate(model, loader, amp, device):
    model.eval()
    n = c1 = c5 = 0
    loss_sum = 0.0
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
        logits = logits.float()
        loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
        a1, a5 = topk_correct(logits, y)
        c1 += a1
        c5 += a5
        n += y.numel()
    return 100.0 * c1 / n, 100.0 * c5 / n, loss_sum / n


def main():
    args = parse_args()
    device = get_device(args.device)
    amp = bool(args.amp and device.type == "cuda")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    run_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    print("Run %s on %s (amp=%s)" % (args.name, device, amp))
    print(json.dumps(vars(args), indent=2))

    # ---------------- data ----------------
    limit = 512 if args.smoke else None
    xtr, ytr = load_arrays(args.dataset, args.data_dir, train=True)
    xte, yte = load_arrays(args.dataset, args.data_dir, train=False)
    train_loader = TensorLoader(xtr, ytr, args.dataset, args.batch_size, True, device, args.seed, limit)
    test_loader = TensorLoader(xte, yte, args.dataset, 500, False, device, limit=limit)
    num_classes = NUM_CLASSES[args.dataset]

    # ---------------- models ----------------
    student = build_model(args.arch, num_classes).to(device)
    teacher = None
    if args.method != "ce":
        teacher = build_model(args.teacher_arch, num_classes).to(device)
        load_model_weights(teacher, args.teacher_ckpt)
        teacher.eval()  # frozen: BN uses running statistics, no gradients
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        t_acc, t_acc5, _ = evaluate(teacher, test_loader, amp, device)
        print("Teacher %s test top-1 = %.2f%%" % (args.teacher_arch, t_acc))
        if not args.smoke and t_acc < 50:
            raise SystemExit("Teacher accuracy is suspiciously low - wrong checkpoint/dataset?")
    else:
        t_acc = None
    if device.type == "cuda":
        student = student.to(memory_format=torch.channels_last)
        if teacher is not None:
            teacher = teacher.to(memory_format=torch.channels_last)
    print("Student %s: %.3fM parameters" % (args.arch, count_parameters(student) / 1e6))

    opt = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=False)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    start_epoch, best_acc, best_epoch = 0, 0.0, -1
    hist_path = os.path.join(run_dir, "history.csv")
    last_path = os.path.join(run_dir, "last.pth")
    if args.resume and os.path.exists(last_path):
        ck = torch_load(last_path, map_location=device)
        student.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_acc, best_epoch = ck["epoch"] + 1, ck["best_acc"], ck["best_epoch"]
        print("Resumed from epoch %d (best %.2f%%)" % (start_epoch, best_acc))
    elif os.path.exists(hist_path):
        os.remove(hist_path)

    def checkpoint(epoch, test_acc, extra=None):
        d = {
            "model": student.state_dict(), "arch": args.arch, "num_classes": num_classes,
            "dataset": args.dataset, "method": args.method, "class_names": CLASS_NAMES[args.dataset],
            "norm_mean": STATS[args.dataset][0], "norm_std": STATS[args.dataset][1],
            "epoch": epoch, "test_acc": test_acc, "best_acc": best_acc, "best_epoch": best_epoch,
            "args": vars(args), "teacher_acc": t_acc,
        }
        if extra:
            d.update(extra)
        return d

    fields = ["epoch", "lr", "train_loss", "train_ce", "train_kd", "train_acc",
              "test_acc", "test_acc5", "test_loss", "epoch_time_s"]
    total_t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        lr = lr_at(epoch, args)
        for g in opt.param_groups:
            g["lr"] = lr
        student.train()
        t0 = time.time()
        n = correct = 0
        s_loss = s_ce = s_kd = 0.0
        for x, y in train_loader:
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, enabled=amp):
                s_logits = student(x)
                t_logits = None
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x)
            loss, ce_term, kd_term = total_loss(
                s_logits, y, t_logits, args.method, args.temperature, args.ce_weight, args.kd_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            b = y.numel()
            n += b
            # accumulate on-device (avoids a GPU->CPU sync every step)
            s_loss = s_loss + loss.detach() * b
            s_ce = s_ce + ce_term.detach() * b
            s_kd = s_kd + kd_term.detach() * b
            correct = correct + (s_logits.argmax(1) == y).sum()
        s_loss, s_ce, s_kd, correct = float(s_loss), float(s_ce), float(s_kd), int(correct)
        if not torch.isfinite(torch.tensor(s_loss)):
            raise SystemExit("Loss became NaN/inf at epoch %d" % epoch)
        test_acc, test_acc5, test_loss = evaluate(student, test_loader, amp, device)
        if test_acc > best_acc:
            best_acc, best_epoch = test_acc, epoch
            torch.save(checkpoint(epoch, test_acc), os.path.join(run_dir, "best.pth"))
        dt = time.time() - t0
        row = dict(epoch=epoch, lr=lr, train_loss=s_loss / n, train_ce=s_ce / n, train_kd=s_kd / n,
                   train_acc=100.0 * correct / n, test_acc=test_acc, test_acc5=test_acc5,
                   test_loss=test_loss, epoch_time_s=dt)
        new_file = not os.path.exists(hist_path)
        with open(hist_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                w.writeheader()
            w.writerow(row)
        ck = checkpoint(epoch, test_acc, {"optimizer": opt.state_dict(), "scaler": scaler.state_dict()})
        torch.save(ck, last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)  # atomic: safe if Colab disconnects mid-write
        print("ep %3d/%d lr %.4f | loss %.4f (ce %.4f kd %.4f) train %.2f%% | test %.2f%% (best %.2f%% @%d) | %.1fs"
              % (epoch + 1, args.epochs, lr, row["train_loss"], row["train_ce"], row["train_kd"],
                 row["train_acc"], test_acc, best_acc, best_epoch + 1, dt), flush=True)

    # final (last-epoch) weights are the ones we report and deploy; best.pth is kept for reference
    final_acc, final_acc5, _ = evaluate(student, test_loader, amp, device)
    torch.save(checkpoint(args.epochs - 1, final_acc), os.path.join(run_dir, "final.pth"))
    summary = dict(name=args.name, preset=args.preset, dataset=args.dataset, method=args.method,
                   arch=args.arch, teacher_arch=args.teacher_arch if teacher is not None else None,
                   teacher_acc=t_acc, temperature=args.temperature, ce_weight=args.ce_weight,
                   kd_weight=args.kd_weight, epochs=args.epochs, seed=args.seed,
                   final_acc=final_acc, final_acc5=final_acc5, best_acc=best_acc,
                   best_epoch=best_epoch + 1, train_minutes=(time.time() - total_t0) / 60,
                   params_M=count_parameters(student) / 1e6, gpu=torch.cuda.get_device_name(0)
                   if device.type == "cuda" else "cpu", torch=torch.__version__)
    save_json(summary, os.path.join(run_dir, "summary.json"))
    save_json(vars(args), os.path.join(run_dir, "config.json"))
    print("DONE", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
