"""Evaluate trained models on the CIFAR test set and compute the diagnostics
used in the report.

  python evaluate.py --dataset cifar100 \
      --teacher checkpoints/teachers/resnet56_cifar100.pth \
      --models KD=runs/c100_kd/final.pth KD+LS=runs/c100_kd_ls/final.pth

For every model it reports:
  * top-1 / top-5 accuracy, per-class accuracy, expected calibration error (ECE)
  * logit statistics: average per-sample logit mean and standard deviation
    (paper Sec. 5.2 / Fig. 3: LS students should be free to have a different
    logit scale from the teacher)
  * w.r.t. the teacher: top-1 agreement, Spearman rank correlation of the
    100 logits (the "innate logit relation" the paper says matters), and the
    KL divergence measured both with raw logits (T=4) and with Z-scored logits (tau=2)
  * pairwise prediction agreement between all models (proposal: "prediction
    differences between the two student models")

Outputs: results/eval_<tag>.json and results/logits_<tag>.npz

ATTRIBUTION: IMPLEMENTED BY OUR TEAM.
"""
import argparse
import itertools
import os

import numpy as np
import torch

from lskd.data import CLASS_NAMES, NUM_CLASSES, TensorLoader, load_arrays
from lskd.losses import distillation_loss
from lskd.models import build_model, load_model_weights
from lskd.utils import get_device, save_json, sha256_of, torch_load


def load_any(path, dataset, device, arch=None):
    ck = torch_load(path)
    arch = arch or (ck.get("arch") if isinstance(ck, dict) else None) or "resnet56"
    model = build_model(arch, NUM_CLASSES[dataset])
    load_model_weights(model, path)
    return model.to(device).eval(), arch


@torch.no_grad()
def all_logits(model, loader):
    out = [model(x).float() for x, _ in loader]
    return torch.cat(out)


def ece(probs, labels, n_bins=15):
    conf, pred = probs.max(1)
    acc = pred.eq(labels).float()
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    e = torch.zeros((), device=probs.device)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean() * (conf[m].mean() - acc[m].mean()).abs()
    return 100.0 * e.item()


def spearman_rows(a, b):
    """Mean Spearman correlation between corresponding rows of a and b."""
    ra = a.argsort(1).argsort(1).float()
    rb = b.argsort(1).argsort(1).float()
    ra = ra - ra.mean(1, keepdim=True)
    rb = rb - rb.mean(1, keepdim=True)
    return ((ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1))).mean().item()


def describe(logits, labels, class_names):
    probs = logits.softmax(1)
    pred = logits.argmax(1)
    top5 = logits.topk(5, 1).indices.eq(labels[:, None]).any(1).float().mean().item() * 100
    k = logits.shape[1]
    per_class = [(pred[labels == c] == c).float().mean().item() * 100 for c in range(k)]
    order = np.argsort(per_class)
    # most frequent confusions (true -> predicted)
    wrong = pred != labels
    pairs = torch.stack([labels[wrong], pred[wrong]], 1).cpu().numpy()
    uniq, counts = np.unique(pairs, axis=0, return_counts=True) if len(pairs) else (np.zeros((0, 2)), [])
    top_conf = [dict(true=class_names[int(t)], pred=class_names[int(p)], count=int(c))
                for (t, p), c in sorted(zip(map(tuple, uniq), counts), key=lambda z: -z[1])[:10]]
    return dict(
        top1=pred.eq(labels).float().mean().item() * 100, top5=top5, ece=ece(probs, labels),
        mean_confidence=probs.max(1).values.mean().item() * 100,
        logit_mean=logits.mean(1).mean().item(), logit_std=logits.std(1).mean().item(),
        logit_max=logits.max(1).values.mean().item(),
        worst_classes=[(class_names[i], round(per_class[i], 1)) for i in order[:5]],
        best_classes=[(class_names[i], round(per_class[i], 1)) for i in order[-5:][::-1]],
        top_confusions=top_conf, per_class_acc=per_class,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar100", choices=["cifar100", "cifar10"])
    p.add_argument("--teacher", default=None)
    p.add_argument("--teacher-arch", dest="teacher_arch", default="resnet56")
    p.add_argument("--models", nargs="+", required=True, help="label=path pairs")
    p.add_argument("--tag", default=None)
    p.add_argument("--data-dir", dest="data_dir", default="./data")
    p.add_argument("--out-dir", dest="out_dir", default="./results")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    names = CLASS_NAMES[args.dataset]
    x, y = load_arrays(args.dataset, args.data_dir, train=False)
    loader = TensorLoader(x, y, args.dataset, 500, False, device)
    labels = torch.from_numpy(y).to(device)

    results, logits = {}, {}
    if args.teacher:
        t, _ = load_any(args.teacher, args.dataset, device, args.teacher_arch)
        logits["Teacher"] = all_logits(t, loader)
        results["Teacher"] = describe(logits["Teacher"], labels, names)
        results["Teacher"]["checkpoint"] = args.teacher
    for item in args.models:
        label, path = item.split("=", 1)
        m, arch = load_any(path, args.dataset, device)
        logits[label] = all_logits(m, loader)
        r = describe(logits[label], labels, names)
        r.update(checkpoint=path, sha256=sha256_of(path), arch=arch)
        if args.teacher:
            T = logits["Teacher"]
            S = logits[label]
            r["agree_with_teacher"] = S.argmax(1).eq(T.argmax(1)).float().mean().item() * 100
            r["spearman_with_teacher"] = spearman_rows(S, T)
            r["kl_raw_T4"] = distillation_loss(S, T, 4.0, standardize=False).item() / 16.0
            r["kl_zscore_tau2"] = distillation_loss(S, T, 2.0, standardize=True).item() / 4.0
        results[label] = r
        print("%-12s top1 %.2f  top5 %.2f  ECE %.2f  logit std %.2f" %
              (label, r["top1"], r["top5"], r["ece"], r["logit_std"]))

    labs = list(logits)
    agreement = {}
    for a, b in itertools.combinations(labs, 2):
        pa, pb = logits[a].argmax(1), logits[b].argmax(1)
        both = pa.eq(pb)
        agreement["%s vs %s" % (a, b)] = dict(
            agree=both.float().mean().item() * 100,
            only_first_correct=((pa == labels) & (pb != labels)).sum().item(),
            only_second_correct=((pb == labels) & (pa != labels)).sum().item(),
        )
    tag = args.tag or args.dataset
    save_json(dict(dataset=args.dataset, models=results, agreement=agreement),
              os.path.join(args.out_dir, "eval_%s.json" % tag))
    np.savez_compressed(os.path.join(args.out_dir, "logits_%s.npz" % tag), labels=y,
                        **{k.replace("+", "_plus_"): v.cpu().numpy().astype(np.float16) for k, v in logits.items()})
    print("Saved results/eval_%s.json" % tag)
    for k, v in agreement.items():
        print(k, v)


if __name__ == "__main__":
    main()
