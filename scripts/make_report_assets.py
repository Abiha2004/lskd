"""Turn the training/evaluation outputs into the tables and figures used by
report/final_report.tex. Missing runs are shown as "--" so the report always compiles.

  python scripts/make_report_assets.py --runs runs --results results --out report

Writes report/generated/*.tex and report/figures/*.pdf
ATTRIBUTION: IMPLEMENTED BY OUR TEAM.
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Results reported by Sun et al. (2024), Table 2, ResNet56 -> ResNet20 on CIFAR-100 (mean of 4 runs)
PAPER = {"teacher": 72.34, "student": 69.06, "kd": 70.66, "kd_ls": 71.43}


def load_runs(run_root):
    runs = {}
    if not os.path.isdir(run_root):
        return runs
    for name in sorted(os.listdir(run_root)):
        d = os.path.join(run_root, name)
        s = os.path.join(d, "summary.json")
        h = os.path.join(d, "history.csv")
        if not os.path.exists(h):
            continue
        r = {"name": name}
        if os.path.exists(s):
            with open(s) as f:
                r.update(json.load(f))
        with open(h) as f:
            r["history"] = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
        r["complete"] = os.path.exists(s)
        runs[name] = r
    return runs


def fmt(x, nd=2):
    return "--" if x is None else ("%.*f" % (nd, x))


def acc(runs, name, key="final_acc"):
    r = runs.get(name)
    return r.get(key) if r and r.get("complete") else None


def delta(a, b):
    return None if a is None or b is None else a - b


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def table_repro(runs, gen):
    t_acc = None
    for n in ("c100_kd", "c100_kd_ls"):
        if runs.get(n, {}).get("teacher_acc") is not None:
            t_acc = runs[n]["teacher_acc"]
    rows = [
        ("Teacher ResNet-56 (pretrained)", PAPER["teacher"], t_acc, None),
        ("Student ResNet-20, CE only", PAPER["student"], acc(runs, "c100_student_ce"), acc(runs, "c100_student_ce", "best_acc")),
        ("Vanilla KD ($T{=}4$, $\\lambda_{KD}{=}0.9$)", PAPER["kd"], acc(runs, "c100_kd"), acc(runs, "c100_kd", "best_acc")),
        ("KD + LS ($\\tau{=}2$, $\\lambda_{KD}{=}9$)", PAPER["kd_ls"], acc(runs, "c100_kd_ls"), acc(runs, "c100_kd_ls", "best_acc")),
    ]
    body = "\n".join("%s & %.2f & %s & %s & %s \\\\" % (n, p, fmt(o), fmt(b), fmt(delta(o, p)))
                     for n, p, o, b in rows)
    g_paper = PAPER["kd_ls"] - PAPER["kd"]
    g_ours = delta(acc(runs, "c100_kd_ls"), acc(runs, "c100_kd"))
    body += "\n\\midrule\nGain of LS over KD ($\\Delta$) & %.2f & %s & %s & %s \\\\" % (
        g_paper, fmt(g_ours), fmt(delta(acc(runs, "c100_kd_ls", "best_acc"), acc(runs, "c100_kd", "best_acc"))),
        fmt(delta(g_ours, g_paper)))
    write(os.path.join(gen, "tab_repro.tex"), body + "\n")


def table_ablation(runs, gen):
    cells = {("raw", 0.9): "abl_raw_w0.9", ("raw", 9): "abl_raw_w9",
             ("z", 0.9): "abl_z_w0.9", ("z", 9): "c100_kd_ls"}
    lines = []
    for w in (0.9, 9):
        raw, z = acc(runs, cells[("raw", w)]), acc(runs, cells[("z", w)])
        lines.append("$\\lambda_{KD}=%s$ & %s & %s & %s \\\\" % (w, fmt(raw), fmt(z), fmt(delta(z, raw))))
    r9, r09 = acc(runs, cells[("raw", 9)]), acc(runs, cells[("raw", 0.9)])
    z9, z09 = acc(runs, cells[("z", 9)]), acc(runs, cells[("z", 0.9)])
    lines.append("\\midrule\nEffect of $\\lambda_{KD}$: $0.9\\rightarrow 9$ & %s & %s & \\\\" % (
        fmt(delta(r9, r09)), fmt(delta(z9, z09))))
    write(os.path.join(gen, "tab_ablation.tex"), "\n".join(lines) + "\n")
    return {"raw09": r09, "raw9": r9, "z09": z09, "z9": z9}


def table_c10(runs, gen):
    t = acc(runs, "c10_teacher_resnet56")
    kd, ls = acc(runs, "c10_kd"), acc(runs, "c10_kd_ls")
    c100kd, c100ls = acc(runs, "c100_kd"), acc(runs, "c100_kd_ls")
    lines = [
        "Teacher ResNet-56 (trained by us) & %s & %s \\\\" % (fmt(t), fmt(runs.get("c100_kd", {}).get("teacher_acc"))),
        "Vanilla KD & %s & %s \\\\" % (fmt(kd), fmt(c100kd)),
        "KD + LS & %s & %s \\\\" % (fmt(ls), fmt(c100ls)),
        "\\midrule",
        "Gain of LS ($\\Delta$, pp) & %s & %s \\\\" % (fmt(delta(ls, kd)), fmt(delta(c100ls, c100kd))),
        "Error-rate reduction by LS (\\%%) & %s & %s \\\\" % (
            fmt(None if kd is None or ls is None else 100 * ((100 - kd) - (100 - ls)) / (100 - kd)),
            fmt(None if c100kd is None or c100ls is None else 100 * ((100 - c100kd) - (100 - c100ls)) / (100 - c100kd))),
    ]
    write(os.path.join(gen, "tab_c10.tex"), "\n".join(lines) + "\n")


def table_tau(runs, gen):
    lines = []
    for tau, name in ((1, "tau1_kd_ls"), (2, "c100_kd_ls"), (4, "tau4_kd_ls")):
        lines.append("$\\tau=%d$ & %s & %s \\\\" % (tau, fmt(acc(runs, name)), fmt(acc(runs, name, "best_acc"))))
    write(os.path.join(gen, "tab_tau.tex"), "\n".join(lines) + "\n")


def table_runs(runs, gen):
    lines = []
    for n, r in runs.items():
        if n.endswith("_smoke"):
            continue
        lines.append("\\texttt{%s} & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            n.replace("_", "\\_"), r.get("method", "--").replace("_", "\\_"), fmt(r.get("temperature"), 1),
            fmt(r.get("kd_weight"), 1), str(r.get("epochs", len(r["history"]))), fmt(r.get("final_acc")),
            fmt(r.get("best_acc")), fmt(r.get("train_minutes"), 0)))
    write(os.path.join(gen, "tab_runs.tex"), "\n".join(lines) + "\n" if lines else "\\multicolumn{8}{c}{no runs found} \\\\\n")


def table_diag(res_dir, gen):
    path = os.path.join(res_dir, "eval_cifar100.json")
    if not os.path.exists(path):
        write(os.path.join(gen, "tab_diag.tex"), "\\multicolumn{9}{c}{run evaluate.py first} \\\\\n")
        return None
    with open(path) as f:
        ev = json.load(f)
    lines = []
    for label, r in ev["models"].items():
        lines.append("%s & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            label.replace("+", "+"), fmt(r["top1"]), fmt(r["top5"]), fmt(r["ece"]), fmt(r["logit_std"]),
            fmt(r.get("agree_with_teacher")), fmt(r.get("spearman_with_teacher"), 3),
            fmt(r.get("kl_raw_T4"), 3), fmt(r.get("kl_zscore_tau2"), 3)))
    write(os.path.join(gen, "tab_diag.tex"), "\n".join(lines) + "\n")
    return ev


def fig_curves(runs, figs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, label in (("c100_student_ce", "Student CE"), ("c100_kd", "Vanilla KD"), ("c100_kd_ls", "KD + LS")):
        if name in runs:
            h = runs[name]["history"]
            ep = [r["epoch"] + 1 for r in h]
            axes[0].plot(ep, [r["test_acc"] for r in h], label=label)
            axes[1].plot(ep, [r["train_ce"] / max(runs[name].get("ce_weight", 1.0) or 1.0, 1e-9)
                              if runs[name].get("method") != "ce" else r["train_ce"] for r in h], label=label)
    axes[0].set_ylabel("test top-1 accuracy (%)")
    axes[0].set_ylim(bottom=30)
    axes[1].set_ylabel("training cross-entropy (unweighted)")
    for a in axes:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "curves_c100.pdf"))
    fig.savefig(os.path.join(figs, "curves_c100.png"), dpi=150)
    plt.close(fig)


def fig_ablation(ab, figs):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    xs = np.arange(2)
    raw = [ab["raw09"] or 0, ab["raw9"] or 0]
    z = [ab["z09"] or 0, ab["z9"] or 0]
    ax.bar(xs - 0.18, raw, 0.36, label="raw logits (vanilla KD)", color="#8c8c8c")
    ax.bar(xs + 0.18, z, 0.36, label="Z-scored logits (LS)", color="#4C72B0")
    for x, v in zip(list(xs - 0.18) + list(xs + 0.18), raw + z):
        if v:
            ax.text(x, v + 0.1, "%.2f" % v, ha="center", fontsize=8)
    vals = [v for v in raw + z if v]
    if vals:
        ax.set_ylim(min(vals) - 2, max(vals) + 1)
    ax.set_xticks(xs)
    ax.set_xticklabels(["$\\lambda_{KD}=0.9$", "$\\lambda_{KD}=9$"])
    ax.set_ylabel("test top-1 (%)")
    ax.set_title("Ablation at temperature 2 (ResNet56$\\rightarrow$ResNet20)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "ablation.pdf"))
    fig.savefig(os.path.join(figs, "ablation.png"), dpi=150)
    plt.close(fig)


def fig_logits(res_dir, figs):
    path = os.path.join(res_dir, "logits_cifar100.npz")
    if not os.path.exists(path):
        return
    d = np.load(path)
    keys = [k for k in d.files if k != "labels"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for k in keys:
        z = d[k].astype(np.float32)
        axes[0].hist(z.std(1, ddof=1), bins=60, alpha=0.55, label=k.replace("_plus_", "+"))
        axes[1].hist(z.max(1) - z.mean(1), bins=60, alpha=0.55, label=k.replace("_plus_", "+"))
    axes[0].set_xlabel("per-sample logit standard deviation")
    axes[1].set_xlabel("per-sample (max logit - mean logit)")
    for a in axes:
        a.set_ylabel("# test images")
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "logit_stats.pdf"))
    fig.savefig(os.path.join(figs, "logit_stats.png"), dpi=150)
    plt.close(fig)


def numbers(runs, ev, gen):
    m = {
        "KDacc": acc(runs, "c100_kd"), "LSacc": acc(runs, "c100_kd_ls"),
        "CEacc": acc(runs, "c100_student_ce"),
        "LSgain": delta(acc(runs, "c100_kd_ls"), acc(runs, "c100_kd")),
        "CTENkd": acc(runs, "c10_kd"), "CTENls": acc(runs, "c10_kd_ls"),
        "CTENgain": delta(acc(runs, "c10_kd_ls"), acc(runs, "c10_kd")),
    }
    if ev:
        for label, key in (("Vanilla KD", "KD"), ("KD+LS", "LS"), ("Teacher", "T")):
            if label in ev["models"]:
                m[key + "logitstd"] = ev["models"][label]["logit_std"]
    lines = ["\\renewcommand{\\%s}{%s}" % (k, fmt(v)) for k, v in m.items() if v is not None]
    write(os.path.join(gen, "numbers.tex"), "\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="report")
    a = p.parse_args()
    gen, figs = os.path.join(a.out, "generated"), os.path.join(a.out, "figures")
    os.makedirs(gen, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    runs = load_runs(a.runs)
    print("Found runs:", ", ".join("%s%s" % (k, "" if v["complete"] else " (incomplete)") for k, v in runs.items()))
    table_repro(runs, gen)
    ab = table_ablation(runs, gen)
    table_c10(runs, gen)
    table_tau(runs, gen)
    table_runs(runs, gen)
    ev = table_diag(a.results, gen)
    fig_curves(runs, figs)
    fig_ablation(ab, figs)
    fig_logits(a.results, figs)
    numbers(runs, ev, gen)
    print("Wrote tables to %s and figures to %s" % (gen, figs))


if __name__ == "__main__":
    main()
