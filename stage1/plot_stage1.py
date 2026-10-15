"""Stage-1 figures. python stage1/plot_stage1.py --results results --out results/plots"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# color follows the criterion (fixed slots, never by rank); gray for the unmasked baselines
COLORS = {
    "student_attn": "#2a78d6", "random": "#eb6834", "student_attn_inv": "#1baf7a",
    "tau_match:student_attn": "#eda100", "teacher_attn": "#e87ba4", "teacher_attn_noisy": "#008300",
    "rollout": "#4a3aa7", "student_attn_noce": "#e34948",
    "none": "#555555", "tau_const:2": "#8a8a8a", "tau_const:4": "#b0b0b0",
}
STYLES = {"tau_const:2": "--", "tau_const:4": ":", "none": "-."}
MARKERS = {"student_attn": "o", "random": "s", "student_attn_inv": "v", "tau_match:student_attn": "D",
           "teacher_attn": "^", "teacher_attn_noisy": "P", "rollout": "X", "student_attn_noce": "*",
           "none": "h", "tau_const:2": "<", "tau_const:4": ">"}
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e3dd"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 5,
    "legend.frameon": False, "axes.titleweight": "bold", "axes.titlesize": 10,
})


def color(c):
    return COLORS.get(c, "#333333")


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def load(results):
    runs = []
    for cfgp in sorted(glob.glob(os.path.join(results, "*", "*", "config.json"))):
        d = os.path.dirname(cfgp)
        if "/_" in d:  # helper runs (_equiv, _tp)
            continue
        cfg = json.load(open(cfgp))
        m = read_jsonl(os.path.join(d, "metrics.jsonl"))
        if not m:
            continue
        runs.append(dict(group=cfg["group"], crit=cfg["criterion"], seed=cfg["seed"], metrics=m,
                         train=read_jsonl(os.path.join(d, "train_log.jsonl")),
                         masked=not (cfg["criterion"] == "none" or cfg["criterion"].startswith("tau_"))))
    return runs


def series(run, key):
    xs = [m["epoch"] for m in run["metrics"] if m.get(key) is not None]
    ys = [m[key] for m in run["metrics"] if m.get(key) is not None]
    return np.array(xs), np.array(ys, dtype=float)


def train_by_epoch(run, key):
    acc = defaultdict(list)
    for r in run["train"]:
        if r.get(key) is not None:
            acc[r["epoch"] + 1].append(r[key])
    ep = sorted(acc)
    return np.array(ep), np.array([np.mean(acc[e]) for e in ep])


def label(run, runs):
    multi_group = len({r["group"] for r in runs}) > 1
    return f"{run['crit']} ({run['group']})" if multi_group else run["crit"]


def legend(ax):
    h, l = ax.get_legend_handles_labels()
    if len(h) >= 2:
        ax.legend(fontsize=7, loc="best")


def line_plot(ax, runs, key, title, ylabel, getter=series, only=None):
    for r in runs:
        if only and r["crit"] not in only:
            continue
        x, y = getter(r, key)
        if len(x) == 0:
            continue
        ax.plot(x, y, color=color(r["crit"]), ls=STYLES.get(r["crit"], "-"),
                marker=MARKERS.get(r["crit"], "o"), markersize=4, label=label(r, runs),
                alpha=1.0 if r["seed"] == 0 else 0.6)
    ax.set_title(title, loc="left")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    legend(ax)


def fig_tradeoff(runs, out):
    """x: masked-teacher val top-1 drop vs full teacher (mean over probes after epoch 0; 0 for unmasked);
    y: final student val top-1. Mean ± range over seeds per criterion."""
    per = defaultdict(dict)  # crit -> seed -> (x, y); same (crit, seed) in two groups -> first kept
    for r in runs:
        ms = [m for m in r["metrics"] if m["epoch"] > 0] or r["metrics"]
        x = float(np.mean([m["val_top1_drop_tmask"] for m in ms])) if r["masked"] else 0.0
        y = r["metrics"][-1]["val_top1_full"]
        per[r["crit"]].setdefault(r["seed"], (x, y))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for c, d in sorted(per.items()):
        xs, ys = np.array([v[0] for v in d.values()]), np.array([v[1] for v in d.values()])
        ax.errorbar(xs.mean(), ys.mean(),
                    xerr=[[xs.mean() - xs.min()], [xs.max() - xs.mean()]],
                    yerr=[[ys.mean() - ys.min()], [ys.max() - ys.mean()]],
                    fmt=MARKERS.get(c, "o"), color=color(c), ms=8, capsize=3, elinewidth=1.2,
                    mec="white", mew=1.0, label=f"{c} (n={len(d)})")
        if xs.mean() != 0.0:  # unmasked criteria sit together at x=0; the legend names them
            ax.annotate(c, (xs.mean(), ys.mean()), textcoords="offset points", xytext=(6, 5), fontsize=7, color=INK)
    ax.set_xlabel("masked-teacher top-1 drop vs full teacher (val probe, pts)")
    ax.set_ylabel("final student val top-1 (%)")
    ax.set_title("Teacher degradation vs student accuracy", loc="left")
    legend(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "1_tradeoff.png"))
    plt.close(fig)


def two_panel(runs, out, name, keys, titles, ylabels, getter=series, only=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, k, t, yl in zip(axes, keys, titles, ylabels):
        line_plot(ax, runs, k, t, yl, getter, only)
    fig.tight_layout()
    fig.savefig(os.path.join(out, name))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "plots")
    os.makedirs(out, exist_ok=True)
    runs = load(a.results)
    if not runs:
        raise SystemExit(f"no runs with metrics under {a.results}")

    fig_tradeoff(runs, out)
    two_panel(runs, out, "2_attention_alignment.png", ["val_attn_spearman", "val_attn_masked_iou"],
              ["Student–teacher attention Spearman", "IoU of masked (bottom) patch sets"],
              ["Spearman ρ", "IoU"])
    two_panel(runs, out, "3_kl.png", ["val_kl_tmask_tfull", "val_kl_tmask_student"],
              ["KL(masked T ‖ full T)", "KL(masked T ‖ S)"], ["nats", "nats"])
    ent_only = {"student_attn", "tau_match:student_attn", "tau_const:2", "tau_const:4", "none"}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    line_plot(ax, runs, "entropy_teacher_used", "Entropy of the teacher target used for KD (train)",
              "nats", train_by_epoch, ent_only)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "4_teacher_entropy.png"))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    line_plot(ax, runs, "grad_cos_kd_ce", "Logit-gradient cosine, KD vs CE (train)", "cosine", train_by_epoch)
    ax.axhline(0, color=MUTED, lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "5_grad_cos.png"))
    plt.close(fig)
    two_panel(runs, out, "6_noce_alignment.png", ["val_attn_spearman", "val_attn_masked_iou"],
              ["Alignment Spearman: with vs without CE", "Masked-set IoU: with vs without CE"],
              ["Spearman ρ", "IoU"], only={"student_attn", "student_attn_noce"})
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
