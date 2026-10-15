"""Probe-set measurements (section 6) and results/summary.csv."""
import csv
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from common import cls_attn_scores, teacher_full, teacher_masked
from data_gpu import gpu_eval
from masking import Ctx


class ProbeSet:
    """Fixed, un-augmented probe images (center crop 160). Selection depends only on probe_seed."""

    def __init__(self, train_split, val_split, n_train, n_val, probe_seed=1234, n_vis=16):
        rs = np.random.RandomState(probe_seed)
        self.parts = {}
        for name, sp, n in (("train", train_split, n_train), ("val", val_split, n_val)):
            rows = np.sort(rs.permutation(len(sp))[:min(n, len(sp))])
            x, y = sp.get(rows)
            self.parts[name] = (torch.from_numpy(x).pin_memory(), torch.from_numpy(y), rows)
        self.n_vis = n_vis

    def chunks(self, part, bs=250):
        x, y, _ = self.parts[part]
        for i in range(0, len(y), bs):
            yield (x[i:i + bs].cuda(non_blocking=True).permute(0, 3, 1, 2),
                   y[i:i + bs].cuda(non_blocking=True))


def _kl(p_log, q_log):
    return (p_log.exp() * (p_log - q_log)).sum(1)


def _entropy(logp):
    return -(logp.exp() * logp).sum(1)


def _spearman(a, b):
    ra = a.argsort(1).argsort(1).float()
    rb = b.argsort(1).argsort(1).float()
    ra = ra - ra.mean(1, keepdim=True)
    rb = rb - rb.mean(1, keepdim=True)
    return (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1) + 1e-12)


def _masked_iou(a, b, keep):
    n_mask = a.shape[1] - keep
    ma = torch.zeros_like(a, dtype=torch.bool).scatter_(1, torch.topk(-a, n_mask).indices, True)
    mb = torch.zeros_like(b, dtype=torch.bool).scatter_(1, torch.topk(-b, n_mask).indices, True)
    return (ma & mb).sum(1).float() / (ma | mb).sum(1).float()


def auroc(scores, labels):
    """P(score_pos > score_neg) with ties counted half (Mann-Whitney)."""
    from scipy.stats import rankdata
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _distribution_metrics(prefix, tf, tm, s, y):
    out = {}
    lf, lm, ls = (F.log_softmax(t.float(), 1) for t in (tf, tm, s))
    for name, l, logits in (("tfull", lf, tf), ("tmask", lm, tm), ("student", ls, s)):
        out[f"{prefix}top1_{name}"] = (logits.argmax(1) == y).float().mean().item() * 100
        out[f"{prefix}entropy_{name}"] = _entropy(l).mean().item()
        out[f"{prefix}maxprob_{name}"] = l.exp().max(1).values.mean().item()
    out[f"{prefix}kl_tmask_tfull"] = _kl(lm, lf).mean().item()
    out[f"{prefix}kl_tmask_student"] = _kl(lm, ls).mean().item()
    out[f"{prefix}kl_tfull_student"] = _kl(lf, ls).mean().item()
    out[f"{prefix}agree_tmask_student"] = (tm.argmax(1) == s.argmax(1)).float().mean().item() * 100
    out[f"{prefix}agree_tfull_student"] = (tf.argmax(1) == s.argmax(1)).float().mean().item() * 100
    # non-target distributions: drop the true class, renormalize
    keep = torch.ones_like(lf, dtype=torch.bool).scatter_(1, y[:, None], False)
    nt_m = F.log_softmax(tm.float()[keep].view(len(y), -1), 1)
    nt_f = F.log_softmax(tf.float()[keep].view(len(y), -1), 1)
    out[f"{prefix}kl_nontarget_tmask_tfull"] = _kl(nt_m, nt_f).mean().item()
    drop = (lf.exp().max(1).values - lm.exp().max(1).values).cpu().numpy()
    wrong = (tm.argmax(1) != y).cpu().numpy()
    out[f"{prefix}auroc_maxprob_drop_tmask_wrong"] = auroc(drop, wrong)
    return out


@torch.no_grad()
def run_probes(students, teacher, probe, keep, progress, noise_init, amp_dtype, epoch, vis_dir_fn=None):
    """students: list of objects with .spec (StudentSpec) and .model. Returns {dirname: metrics}."""
    results = {}
    for st in students:
        st.model.eval()
    acc = {st.spec.dirname: {"train": [], "val": []} for st in students}
    for part in ("train", "val"):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(4321 + epoch)  # fixed random / noisy masks per probe
        for xu8, y in probe.chunks(part):
            x = gpu_eval(xu8)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                tf, t_attns = teacher_full(teacher, x, need_attn=True)
                t_sc = cls_attn_scores(t_attns[-1]).float()
                for st in students:
                    s, s_attn = st.model(x)
                    crit = st.spec.probe_criterion()
                    ctx = Ctx(s_attn, t_attns, keep, progress, gen, noise_init)
                    k_idx = crit(ctx)
                    tm = teacher_masked(teacher, x, k_idx)
                    s_sc = cls_attn_scores(s_attn).float()
                    kmask = torch.zeros_like(s_sc, dtype=torch.bool).scatter_(1, k_idx, True)
                    acc[st.spec.dirname][part].append(
                        (tf.float(), tm.float(), s.float(), y, s_sc, t_sc, kmask))
    for st in students:
        d = st.spec.dirname
        m = {"epoch": epoch}
        for part in ("train", "val"):
            tf, tm, s, y, s_sc, t_sc, kmask = (torch.cat(z) for z in zip(*acc[d][part]))
            pre = f"{part}_"
            m.update(_distribution_metrics(pre, tf, tm, s, y))
            m[f"{pre}attn_spearman"] = _spearman(s_sc, t_sc).mean().item()
            m[f"{pre}attn_masked_iou"] = _masked_iou(s_sc, t_sc, keep).mean().item()
            m[f"{pre}top1_drop_tmask"] = m[f"{pre}top1_tfull"] - m[f"{pre}top1_tmask"]
            if part == "val" and vis_dir_fn is not None:
                n = probe.n_vis
                np.savez_compressed(
                    os.path.join(vis_dir_fn(st), f"masks_ep{epoch:03d}.npz"),
                    probe_rows=probe.parts["val"][2][:n], labels=y[:n].cpu().numpy(),
                    student_attn=s_sc[:n].cpu().numpy(), teacher_attn=t_sc[:n].cpu().numpy(),
                    keep_mask=kmask[:n].cpu().numpy(),
                    pred_tmask=tm[:n].argmax(1).cpu().numpy(), pred_student=s[:n].argmax(1).cpu().numpy())
        results[d] = m
    for st in students:
        st.model.train()
    return results


@torch.no_grad()
def evaluate_top1(model, split, amp_dtype, bs=500):
    from data_gpu import Loader
    model.eval()
    correct, n = 0, 0
    for xu8, y in Loader(split, bs, seed=0, epoch=0, shuffle=False, drop_last=False):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(gpu_eval(xu8))
        out = out[0] if isinstance(out, tuple) else out
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    model.train()
    return 100.0 * correct / max(n, 1)


def _last_jsonl(path):
    last = None
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    last = json.loads(line)
    return last


SUMMARY_KEYS = ["epoch", "val_top1_full", "best_val_top1", "val_top1_tfull", "val_top1_tmask",
                "val_top1_drop_tmask", "val_kl_tmask_tfull", "val_kl_tmask_student", "val_attn_spearman",
                "val_attn_masked_iou", "val_entropy_tmask", "val_auroc_maxprob_drop_tmask_wrong",
                "teacher_tokens_per_img", "teacher_gflops_per_img", "teacher_gflops_total"]


def write_summary(results_root):
    rows = []
    for mpath in sorted(glob.glob(os.path.join(results_root, "*", "*", "metrics.jsonl"))):
        sdir = os.path.dirname(mpath)
        last = _last_jsonl(mpath)
        if last is None:
            continue
        cfg = json.load(open(os.path.join(sdir, "config.json")))
        row = {"group": cfg["group"], "student": cfg["criterion"], "seed": cfg["seed"]}
        row.update({k: last.get(k) for k in SUMMARY_KEYS})
        rows.append(row)
    if not rows:
        return None
    out = os.path.join(results_root, "summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out
