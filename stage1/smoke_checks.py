"""Automatic smoke-test checks (section 8). Called by scripts/smoke_test.sh.

python stage1/smoke_checks.py <check> --data_dir D --out O ...
checks: equiv_orig | equiv_group | teacher | monotone | tau_match | outputs | throughput | estimate
Every check prints a line 'CHECK <name> PASS|FAIL|SKIP <details>' and appends it to logs/smoke_report.txt.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PY = sys.executable


def report(name, status, detail):
    line = f"CHECK {name:<12} {status:<4} {detail}"
    print(line, flush=True)
    os.makedirs("logs", exist_ok=True)
    with open("logs/smoke_report.txt", "a") as f:
        f.write(line + "\n")
    return status != "FAIL"


def _setup(data_dir):
    import torch
    from common import build_teacher
    from data_gpu import Split, load_meta
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    meta = load_meta(data_dir)
    teacher = build_teacher(meta["classes"]).cuda()
    return meta, teacher, Split(data_dir, "val")


def equiv_orig(a):
    """Our student_attn mask + masked-teacher KD loss vs the original losses.DistillationLoss path (fp32)."""
    import numpy as np
    import torch
    from common import build_student, kd_loss, label_smoothing_ce, orig_losses, teacher_full, teacher_masked
    from data_gpu import gpu_eval
    from masking import CRITERIA, Ctx
    meta, teacher, val = _setup(a.data_dir)
    xu8, y = val.get(np.arange(min(64, len(val))))
    x = gpu_eval(torch.from_numpy(xu8).cuda().permute(0, 3, 1, 2))
    y = torch.from_numpy(y).cuda()
    torch.manual_seed(0)
    student = build_student().cuda().eval()  # eval: no stochastic depth, so both paths see one forward
    with torch.no_grad():
        out, attn = student(x)

    captured = {}

    class Spy(torch.nn.Module):
        def __init__(self, t):
            super().__init__()
            self.t = t

        def forward(self, inp, len_keep, maskedkd):
            captured["idx"] = len_keep.clone()
            return self.t(inp, len_keep, maskedkd)

    ok = True
    for keep in (a.keep, 50):
        orig = orig_losses.DistillationLoss(label_smoothing_ce(0.1), Spy(teacher), "soft", 0.5, 1.0, keep, True)
        with torch.no_grad():
            l_orig = orig(x, out, y, attn).item()
            idx = CRITERIA["student_attn"](Ctx(attn, None, keep))
            t = teacher_masked(teacher, x, idx)
            l_ours = (0.5 * label_smoothing_ce(0.1)(out, y) + 0.5 * kd_loss(out, t, 1.0)).item()
        same_idx = torch.equal(idx, captured["idx"])
        rel = abs(l_ours - l_orig) / max(abs(l_orig), 1e-12)
        good = same_idx and rel < 1e-4
        ok &= report("equiv_orig", "PASS" if good else "FAIL",
                     f"keep={keep} mask_idx_equal={same_idx} loss_orig={l_orig:.6f} loss_ours={l_ours:.6f} rel_err={rel:.2e}")
    with torch.no_grad():
        l_plain = teacher(x, None, False)
        l_attn, _ = teacher_full(teacher, x, need_attn=True)
    err = (l_plain - l_attn).abs().max().item()
    ok &= report("equiv_tattn", "PASS" if err < 1e-4 else "FAIL",
                 f"teacher forward with attention vs original forward: max|dlogit|={err:.2e}")
    return ok


def equiv_group(a):
    """First-step loss of students trained alone == the same students trained inside the full group."""
    tmp = os.path.join(a.out, "_equiv")
    base = [PY, "-W", "ignore", os.path.join(HERE, "group_train.py"), "--config", a.config,
            "--data_dir", a.data_dir, "--deterministic", "--no_amp", "--batch_size", "64"]

    def run(students, tag):
        path = os.path.join(tmp, f"{tag}.json")
        os.makedirs(tmp, exist_ok=True)
        cmd = base + ["--out", os.path.join(tmp, tag), "--group", tag, "--first_step_dump", path]
        if students:
            cmd += ["--students"] + students
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        return json.load(open(path))

    group = run(None, "group")
    ok = True
    for name in a.solo:
        solo = run([name], "solo_" + name.replace(":", "_"))
        lg, ls = group[name], solo[name]
        rel = abs(lg - ls) / max(abs(ls), 1e-12)
        ok &= report("equiv_group", "PASS" if rel < 1e-4 else "FAIL",
                     f"{name}: group={lg:.6f} solo={ls:.6f} rel_err={rel:.2e}")
    return ok


def _masked_top1(teacher, val, keeps, mode, amp=True):
    import numpy as np
    import torch
    from common import cls_attn_scores, teacher_full, teacher_masked
    from data_gpu import gpu_eval
    corr = {k: 0 for k in keeps}
    full_c, n = 0, 0
    g = torch.Generator(device="cuda")
    g.manual_seed(0)
    for i in range(0, len(val), 250):
        xu8, y = val.get(np.arange(i, min(i + 250, len(val))))
        x = gpu_eval(torch.from_numpy(xu8).cuda().permute(0, 3, 1, 2))
        y = torch.from_numpy(y).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            tf, attns = teacher_full(teacher, x, need_attn=True)
            full_c += (tf.argmax(1) == y).sum().item()
            if mode == "random":
                sc = torch.rand(x.shape[0], 100, generator=g, device="cuda")  # nested masks across keeps
            else:
                sc = cls_attn_scores(attns[-1]).float()
            for k in keeps:
                tm = teacher_masked(teacher, x, torch.topk(sc, k).indices)
                corr[k] += (tm.argmax(1) == y).sum().item()
        n += len(y)
    return 100 * full_c / n, {k: 100 * c / n for k, c in corr.items()}


def teacher_check(a):
    meta, teacher, val = _setup(a.data_dir)
    full, _ = _masked_top1(teacher, val, [], "random")
    detail = f"teacher val top-1 @160 (100-way head) = {full:.2f}% on {len(val)} images (threshold 75%)"
    if meta.get("synthetic"):
        return report("teacher", "SKIP", detail + " [synthetic data: not meaningful]")
    return report("teacher", "PASS" if full >= 75 else "FAIL", detail)


def monotone(a):
    meta, teacher, val = _setup(a.data_dir)
    ok = True
    for mode in ("random", "teacher_attn"):
        full, accs = _masked_top1(teacher, val, [100, 50, 30], mode)
        mono = accs[100] >= accs[50] >= accs[30] and accs[100] > accs[30]
        detail = f"{mode} masks: keep100={accs[100]:.2f} keep50={accs[50]:.2f} keep30={accs[30]:.2f} (full={full:.2f})"
        if meta.get("synthetic"):
            report("monotone", "SKIP", detail + " [synthetic data]")
        else:
            ok &= report("monotone", "PASS" if mono else "FAIL", detail)
    return ok


def tau_match(a):
    gaps = []
    for p in glob.glob(os.path.join(a.out, "*", "tau_match_*", "train_log.jsonl")):
        gaps += [abs(json.loads(l)["entropy_gap"]) for l in open(p) if l.strip()]
    if not gaps:
        return report("tau_match", "FAIL", "no tau_match train_log entries found")
    m = sum(gaps) / len(gaps)
    return report("tau_match", "PASS" if m < 0.01 else "FAIL",
                  f"mean |H(matched full T) - H(masked T)| = {m:.2e} nats over {len(gaps)} log windows")


def outputs(a):
    import math
    ok = True
    sdirs = sorted(d for d in glob.glob(os.path.join(a.out, "*", "*")) if os.path.isdir(d) and "_equiv" not in d)
    missing, bad = [], []
    for d in sdirs:
        for f in ("metrics.jsonl", "train_log.jsonl", "config.json", "ckpt.pth"):
            if not os.path.exists(os.path.join(d, f)):
                missing.append(os.path.join(d, f))
        if not glob.glob(os.path.join(d, "masks_*.npz")):
            missing.append(os.path.join(d, "masks_*.npz"))
        for f in ("metrics.jsonl", "train_log.jsonl"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                for l in open(p):
                    for k, v in json.loads(l).items():
                        if isinstance(v, float) and math.isinf(v) or (isinstance(v, float) and math.isnan(v)
                                                                      and "auroc" not in k):
                            bad.append(f"{p}:{k}")
                    if json.loads(l).get("nonfinite_steps", 0) > 0:
                        bad.append(f"{p}:nonfinite_steps")
    if not os.path.exists(os.path.join(a.out, "summary.csv")):
        missing.append("summary.csv")
    ok &= report("nan_inf", "PASS" if not bad else "FAIL",
                 f"{len(sdirs)} students, non-finite values: {bad[:5] if bad else 'none'}")
    ok &= report("files", "PASS" if not missing else "FAIL", f"missing: {missing[:5] if missing else 'none'}")
    plot_dir = os.path.join(a.out, "plots")
    r = subprocess.run([PY, os.path.join(HERE, "plot_stage1.py"), "--results", a.out, "--out", plot_dir],
                       capture_output=True, text=True)
    pngs = glob.glob(os.path.join(plot_dir, "*.png"))
    ok &= report("plot", "PASS" if r.returncode == 0 and len(pngs) >= 6 else "FAIL",
                 f"{len(pngs)} figures in {plot_dir}" + ("" if r.returncode == 0 else f" | {r.stderr[-400:]}"))
    return ok


def throughput(a):
    """50 steps per real group config; auto-halve batch if peak memory exceeds the limit."""
    import yaml
    os.makedirs("logs", exist_ok=True)
    recs, overrides, ok = [], {}, True
    for cfgp in a.configs:
        cfg = yaml.safe_load(open(cfgp))
        bs = cfg.get("batch_size", 256)
        while True:
            out = os.path.join(a.out, "_tp", cfg["group"] + ".json")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            cmd = [PY, "-W", "ignore", os.path.join(HERE, "group_train.py"), "--config", cfgp,
                   "--data_dir", a.data_dir, "--out", os.path.join(a.out, "_tp"), "--batch_size", str(bs),
                   "--max_steps", str(a.steps), "--throughput_out", out]
            r = subprocess.run(cmd, capture_output=True, text=True)
            oom = r.returncode != 0 and "out of memory" in (r.stderr + r.stdout).lower()
            if r.returncode != 0 and not oom:
                return report("throughput", "FAIL", f"{cfg['group']} crashed: {r.stderr[-600:]}")
            rec = json.load(open(out)) if not oom else None
            if (oom or rec["peak_mem_gb"] > a.mem_limit_gb) and bs > 32:
                print(f"  {cfg['group']}: batch {bs} peak "
                      f"{'OOM' if oom else '%.1fGB' % rec['peak_mem_gb']} > {a.mem_limit_gb}GB -> batch {bs // 2}")
                bs //= 2
                continue
            break
        if bs != cfg.get("batch_size", 256):
            overrides[cfg["group"]] = bs
        rec["config"] = cfgp
        rec["epochs"] = cfg.get("epochs", 40)
        recs.append(rec)
        report("throughput", "PASS", f"{cfg['group']}: {rec['step_time'] * 1000:.0f} ms/step @batch {bs}, "
                                     f"peak {rec['peak_mem_gb']:.1f}GB, compiled={rec['compiled']}")
    with open("logs/batch_override.json", "w") as f:
        json.dump(overrides, f)
    with open(os.path.join(a.out, "_tp", "all.json"), "w") as f:
        json.dump(recs, f)
    return ok


def estimate(a):
    """Projected wall time per group from 50-step timings + probe cost measured in the smoke group."""
    import datetime
    recs = json.load(open(os.path.join(a.out, "_tp", "all.json")))
    glog = [json.loads(l) for l in open(glob.glob(os.path.join(a.out, "smoke", "group_log.jsonl"))[0])]
    smoke_cfg = json.load(open(glob.glob(os.path.join(a.out, "smoke", "none", "config.json"))[0]))
    n_smoke = len(smoke_cfg["students"])
    from data_gpu import Split
    n_val = len(Split(a.data_dir, "val"))
    smoke_probe_imgs = smoke_cfg["probe_train"] + smoke_cfg["probe_val"] + n_val
    probe_s = sorted(g["probe_time"] for g in glog)[len(glog) // 2]
    per_img_student = probe_s / (n_smoke * smoke_probe_imgs)
    lines, total = [], 0.0
    for r in recs:
        epochs = r["epochs"]
        spe = a.train_size // r["batch_size"]
        train_s = epochs * spe * r["step_time"]
        n_probe = epochs // 2 + 1 + (epochs % 2)
        probe_s_total = n_probe * len(r["students"]) * (1000 + 2000 + max(n_val, 5000)) * per_img_student
        g_total = (train_s + probe_s_total) * 1.03  # +3% for checkpoints / epoch boundaries
        total += g_total
        lines.append(f"{r['config']}: {len(r['students'])} students, batch {r['batch_size']}, "
                     f"{r['step_time'] * 1000:.0f} ms/step x {spe} steps x {epochs} ep = {train_s / 3600:.2f} h train"
                     f" + {probe_s_total / 3600:.2f} h probes -> {g_total / 3600:.2f} h (peak {r['peak_mem_gb']:.1f} GB)")
    now = datetime.datetime.now()
    end = now + datetime.timedelta(seconds=total)
    txt = "\n".join([f"# estimate made {now:%Y-%m-%d %H:%M} on {recs[0]['gpu']}",
                     f"# train size assumed {a.train_size}, groups run sequentially"] + lines +
                    [f"TOTAL {total / 3600:.2f} h  -> expected finish {end:%Y-%m-%d %H:%M} (if started now)"])
    print(txt)
    with open("logs/estimate.txt", "w") as f:
        f.write(txt + "\n")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("check")
    ap.add_argument("--data_dir", default="data/in100")
    ap.add_argument("--out", default="results_smoke")
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "smoke.yaml"))
    ap.add_argument("--configs", nargs="+", default=[os.path.join(HERE, "configs", c) for c in
                                                      ("core_s0.yaml", "core_s1.yaml", "extra_s0.yaml")])
    ap.add_argument("--keep", type=int, default=30)
    ap.add_argument("--solo", nargs="+", default=["student_attn", "random", "teacher_attn_noisy"])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--mem_limit_gb", type=float, default=float(os.environ.get("MEM_LIMIT_GB", 20)))
    ap.add_argument("--train_size", type=int, default=50000)
    a = ap.parse_args()
    fn = {"equiv_orig": equiv_orig, "equiv_group": equiv_group, "teacher": teacher_check, "monotone": monotone,
          "tau_match": tau_match, "outputs": outputs, "throughput": throughput, "estimate": estimate}[a.check]
    t0 = time.time()
    ok = fn(a)
    print(f"[{a.check}] {time.time() - t0:.0f}s", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
