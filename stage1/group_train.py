"""Train several students in one process on the same batches ("group").

python stage1/group_train.py --config stage1/configs/core_s0.yaml [--epochs 40] [--keep 30] [--resume]
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (NUM_PATCHES, REPO, build_student, build_teacher, kd_loss, label_smoothing_ce,  # noqa: E402
                    mean_entropy, teacher_full, teacher_masked, vit_gflops)
from data_gpu import Loader, Split, gpu_train_aug, load_meta  # noqa: E402
from masking import Ctx, StudentSpec  # noqa: E402
from probes import ProbeSet, evaluate_top1, run_probes, write_summary  # noqa: E402

DEFAULTS = dict(
    group="group", seed=0, data_dir="data/in100", out="results", epochs=40, batch_size=256, lr=5e-4,
    warmup_epochs=3, warmup_lr=1e-6, min_lr=1e-5, weight_decay=0.05, keep=30, alpha=0.5, tau=1.0,
    smoothing=0.1, drop_path=0.1, noise_init=2.0, probe_every=2, ckpt_every=5, log_every=50,
    train_limit=None, probe_train=1000, probe_val=2000, tau_match_range=[0.5, 20.0], tau_match_iters=20,
    warn_epoch=10, warn_top1=30.0, students=["none"],
)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--keep", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--data_dir")
    ap.add_argument("--out")
    ap.add_argument("--students", nargs="+", help="override the config's student list")
    ap.add_argument("--group", help="override group name")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--compile", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--deterministic", action="store_true", help="cudnn.benchmark off, TF32 off (equivalence tests)")
    ap.add_argument("--max_steps", type=int, help="throughput mode: run N steps, no probes/ckpt, print timing")
    ap.add_argument("--throughput_out")
    ap.add_argument("--first_step_dump", help="write first-step losses as json and exit")
    return ap.parse_args()


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def param_groups(model, wd):
    skip = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith(".bias") or n in skip else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


class Student:
    def __init__(self, spec, cfg, out_dir, lr):
        self.spec = spec
        self.dir = os.path.join(out_dir, spec.dirname)
        os.makedirs(self.dir, exist_ok=True)
        torch.manual_seed(cfg["seed"])  # identical init for every student of the group (paired)
        self.model = build_student(drop_path=cfg["drop_path"]).cuda().to(memory_format=torch.channels_last)
        self.fwd = self.model
        try:
            self.opt = torch.optim.AdamW(param_groups(self.model, cfg["weight_decay"]), lr=lr, fused=True)
        except Exception:
            self.opt = torch.optim.AdamW(param_groups(self.model, cfg["weight_decay"]), lr=lr)
        # per-student RNG streams so stochastic depth / random masks don't depend on group composition
        g = torch.Generator(device="cuda")
        g.manual_seed(cfg["seed"])
        self.rng = g.get_state()
        self.mask_gen = torch.Generator(device="cuda")
        self.mask_gen.manual_seed(cfg["seed"] * 1000003 + zlib.crc32(spec.raw.encode()))
        self.best = 0.0
        self.teacher_tokens = 0  # teacher tokens (incl. CLS) this student's KD consumed, summed over images
        self.teacher_gflops = 0.0
        self.images = 0
        self.reset_acc()

    def reset_acc(self):
        self.acc = {k: torch.zeros((), device="cuda") for k in
                    ("kd", "ce", "loss", "gcos", "tau", "h_target", "h_match", "h_used", "nonfinite")}
        self.acc_n = 0


def lr_at(step, cfg, total, steps_per_epoch, peak):
    warm = cfg["warmup_epochs"] * steps_per_epoch
    if step < warm:
        return cfg["warmup_lr"] + (peak - cfg["warmup_lr"]) * step / max(1, warm)
    t = (step - warm) / max(1, total - warm)
    return cfg["min_lr"] + 0.5 * (peak - cfg["min_lr"]) * (1 + math.cos(math.pi * min(1.0, t)))


def match_temperature(t_logits, target, lo, hi, iters):
    """Bisection (no host sync) for T with mean entropy(softmax(t/T)) == target; entropy increases with T."""
    lo = torch.full((), lo, device=t_logits.device)
    hi = torch.full((), hi, device=t_logits.device)
    for _ in range(iters):
        mid = (lo + hi) / 2
        below = mean_entropy(t_logits, mid) < target
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    return (lo + hi) / 2


def order_students(specs):
    names = [s.raw for s in specs]
    for s in specs:
        if s.tau_match is not None:
            if s.tau_match not in names:
                raise SystemExit(f"{s.raw}: linked student {s.tau_match!r} is not in this group {names}")
            if not StudentSpec(s.tau_match).masks:
                raise SystemExit(f"{s.raw}: linked student {s.tau_match!r} does not mask the teacher")
    return [s for s in specs if s.tau_match is None] + [s for s in specs if s.tau_match is not None]


def try_compile(students, teacher_mfwd, x, mode):
    """torch.compile students + masked teacher; on any failure fall back to eager (auto) or raise (on)."""
    if mode == "off":
        return students, teacher_mfwd, False
    try:
        tm = torch.compile(teacher_mfwd)
        for st in students:
            st.fwd = torch.compile(st.model)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s, a = students[0].fwd(x)
            idx = torch.topk(a.mean(1)[:, 0, 1:], 30).indices
            tm(x, idx)
        s.float().sum().backward()
        students[0].model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return students, tm, True
    except Exception as e:  # noqa: BLE001
        if mode == "on":
            raise
        print(f"[compile] disabled: {type(e).__name__}: {str(e)[:300]}", flush=True)
        for st in students:
            st.fwd = st.model
            st.model.zero_grad(set_to_none=True)
        return students, teacher_mfwd, False


def main():
    args = parse()
    cfg = dict(DEFAULTS)
    with open(args.config) as f:
        cfg.update(yaml.safe_load(f) or {})
    for k in ("epochs", "keep", "batch_size", "data_dir", "out", "students", "group"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    amp = None if args.no_amp else torch.bfloat16
    torch.backends.cudnn.benchmark = not args.deterministic
    torch.backends.cuda.matmul.allow_tf32 = not args.deterministic
    torch.backends.cudnn.allow_tf32 = not args.deterministic
    torch.backends.cudnn.deterministic = args.deterministic
    compile_mode = "off" if args.deterministic else args.compile

    meta = load_meta(cfg["data_dir"])
    train = Split(cfg["data_dir"], "train", cfg["train_limit"])
    val = Split(cfg["data_dir"], "val")
    bs = cfg["batch_size"]
    steps_per_epoch = len(train) // bs
    if steps_per_epoch == 0:
        raise SystemExit(f"train set ({len(train)}) smaller than batch ({bs})")
    total_steps = cfg["epochs"] * steps_per_epoch
    peak_lr = cfg["lr"] * bs / 512.0
    keep = cfg["keep"]
    gdir = os.path.join(cfg["out"], cfg["group"])
    os.makedirs(gdir, exist_ok=True)

    teacher = build_teacher(meta["classes"]).cuda().to(memory_format=torch.channels_last)
    specs = order_students([StudentSpec(s) for s in cfg["students"]])
    students = [Student(s, cfg, gdir, peak_lr) for s in specs]
    need_full = any(s.needs_full_teacher for s in specs)
    need_tattn = any(s.needs_teacher_attn for s in specs)
    ce_fn = label_smoothing_ce(cfg["smoothing"])

    t_mfwd = lambda x, idx: teacher_masked(teacher, x, idx)  # noqa: E731
    dummy = torch.zeros(bs, 3, 160, 160, device="cuda").contiguous(memory_format=torch.channels_last)
    students, t_mfwd, compiled = try_compile(students, t_mfwd, dummy, compile_mode)
    del dummy

    start_epoch, step, gpu_time = 0, 0, 0.0
    if args.resume and os.path.exists(os.path.join(gdir, "group_state.pth")):
        gs = torch.load(os.path.join(gdir, "group_state.pth"), weights_only=False)
        start_epoch, step, gpu_time = gs["epoch"], gs["step"], gs["gpu_time"]
        for st in students:
            ck = torch.load(os.path.join(st.dir, "ckpt.pth"), map_location="cuda", weights_only=False)
            st.model.load_state_dict(ck["model"])
            st.opt.load_state_dict(ck["opt"])
            st.rng = ck["rng"]
            st.mask_gen.set_state(ck["mask_gen"])
            st.best, st.teacher_tokens = ck["best"], ck["teacher_tokens"]
            st.teacher_gflops, st.images = ck["teacher_gflops"], ck["images"]
        print(f"resumed {cfg['group']} at epoch {start_epoch} step {step}", flush=True)

    run_info = dict(cfg, git_commit=git_hash(), peak_lr=peak_lr, steps_per_epoch=steps_per_epoch,
                    compiled=compiled, amp=str(amp), torch=torch.__version__,
                    gpu=torch.cuda.get_device_name(), synthetic_data=meta.get("synthetic", False))
    for st in students:
        with open(os.path.join(st.dir, "config.json"), "w") as f:
            json.dump(dict(run_info, criterion=st.spec.raw), f, indent=1)

    throughput = args.max_steps is not None
    probe = None
    if not throughput and not args.first_step_dump:
        probe = ProbeSet(train, val, cfg["probe_train"], cfg["probe_val"])

    full_gf = vit_gflops(NUM_PATCHES + 1)
    mask_gf = vit_gflops(keep + 1)
    aug_gen = torch.Generator(device="cuda")
    step_times = []

    def do_probe(epoch):
        t0 = time.time()
        prog = min(1.0, step / max(1, total_steps))
        res = run_probes(students, teacher, probe, keep, prog, cfg["noise_init"], amp, epoch,
                         vis_dir_fn=lambda st: st.dir)
        for st in students:
            v = evaluate_top1(st.model, val, amp)
            st.best = max(st.best, v)
            m = res[st.spec.dirname]
            m.update(step=step, val_top1_full=v, best_val_top1=st.best,
                     teacher_tokens_per_img=st.teacher_tokens / max(1, st.images),
                     teacher_gflops_per_img=st.teacher_gflops / max(1, st.images),
                     teacher_gflops_total=st.teacher_gflops)
            with open(os.path.join(st.dir, "metrics.jsonl"), "a") as f:
                f.write(json.dumps(m) + "\n")
            if st.spec.raw == "none" and epoch == cfg["warn_epoch"] and v < cfg["warn_top1"]:
                os.makedirs("logs", exist_ok=True)
                with open("logs/WARNING.txt", "a") as f:
                    f.write(f"[{cfg['group']}] none student val top-1 at epoch {epoch} = {v:.2f}% "
                            f"< {cfg['warn_top1']}% : training may be too weak for attention to be meaningful\n")
        print(f"[{cfg['group']}] probe ep{epoch}: " + ", ".join(
            f"{st.spec.raw}={res[st.spec.dirname]['val_top1_full']:.1f}" for st in students)
            + f" ({time.time() - t0:.0f}s)", flush=True)
        write_summary(cfg["out"])

    if probe is not None and start_epoch == 0:
        do_probe(0)

    first_losses = {}
    for epoch in range(start_epoch, cfg["epochs"] if not throughput else 10 ** 9):
        aug_gen.manual_seed(cfg["seed"] * 7919 + epoch)
        loader = Loader(train, bs, cfg["seed"], epoch)
        torch.cuda.synchronize()
        t_epoch = time.time()
        for xu8, y in loader:
            ts = time.time()
            x = gpu_train_aug(xu8, aug_gen)
            lr = lr_at(step, cfg, total_steps, steps_per_epoch, peak_lr)
            prog = step / max(1, total_steps)
            t_full, t_attns = None, None
            with torch.no_grad(), torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                if need_full:
                    t_full, t_attns = teacher_full(teacher, x, need_attn=need_tattn)
            h_masked = {}
            for st in students:
                for g in st.opt.param_groups:
                    g["lr"] = lr
                loss = train_step(st, x, y, t_full, t_attns, h_masked, t_mfwd, ce_fn, cfg, amp, prog,
                                  full_gf, mask_gf)
                if args.first_step_dump and step == 0:
                    first_losses[st.spec.raw] = loss.item()
            step += 1
            if args.first_step_dump:
                with open(args.first_step_dump, "w") as f:
                    json.dump(first_losses, f)
                print("first-step losses:", first_losses)
                return
            if step % cfg["log_every"] == 0:
                flush_logs(students, step, epoch, lr)
            if throughput:
                torch.cuda.synchronize()
                step_times.append(time.time() - ts)
                if step >= args.max_steps:
                    report_throughput(args, cfg, step_times, compiled, steps_per_epoch)
                    return
        torch.cuda.synchronize()
        if throughput:
            continue
        train_time = time.time() - t_epoch
        gpu_time += train_time
        ep = epoch + 1
        t_probe = time.time()
        if ep % cfg["probe_every"] == 0 or ep == cfg["epochs"]:
            do_probe(ep)
        probe_time = time.time() - t_probe
        gpu_time += probe_time
        with open(os.path.join(gdir, "group_log.jsonl"), "a") as f:
            f.write(json.dumps(dict(epoch=ep, step=step, train_time=train_time, probe_time=probe_time,
                                    gpu_time_total=gpu_time, compiled=compiled,
                                    peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30)) + "\n")
        print(f"[{cfg['group']}] epoch {ep}/{cfg['epochs']} done: train {train_time:.0f}s probe {probe_time:.0f}s"
              f" total {gpu_time / 3600:.2f}h", flush=True)
        if ep % cfg["ckpt_every"] == 0 or ep == cfg["epochs"]:
            save_ckpt(students, gdir, ep, step, gpu_time, cfg)
    write_summary(cfg["out"])


def train_step(st, x, y, t_full, t_attns, h_masked, t_mfwd, ce_fn, cfg, amp, prog, full_gf, mask_gf):
    spec = st.spec
    torch.cuda.set_rng_state(st.rng)  # this student's own drop-path stream
    with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
        s_logits, s_attn = st.fwd(x)
    B = x.shape[0]
    with torch.no_grad():
        idx = spec.crit(Ctx(s_attn.detach(), t_attns, cfg["keep"], prog, st.mask_gen, cfg["noise_init"]))
        if idx is not None:
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                t_logits = t_mfwd(x, idx)
            tokens, gf = cfg["keep"] + 1, mask_gf
            if spec.needs_teacher_attn:  # also needed the full forward for its attention
                tokens, gf = tokens + NUM_PATCHES + 1, gf + full_gf
            h_masked[spec.raw] = mean_entropy(t_logits)
        else:
            t_logits = t_full
            tokens, gf = NUM_PATCHES + 1, full_gf
        t_logits = t_logits.float()
        t_temp = None
        if spec.tau_const is not None:
            t_temp = torch.tensor(spec.tau_const, device=x.device)
        elif spec.tau_match is not None:
            lo, hi = cfg["tau_match_range"]
            target = h_masked[spec.tau_match]
            t_temp = match_temperature(t_logits, target, lo, hi, cfg["tau_match_iters"])
            st.acc["h_target"] += target
            st.acc["h_match"] += mean_entropy(t_logits, t_temp)
    kd = kd_loss(s_logits, t_logits, cfg["tau"], t_temp)
    ce = ce_fn(s_logits.float(), y)
    loss = kd if spec.noce else ce * (1 - cfg["alpha"]) + kd * cfg["alpha"]
    st.opt.zero_grad(set_to_none=True)
    loss.backward()
    st.opt.step()
    st.rng = torch.cuda.get_rng_state()
    with torch.no_grad():
        sl = s_logits.detach().float()
        p_s_tau = F.softmax(sl / cfg["tau"], 1)
        p_t = F.softmax(t_logits / (cfg["tau"] if t_temp is None else t_temp), 1)
        g_kd = p_s_tau - p_t
        g_ce = F.softmax(sl, 1) - F.one_hot(y, sl.shape[1]).float()
        st.acc["gcos"] += F.cosine_similarity(g_kd, g_ce, dim=1).mean()
        st.acc["h_used"] += -(p_t * torch.log(p_t.clamp_min(1e-30))).sum(1).mean()
        st.acc["kd"] += kd.detach()
        st.acc["ce"] += ce.detach()
        st.acc["loss"] += loss.detach()
        st.acc["nonfinite"] += (~torch.isfinite(loss.detach())).float()
        if t_temp is not None:
            st.acc["tau"] += t_temp
    st.acc_n += 1
    st.teacher_tokens += tokens * B
    st.teacher_gflops += gf * B
    st.images += B
    return loss.detach()


def flush_logs(students, step, epoch, lr):
    for st in students:
        n = max(1, st.acc_n)
        vals = {k: (v / n).item() for k, v in st.acc.items()}
        rec = dict(step=step, epoch=epoch, lr=lr, kd=vals["kd"], ce=vals["ce"], loss=vals["loss"],
                   grad_cos_kd_ce=vals["gcos"], entropy_teacher_used=vals["h_used"], nonfinite_steps=vals["nonfinite"] * n)
        if st.spec.tau_const is not None or st.spec.tau_match is not None:
            rec["tau_teacher"] = vals["tau"]
        if st.spec.tau_match is not None:
            rec["entropy_target"] = vals["h_target"]
            rec["entropy_matched"] = vals["h_match"]
            rec["entropy_gap"] = vals["h_match"] - vals["h_target"]
        with open(os.path.join(st.dir, "train_log.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        if rec["nonfinite_steps"] > 0:
            raise SystemExit(f"non-finite loss in {st.spec.raw} around step {step}")
        st.reset_acc()


def save_ckpt(students, gdir, epoch, step, gpu_time, cfg):
    for st in students:
        tmp = os.path.join(st.dir, "ckpt.pth.tmp")
        torch.save(dict(model=st.model.state_dict(), opt=st.opt.state_dict(), rng=st.rng,
                        mask_gen=st.mask_gen.get_state(), best=st.best, teacher_tokens=st.teacher_tokens,
                        teacher_gflops=st.teacher_gflops, images=st.images, epoch=epoch, step=step), tmp)
        os.replace(tmp, os.path.join(st.dir, "ckpt.pth"))
    torch.save(dict(epoch=epoch, step=step, gpu_time=gpu_time, cfg=cfg), os.path.join(gdir, "group_state.pth"))


def report_throughput(args, cfg, step_times, compiled, steps_per_epoch):
    warm = min(10, len(step_times) // 2)
    t = float(np.median(step_times[warm:]))
    rec = dict(group=cfg["group"], students=cfg["students"], batch_size=cfg["batch_size"], step_time=t,
               peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30, compiled=compiled,
               gpu=torch.cuda.get_device_name(), steps=len(step_times))
    print("THROUGHPUT " + json.dumps(rec), flush=True)
    if args.throughput_out:
        with open(args.throughput_out, "w") as f:
            json.dump(rec, f)


if __name__ == "__main__":
    main()
