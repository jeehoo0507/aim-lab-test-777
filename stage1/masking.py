"""Registry of patch-selection criteria. Each returns keep indices (B, K) into the patch tokens, or None
for 'no masking'. Add a new criterion (e.g. Grad-CAM) with:

    @register("gradcam", needs_teacher_attn=False, needs_teacher_grad=True)
    def gradcam(ctx): ...
"""
import torch

from common import cls_attn_scores

CRITERIA = {}


class Criterion:
    def __init__(self, name, fn, needs_teacher_attn=False, masks=True):
        self.name, self.fn = name, fn
        self.needs_teacher_attn = needs_teacher_attn
        self.masks = masks

    def __call__(self, ctx):
        return self.fn(ctx)


def register(name, needs_teacher_attn=False, masks=True):
    def deco(fn):
        CRITERIA[name] = Criterion(name, fn, needs_teacher_attn, masks)
        return fn
    return deco


class Ctx:
    """s_attn: student last-block attention (B,H,N+1,N+1); t_attns: list of full-teacher attentions per layer;
    keep: K; progress: training progress in [0,1]; gen: torch.Generator on the batch device."""

    def __init__(self, s_attn=None, t_attns=None, keep=30, progress=0.0, gen=None, noise_init=2.0):
        self.s_attn, self.t_attns, self.keep = s_attn, t_attns, keep
        self.progress, self.gen, self.noise_init = progress, gen, noise_init


def topk_keep(scores, k):
    # identical to the original: torch.topk(scores, k).indices (descending order, used as gather index)
    return torch.topk(scores, k).indices


@register("none", masks=False)
def _none(ctx):
    return None


@register("student_attn")
def student_attn(ctx):
    return topk_keep(cls_attn_scores(ctx.s_attn), ctx.keep)


@register("student_attn_inv")
def student_attn_inv(ctx):
    return topk_keep(-cls_attn_scores(ctx.s_attn), ctx.keep)


@register("random")
def random_keep(ctx):
    B, N = ctx.s_attn.shape[0], ctx.s_attn.shape[-1] - 1
    u = torch.rand(B, N, generator=ctx.gen, device=ctx.s_attn.device)
    return topk_keep(u, ctx.keep)


def teacher_scores(ctx):
    return cls_attn_scores(ctx.t_attns[-1])


@register("teacher_attn", needs_teacher_attn=True)
def teacher_attn(ctx):
    return topk_keep(teacher_scores(ctx), ctx.keep)


@register("teacher_attn_noisy", needs_teacher_attn=True)
def teacher_attn_noisy(ctx):
    """Gaussian noise on per-image z-scored teacher scores; std decays linearly noise_init -> 0."""
    s = teacher_scores(ctx).float()
    s = (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True) + 1e-8)
    sigma = ctx.noise_init * max(0.0, 1.0 - ctx.progress)
    noise = torch.randn(s.shape, generator=ctx.gen, device=s.device) * sigma
    return topk_keep(s + noise, ctx.keep)


def attention_rollout(t_attns, residual=0.5):
    """Abnar & Zuidema rollout: head-mean, add identity for residual, row-normalize, multiply over layers."""
    r = None
    for a in t_attns:
        a = a.float().mean(1)
        a = residual * a + (1 - residual) * torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        r = a if r is None else a @ r
    return r[:, 0, 1:]


@register("rollout", needs_teacher_attn=True)
def rollout(ctx):
    return topk_keep(attention_rollout(ctx.t_attns), ctx.keep)


class StudentSpec:
    """Parses a --criterion string: base criterion, *_noce, tau_const:T, tau_match:<name>."""

    def __init__(self, s):
        self.raw = s
        self.noce = False
        self.tau_const = None
        self.tau_match = None
        base = s
        if s.startswith("tau_const:"):
            self.tau_const = float(s.split(":", 1)[1])
            base = "none"
        elif s.startswith("tau_match:"):
            self.tau_match = s.split(":", 1)[1]
            base = "none"
        elif s.endswith("_noce"):
            self.noce = True
            base = s[: -len("_noce")]
        if base not in CRITERIA:
            raise ValueError(f"unknown criterion {base!r} (from {s!r}); known: {sorted(CRITERIA)}")
        self.crit = CRITERIA[base]
        self.dirname = s.replace(":", "_")

    @property
    def masks(self):
        return self.crit.masks

    @property
    def needs_full_teacher(self):
        # full logits are needed for unmasked KD and for tau_* ; full attention for teacher-based criteria
        return (not self.crit.masks) or self.crit.needs_teacher_attn

    @property
    def needs_teacher_attn(self):
        return self.crit.needs_teacher_attn

    def probe_criterion(self):
        """Mask used at probe time for 'the teacher masked by this student's criterion'. Unmasked
        students (none / tau_*) use the original MaskedKD rule on their own attention."""
        return self.crit if self.crit.masks else CRITERIA["student_attn"]
