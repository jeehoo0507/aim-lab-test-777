"""Shared pieces for stage1: original-model loading, teacher helpers, KD loss, FLOPs."""
import os
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

with warnings.catch_warnings():
    # the original files re-register timm model names; harmless, we call the functions directly
    warnings.simplefilter("ignore")
    import models_student  # noqa: E402
    import models_teacher  # noqa: E402
    import losses as orig_losses  # noqa: E402

DEIT_SMALL_URL = "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
IMG_SIZE = 160
PATCH = 16
NUM_PATCHES = (IMG_SIZE // PATCH) ** 2


def imagenet1k_wnids():
    from timm.data import ImageNetInfo
    return list(ImageNetInfo().label_names())


def build_student(num_classes=100, drop_path=0.1):
    # scratch, never pretrained
    return models_student.deit_tiny_patch16_224(
        pretrained=False, img_size=IMG_SIZE, num_classes=num_classes, drop_path_rate=drop_path)


def _resize_pos_embed(pe, new_grid):
    cls, grid = pe[:, :1], pe[:, 1:]
    old = int(grid.shape[1] ** 0.5)
    grid = grid.reshape(1, old, old, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid.float(), size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)
    return torch.cat([cls, grid.to(pe.dtype)], dim=1)


def build_teacher(class_wnids):
    """Original models_teacher DeiT-S at 160px: pos embed 14x14 -> 10x10, head rows sliced to class_wnids."""
    t = models_teacher.deit_small_patch16_224(pretrained=False, img_size=IMG_SIZE, num_classes=1000)
    sd = torch.hub.load_state_dict_from_url(DEIT_SMALL_URL, map_location="cpu", check_hash=True)["model"]
    sd["pos_embed"] = _resize_pos_embed(sd["pos_embed"], IMG_SIZE // PATCH)
    t.load_state_dict(sd)
    all_wnids = imagenet1k_wnids()
    rows = torch.tensor([all_wnids.index(w) for w in class_wnids])
    head = nn.Linear(t.head.in_features, len(rows))
    with torch.no_grad():
        head.weight.copy_(t.head.weight[rows])
        head.bias.copy_(t.head.bias[rows])
    t.head = head
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


@torch.no_grad()
def teacher_masked(teacher, x, keep_idx):
    """Original MaskedKD teacher path (models_teacher.VisionTransformer.forward with maskedkd=True)."""
    return teacher(x, keep_idx, True)


@torch.no_grad()
def teacher_full(teacher, x, need_attn=False):
    """Full teacher forward. With need_attn, re-runs the original blocks' ops step by step to expose
    every layer's attention (B, H, N, N). Equivalence with teacher(x, None, False) is checked in smoke."""
    if not need_attn:
        return teacher(x, None, False), None
    B = x.shape[0]
    h = teacher.patch_embed(x)
    h = torch.cat((teacher.cls_token.expand(B, -1, -1), h), dim=1) + teacher.pos_embed
    h = teacher.pos_drop(h)
    attns = []
    for blk in teacher.blocks:
        a_mod = blk.attn
        y = blk.norm1(h)
        Bn, N, C = y.shape
        qkv = a_mod.qkv(y).reshape(Bn, N, 3, a_mod.num_heads, C // a_mod.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q @ k.transpose(-2, -1)) * a_mod.scale).softmax(dim=-1)
        attn = a_mod.attn_drop(attn)
        y = (attn @ v).transpose(1, 2).reshape(Bn, N, C)
        y = a_mod.proj_drop(a_mod.proj(y))
        h = h + blk.drop_path(y)
        h = h + blk.drop_path(blk.mlp(blk.norm2(h)))
        attns.append(attn)
    h = teacher.norm(h)
    return teacher.head(h[:, 0]), attns


def cls_attn_scores(attn):
    """Original MaskedKD score: head-mean, CLS query row, patch keys. attn: (B,H,N+1,N+1) -> (B,N)."""
    return attn.mean(dim=1)[:, 0, 1:]


def kd_loss(s_logits, t_logits, tau=1.0, t_temp=None):
    """Original soft KD: KLDiv(batchmean)(log_softmax(s/T), softmax(t/T)) * T^2.
    t_temp: extra temperature applied to teacher logits only (tau_const / tau_match); student at tau."""
    s_logits = s_logits.float()
    t_logits = t_logits.float()
    tt = tau if t_temp is None else t_temp
    return nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(s_logits / tau, dim=1), F.softmax(t_logits / tt, dim=1)) * (tau * tau)


def label_smoothing_ce(smoothing=0.1):
    from timm.loss import LabelSmoothingCrossEntropy
    return LabelSmoothingCrossEntropy(smoothing=smoothing) if smoothing > 0 else nn.CrossEntropyLoss()


def mean_entropy(logits, temp=1.0):
    logp = F.log_softmax(logits.float() / temp, dim=1)
    return -(logp.exp() * logp).sum(1).mean()


def vit_gflops(n_tokens, dim=384, depth=12, mlp_ratio=4, patch_tokens_embedded=NUM_PATCHES, num_classes=100):
    """Forward GFLOPs (2*MACs) of a ViT processing n_tokens (incl. CLS) through its blocks.
    Patch embedding is computed on all patches in the original code, so it is counted on all patches."""
    macs = patch_tokens_embedded * 3 * PATCH * PATCH * dim
    per_layer = n_tokens * (4 * dim * dim + 2 * mlp_ratio * dim * dim) + 2 * n_tokens * n_tokens * dim
    macs += depth * per_layer + dim * num_classes
    return 2 * macs / 1e9
