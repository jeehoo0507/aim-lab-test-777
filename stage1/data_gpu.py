"""uint8 memmap loader with a prefetch thread + batched GPU augmentation."""
import json
import math
import os
import queue
import threading

import numpy as np
import torch
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]) * 255
STD = torch.tensor([0.229, 0.224, 0.225]) * 255
OUT = 160


class Split:
    def __init__(self, data_dir, split, limit=None):
        self.x = np.load(os.path.join(data_dir, f"{split}_x.npy"), mmap_mode="r")
        self.y = np.load(os.path.join(data_dir, f"{split}_y.npy")).astype(np.int64)
        if limit:
            # deterministic random subset (fixed permutation), independent of the training seed
            idx = np.sort(np.random.RandomState(999).permutation(len(self.y))[:limit])
            self.idx = idx
        else:
            self.idx = np.arange(len(self.y))

    def __len__(self):
        return len(self.idx)

    def get(self, rows):
        """rows index into this (possibly limited) split; returns uint8 NHWC numpy + labels."""
        real = self.idx[rows]
        order = np.argsort(real)  # sorted reads are friendlier to the memmap
        xs = np.empty((len(real),) + self.x.shape[1:], dtype=np.uint8)
        xs[order] = self.x[real[order]]
        return xs, self.y[real]


def load_meta(data_dir):
    with open(os.path.join(data_dir, "meta.json")) as f:
        return json.load(f)


class Loader:
    """Iterates one epoch of (uint8 NCHW cuda tensor, labels cuda) with drop_last. Order depends only on
    (seed, epoch), so resuming at an epoch boundary reproduces the data stream."""

    def __init__(self, split, batch_size, seed, epoch, shuffle=True, drop_last=True, device="cuda", prefetch=3):
        n = len(split)
        if shuffle:
            order = np.random.RandomState(seed * 100003 + epoch).permutation(n)
        else:
            order = np.arange(n)
        nb = n // batch_size if drop_last else math.ceil(n / batch_size)
        self.batches = [order[i * batch_size:(i + 1) * batch_size] for i in range(nb)]
        self.split, self.device = split, device
        self.q = queue.Queue(maxsize=prefetch)
        self.t = threading.Thread(target=self._work, daemon=True)
        self.t.start()

    def _work(self):
        for rows in self.batches:
            x, y = self.split.get(rows)
            self.q.put((torch.from_numpy(x).pin_memory(), torch.from_numpy(y).pin_memory()))
        self.q.put(None)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            x, y = item
            x = x.to(self.device, non_blocking=True).permute(0, 3, 1, 2)  # NCHW view of NHWC = channels_last
            yield x, y.to(self.device, non_blocking=True)


def normalize(x_float255):
    m = MEAN.to(x_float255.device).view(1, 3, 1, 1)
    s = STD.to(x_float255.device).view(1, 3, 1, 1)
    return ((x_float255 - m) / s).contiguous(memory_format=torch.channels_last)


def gpu_train_aug(x_u8, gen, scale=(0.35, 1.0), ratio=(3 / 4, 4 / 3)):
    """Per-sample RandomResizedCrop(160, scale) + horizontal flip via one affine grid_sample.
    torchvision-style area/aspect sampling (single draw, clamped to the image instead of 10 retries)."""
    B, _, H, W = x_u8.shape
    dev = x_u8.device
    u = torch.rand(B, 4, generator=gen, device=dev)
    area = scale[0] + (scale[1] - scale[0]) * u[:, 0]
    logr = math.log(ratio[0]) + (math.log(ratio[1]) - math.log(ratio[0])) * u[:, 1]
    r = torch.exp(logr)
    w = torch.sqrt(area * r).clamp(max=1.0)  # fraction of width
    h = torch.sqrt(area / r).clamp(max=1.0)
    cx = (1 - w) * u[:, 2] + w / 2  # crop center in [0,1]
    cy = (1 - h) * u[:, 3] + h / 2
    flip = (torch.rand(B, generator=gen, device=dev) < 0.5).float() * -2 + 1
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = w * flip
    theta[:, 0, 2] = cx * 2 - 1
    theta[:, 1, 1] = h
    theta[:, 1, 2] = cy * 2 - 1
    grid = F.affine_grid(theta, (B, 3, OUT, OUT), align_corners=False)
    out = F.grid_sample(x_u8.float(), grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    return normalize(out)


def gpu_eval(x_u8):
    """Center crop 160 from the stored 184 (crop_pct ~0.87)."""
    H = x_u8.shape[-1]
    o = (H - OUT) // 2
    return normalize(x_u8[:, :, o:o + OUT, o:o + OUT].float())
