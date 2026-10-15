"""Build the ImageNet-100 fast-mode dataset as uint8 NHWC .npy memmaps.

Real:      python stage1/prepare_in100.py --root $IMAGENET_ROOT --out data/in100
Synthetic: python stage1/prepare_in100.py --synthetic --out data/synth --train-per-class 20 --val-per-class 4

Output: {train,val}_x.npy (N,184,184,3) uint8, {train,val}_y.npy int16, meta.json
Labels index the classes in the order of stage1/imagenet100.txt (CMC list).
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 184


def load_resize(path):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = SIZE / min(w, h)
        nw, nh = max(SIZE, round(w * s)), max(SIZE, round(h * s))
        im = im.resize((nw, nh), Image.BICUBIC)
        l, t = (nw - SIZE) // 2, (nh - SIZE) // 2
        return np.asarray(im.crop((l, t, l + SIZE, t + SIZE)), dtype=np.uint8)


def write_split(out, split, files, labels, workers):
    n = len(files)
    x = np.lib.format.open_memmap(os.path.join(out, f"{split}_x.npy.tmp"), mode="w+",
                                  dtype=np.uint8, shape=(n, SIZE, SIZE, 3))
    t0 = time.time()
    with Pool(workers) as pool:
        for i, arr in enumerate(pool.imap(load_resize, files, chunksize=64)):
            x[i] = arr
            if (i + 1) % 5000 == 0:
                print(f"  {split}: {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    x.flush()
    del x
    os.replace(os.path.join(out, f"{split}_x.npy.tmp"), os.path.join(out, f"{split}_x.npy"))
    np.save(os.path.join(out, f"{split}_y.npy"), np.asarray(labels, dtype=np.int16))


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith((".jpeg", ".jpg", ".png")))


def prepare_real(args, wnids):
    rng = np.random.RandomState(args.seed)
    tr_files, tr_y, va_files, va_y = [], [], [], []
    for c, w in enumerate(wnids):
        tdir = os.path.join(args.root, "train", w)
        vdir = os.path.join(args.root, "val", w)
        if not os.path.isdir(tdir) or not os.path.isdir(vdir):
            raise SystemExit(f"missing class folder for {w} under {args.root}/train or /val")
        fs = list_images(tdir)
        pick = sorted(rng.choice(len(fs), size=min(args.train_per_class, len(fs)), replace=False))
        tr_files += [os.path.join(tdir, fs[i]) for i in pick]
        tr_y += [c] * len(pick)
        vs = list_images(vdir)
        if args.val_per_class:
            vs = vs[:args.val_per_class]
        va_files += [os.path.join(vdir, f) for f in vs]
        va_y += [c] * len(vs)
    print(f"train {len(tr_files)} / val {len(va_files)} images", flush=True)
    write_split(args.out, "val", va_files, va_y, args.workers)
    write_split(args.out, "train", tr_files, tr_y, args.workers)
    return {"train_files": [os.path.relpath(f, args.root) for f in tr_files]}


def synth_split(n_per_class, n_classes, seed):
    """Class-dependent learnable patterns: each class has a fixed color and a fixed set of 'hot' patch
    cells; images are noise + blobs at a random subset of the class cells. Not meaningful for the teacher."""
    rng = np.random.RandomState(seed)
    proto = np.random.RandomState(1234)
    colors = proto.randint(0, 256, size=(n_classes, 3))
    cells = [proto.choice(121, size=12, replace=False) for _ in range(n_classes)]
    n = n_per_class * n_classes
    x = rng.randint(60, 120, size=(n, SIZE, SIZE, 3)).astype(np.uint8)
    y = np.repeat(np.arange(n_classes), n_per_class)
    cs = SIZE // 11
    for i in range(n):
        c = y[i]
        for cell in rng.choice(cells[c], size=8, replace=False):
            r, q = divmod(int(cell), 11)
            x[i, r * cs:(r + 1) * cs, q * cs:(q + 1) * cs] = colors[c]
    perm = rng.permutation(n)
    return x[perm], y[perm].astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("IMAGENET_ROOT"))
    ap.add_argument("--out", default="data/in100")
    ap.add_argument("--classes", default=os.path.join(HERE, "imagenet100.txt"))
    ap.add_argument("--train-per-class", type=int, default=500)
    ap.add_argument("--val-per-class", type=int, default=0, help="0 = all val images of the class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    wnids = [l.strip() for l in open(args.classes) if l.strip()]
    os.makedirs(args.out, exist_ok=True)
    meta = {"classes": wnids, "size": SIZE, "seed": args.seed, "synthetic": args.synthetic,
            "train_per_class": args.train_per_class}
    if args.synthetic:
        xt, yt = synth_split(args.train_per_class, len(wnids), args.seed)
        xv, yv = synth_split(args.val_per_class or 4, len(wnids), args.seed + 1)
        for split, x, y in (("train", xt, yt), ("val", xv, yv)):
            np.save(os.path.join(args.out, f"{split}_x.npy"), x)
            np.save(os.path.join(args.out, f"{split}_y.npy"), y)
    else:
        if not args.root:
            raise SystemExit("IMAGENET_ROOT is not set (expects train/ and val/ class folders)")
        meta.update(prepare_real(args, wnids))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print("done:", args.out)


if __name__ == "__main__":
    main()
