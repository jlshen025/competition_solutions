"""Preprocessed-data access and the per-window rel_l2 used by calibration."""
import json, os
import numpy as np

OUT = os.environ.get("RPDE_DATA32", "data32")
T_IN = T_OUT = 20


def load(split):
    """-> (memmapped (T_total, 32, 64, C) float32 array, per-trajectory index)."""
    arr = np.load(os.path.join(OUT, f"{split}.npy"), mmap_mode="r")
    idx = json.load(open(os.path.join(OUT, f"{split}_index.json")))
    return arr, idx


def windows(idx, stride=T_OUT, keep=None, t_lo=0.0, t_hi=1.0):
    """(traj_i, t0) pairs. With stride >= T_OUT no window's target overlaps
    another window's input."""
    out = []
    for i, r in enumerate(idx):
        if keep is not None and not keep(r):
            continue
        T = r["T"]
        a, b = int(t_lo * T), int(t_hi * T)
        for t0 in range(a, b - (T_IN + T_OUT) + 1, stride):
            out.append((i, t0))
    return out


def gather(arr, idx, wins, nchan_out=3):
    """-> x (N,20,H,W,3), y (N,20,H,W,3); channels beyond the source stay 0."""
    N = len(wins)
    H, W, C = arr.shape[1], arr.shape[2], arr.shape[3]
    x = np.zeros((N, T_IN, H, W, nchan_out), np.float32)
    y = np.zeros((N, T_OUT, H, W, nchan_out), np.float32)
    for n, (i, t0) in enumerate(wins):
        s = idx[i]["start"] + t0
        blk = np.asarray(arr[s:s + T_IN + T_OUT])
        c = min(C, nchan_out)
        x[n, ..., :c] = blk[:T_IN, ..., :c]
        y[n, ..., :c] = blk[T_IN:, ..., :c]
    return x, y


def _rel(p, t):
    a = p.reshape(p.shape[0], -1)
    b = t.reshape(t.shape[0], -1)
    return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1).clip(min=1e-8)


def rel_l2_ps(p, t, c=2):
    """Per-sample relative L2 over the first c channels (the scorer's rel_l2)."""
    return _rel(p[..., :c], t[..., :c])
