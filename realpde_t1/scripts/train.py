"""Train one member of the RealPDE Track-1 ensemble.

The loss mirrors the scored metric in physical units: per-window relative L2,
plus the TKE and MVPE statistics, a smooth interval-coverage term on the mean
head, and a Gaussian NLL that trains only the scale head.

Validation uses whole held-out Reynolds numbers (--holdout-re), because the
private test set is drawn from unseen conditions and consecutive windows of one
trajectory are near-duplicates. With --train-all 1 those trajectories are also
trained on, so validation is in-sample and only picks the checkpoint.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rpde import data as D
from rpde.model import Net, n_params

OUT = os.environ.get("EXP_OUTPUT_DIR", ".")
T_IN = T_OUT = 20
# Fixed global normalisation: mean/std of real-data (u, v) over train_real.
MEAN_R = np.array([0.154960856, -0.000513992854], np.float32)
STD_R = np.array([0.0968056545, 0.015960684], np.float32)


class GPUWindows:
    """Whole split resident on the GPU; windows are gathered by index."""

    def __init__(self, split, dev, keep=None, chans=2):
        arr, idx = D.load(split)
        self.idx = [r for r in idx if keep is None or keep(r)]
        blocks, starts, off = [], [], 0
        for r in self.idx:
            blocks.append(np.asarray(arr[r["start"]:r["start"] + r["T"], ..., :chans]))
            starts.append(off); off += r["T"]
        self.data = torch.from_numpy(np.concatenate(blocks, 0)).to(dev)
        self.starts = starts
        self.dev = dev
        self.pairs = [(i, t0) for i, r in enumerate(self.idx)
                      for t0 in range(0, r["T"] - (T_IN + T_OUT) + 1)]
        self._fp = {}

    def frac_pairs_for(self, kmax):
        """(i, t0) whose time-rescaled window (k <= kmax) fits in trajectory i,
        leaving room for the +1 upper interpolation index."""
        key = round(float(kmax), 6)
        if key not in self._fp:
            span = int(math.ceil((T_IN + T_OUT - 1) * max(kmax, 1.0))) + 1
            self._fp[key] = [(i, t0) for i, r in enumerate(self.idx)
                             for t0 in range(0, r["T"] - span)]
        return self._fp[key]

    def batch_frac(self, n, rng, klo, khi):
        """Reynolds-similarity augmentation: u'(x, t) = k * u(x, k*t) is an
        exact Navier-Stokes solution at Re' = k^2 Re. k is drawn log-uniform in
        [klo, khi]; the window is resampled at times k*t by linear
        interpolation between neighbouring frames, then scaled by k."""
        pairs = self.frac_pairs_for(khi)
        sel = [pairs[j] for j in rng.integers(0, len(pairs), n)]
        base = torch.tensor([self.starts[i] + t0 for i, t0 in sel],
                            device=self.dev)
        ks = np.exp(rng.uniform(math.log(klo), math.log(khi), n))
        kt = torch.as_tensor(ks, device=self.dev, dtype=torch.float32)
        off = torch.arange(T_IN + T_OUT, device=self.dev, dtype=torch.float32)
        pos = kt[:, None] * off[None, :]                      # (n, 40)
        tail = self.data.shape[1:]
        i0 = pos.floor().long()
        w = (pos - i0.float()).reshape(n, T_IN + T_OUT, *([1] * len(tail)))
        g0 = (base[:, None] + i0).reshape(-1)
        b0 = self.data[g0].reshape(n, T_IN + T_OUT, *tail)
        b1 = self.data[g0 + 1].reshape(n, T_IN + T_OUT, *tail)
        blk = b0 * (1.0 - w) + b1 * w
        blk = blk * kt.reshape(n, 1, *([1] * len(tail)))
        return blk[:, :T_IN].contiguous(), blk[:, T_IN:].contiguous()

    def batch(self, n, rng):
        """n native windows drawn uniformly."""
        sel = [self.pairs[j] for j in rng.integers(0, len(self.pairs), n)]
        base = torch.tensor([self.starts[i] + t0 for i, t0 in sel], device=self.dev)
        off = torch.arange(T_IN + T_OUT, device=self.dev)
        blk = self.data[(base[:, None] + off[None, :]).reshape(-1)]
        blk = blk.reshape(n, T_IN + T_OUT, *self.data.shape[1:])
        return blk[:, :T_IN].contiguous(), blk[:, T_IN:].contiguous()

    def eval_set(self, stride=T_OUT, cap=None):
        pairs = [(i, t0) for i, r in enumerate(self.idx)
                 for t0 in range(0, r["T"] - (T_IN + T_OUT) + 1, stride)]
        if cap and len(pairs) > cap:
            pairs = pairs[::max(1, len(pairs) // cap)][:cap]
        base = torch.tensor([self.starts[i] + t0 for i, t0 in pairs], device=self.dev)
        off = torch.arange(T_IN + T_OUT, device=self.dev)
        blk = self.data[(base[:, None] + off[None, :]).reshape(-1)]
        blk = blk.reshape(len(pairs), T_IN + T_OUT, *self.data.shape[1:])
        return blk[:, :T_IN].contiguous(), blk[:, T_IN:].contiguous()


def rel_l2(p, t):
    """Per-window relative L2, averaged over the batch (the scored rel_l2)."""
    a = (p - t).flatten(1)
    d = t.flatten(1).norm(dim=1).clamp(min=1e-8)
    return (a.norm(dim=1) / d).mean()


def tke_field(x):
    u, v = x[..., 0], x[..., 1]
    return 0.5 * (((u - u.mean(1, keepdim=True)) ** 2).mean(1)
                  + ((v - v.mean(1, keepdim=True)) ** 2).mean(1))


def tke_loss(p, t):
    """Relative L2 of the spatial map of temporal velocity variance."""
    a = (tke_field(p) - tke_field(t)).flatten(1)
    b = tke_field(t).flatten(1)
    return (a.norm(dim=1) / b.norm(dim=1).clamp(min=1e-8)).mean()


# MVPE probe geometry of the official scorer at 32x64
# (d=16, center_x=10, center_y=32, n_probe=9, subsampling 2).
PROBE_Y = [8, 10, 12, 14, 16, 18, 20, 22, 24]
PROBE_X = [13, 21, 29, 37]


def mvpe_loss(p, t):
    """Time-averaged velocity profile error at the 9x4 wake probes."""
    ps = p[:, :, PROBE_Y][:, :, :, PROBE_X].mean(1)      # (B,9,4,2)
    ts = t[:, :, PROBE_Y][:, :, :, PROBE_X].mean(1)
    errs = []
    for i in range(len(PROBE_X)):
        a = (ps[:, :, i] - ts[:, :, i]).flatten(1)
        b = ts[:, :, i].flatten(1)
        errs.append(a.norm(dim=1) / b.norm(dim=1).clamp(min=1e-8))
    return torch.stack(errs, 0).mean()


def _mirror(t, nf):
    """Reflect the first nf samples across the cross-stream axis (H) and negate
    v. Out of place: batches may be views of the GPU data store."""
    m = torch.flip(t[:nf], dims=[2]).clone()
    m[..., 1] = -m[..., 1]
    return torch.cat([m, t[nf:]], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="real", choices=["real"])
    ap.add_argument("--ar-m", type=int, default=0,
                    help="autoregressive rollout: frames per step (0 = direct map)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--spectral", type=int, default=0, choices=[0],
                    help="kept for command-line compatibility; only 0 (no "
                         "spectral layers) is implemented")
    ap.add_argument("--delta", type=int, default=0,
                    help="predict the change from the last observed frame")
    ap.add_argument("--lam-tke", type=float, default=0.0)
    ap.add_argument("--lam-mvpe", type=float, default=0.0)
    ap.add_argument("--lam-nll", type=float, default=0.2,
                    help="Gaussian NLL weight; the residual is detached, so "
                         "this trains only the scale head")
    ap.add_argument("--lam-cov", type=float, default=0.0,
                    help="weight of the smooth coverage term "
                         "mean(1 - exp(-(r / (cov_k * shat))^2)) on the mean head")
    ap.add_argument("--cov-k", type=float, default=2.0,
                    help="coverage bandwidth in units of the predicted scale")
    ap.add_argument("--holdout-re", default="13950,22875")
    ap.add_argument("--train-all", type=int, default=0,
                    help="also train on the held-out Re; validation is then "
                         "in-sample and only selects the checkpoint")
    ap.add_argument("--select", default="composite", choices=["composite"],
                    help="checkpoint criterion: 0.737*rel_l2_score + "
                         "0.123*tke_score on the held-out-Re windows")
    ap.add_argument("--aug-krange", default="",
                    help="'lo,hi': Reynolds-similarity rescaling with k "
                         "log-uniform in [lo, hi]")
    ap.add_argument("--aug-kmode", default="interp", choices=["interp"],
                    help="time resampling for --aug-krange: linear interpolation")
    ap.add_argument("--aug-frac", type=float, default=0.0,
                    help="share of each batch drawn with --aug-krange")
    ap.add_argument("--aug-flip", type=float, default=0.0,
                    help="share of each batch mirrored across the cross-stream "
                         "axis (y -> -y, v -> -v)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"

    ho = tuple(int(x) for x in a.holdout_re.split(",") if x)
    tr = GPUWindows(a.split, dev,
                    keep=None if a.train_all else (lambda r: r["re"] not in ho))
    va = GPUWindows("real", dev, keep=lambda r: r["re"] in ho)
    print(f"train {len(tr.idx)} traj / {len(tr.pairs)} windows | "
          f"val {len(va.idx)} traj"
          + ("  [train-all: val is in-sample, used only to pick the checkpoint]"
             if a.train_all else ""), flush=True)

    mean = torch.tensor(MEAN_R, device=dev)
    std = torch.tensor(STD_R, device=dev)

    net = Net(width=a.width, depth=a.depth, delta=bool(a.delta),
              ar_m=int(a.ar_m)).to(dev)
    if a.ar_m:
        print(f"rollout: {20 // a.ar_m} step(s) of a 20->{a.ar_m} operator",
              flush=True)
    print(f"params {n_params(net)/1e6:.2f}M "
          f"({n_params(net)*4/1e6:.1f} MB fp32)", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                                pct_start=0.05)
    # torch>=2.3 has torch.amp.GradScaler; 2.2.x only torch.cuda.amp.GradScaler.
    scaler = (torch.amp.GradScaler("cuda") if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler())

    xv, yv = va.eval_set(stride=T_OUT, cap=400)
    best, best_v, best_step, hist = 1e9, 1e9, -1, []
    t0 = time.time()
    krange = [float(x) for x in a.aug_krange.split(",") if x.strip()]
    assert len(krange) in (0, 2), "--aug-krange takes 'lo,hi'"
    n_aug = int(round(a.bs * a.aug_frac)) if (krange and a.aug_frac > 0) else 0
    n_flip = int(round(a.bs * a.aug_flip)) if a.aug_flip > 0 else 0
    print(f"every batch: {a.bs - n_aug} native + {n_aug} k-rescaled "
          f"(k in {krange}), first {n_flip} mirrored", flush=True)
    for step in range(a.steps):
        if n_aug > 0:
            xb, yb = tr.batch(a.bs - n_aug, rng)
            xa, ya = tr.batch_frac(n_aug, rng, krange[0], krange[1])
            x = torch.cat([xb, xa], 0)
            y = torch.cat([yb, ya], 0)
        else:
            x, y = tr.batch(a.bs, rng)
        if n_flip > 0:
            x, y = _mirror(x, n_flip), _mirror(y, n_flip)
        xn = (x - mean) / std
        with torch.autocast("cuda", torch.bfloat16):
            mu, ls = net(xn)
        mu, ls = mu.float(), ls.float()
        # Loss in physical units, where u dominates the norm as it does in the
        # metric; the scale head then predicts a physical sigma, which is what
        # the width table consumes.
        mu_p, tgt = mu * std + mean, y
        ls_p = ls + torch.log(std).mean()
        loss = rel_l2(mu_p, tgt)
        if a.lam_tke > 0:
            loss = loss + a.lam_tke * tke_loss(mu_p, tgt)
        if a.lam_mvpe > 0:
            loss = loss + a.lam_mvpe * mvpe_loss(mu_p, tgt)
        if a.lam_cov > 0:
            # Smooth, bounded surrogate for the fraction of elements inside an
            # interval of the predicted scale (the SPS reward is per element).
            # The scale is detached, so this term trains only the mean.
            c = (a.cov_k * ls_p.detach().exp()).clamp(min=1e-6)
            cov = 1.0 - torch.exp(-((mu_p - tgt) / c) ** 2)
            loss = loss + a.lam_cov * cov.mean()
        if a.lam_nll > 0:
            r = (tgt - mu_p).detach()
            loss = loss + a.lam_nll * (0.5 * (r / ls_p.exp()) ** 2 + ls_p).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()

        if step % 500 == 0 or step == a.steps - 1:
            net.eval()
            with torch.no_grad():
                mus = []
                for i in range(0, xv.shape[0], 64):
                    xnv = (xv[i:i + 64] - mean) / std
                    m_, _ = net(xnv)
                    mus.append(m_.float() * std + mean)
                pv = torch.cat(mus, 0)
                v = rel_l2(pv, yv).item()
                vt = tke_loss(pv, yv).item()
            net.train()
            hist.append(dict(step=step, train=float(loss.item()), val_rel_l2=v,
                             val_tke=vt))
            print(f"step {step:6d} loss {loss.item():.4f} | val relL2 {v:.5f} "
                  f"(score {100/(1+0.5*v):.3f}) tke {vt:.4f} "
                  f"(score {100/(1+0.5*vt):.3f}) | {time.time()-t0:.0f}s", flush=True)
            crit = -(0.737 * (100 / (1 + 0.5 * v)) + 0.123 * (100 / (1 + 0.5 * vt)))
            if crit < best:
                best, best_v, best_step = crit, v, step
                torch.save(net.state_dict(), os.path.join(OUT, "model.pth"))

    # best_val_rel_l2 is the val rel_l2 of the saved checkpoint, which the
    # composite criterion picks (it is not necessarily the minimum seen).
    bh = [r for r in hist if r["step"] == best_step]
    json.dump(dict(args=vars(a), hist=hist, best_val_rel_l2=best_v,
                   best_val_rel_l2_score=100 / (1 + 0.5 * best_v),
                   select=a.select, best_step=best_step,
                   best_val_tke=(bh[0]["val_tke"] if bh else None),
                   n_params=n_params(net)),
              open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print(f"SAVED checkpoint from step {best_step} (select={a.select}): "
          f"val rel_l2 {best_v:.5f} -> rel_l2_score {100/(1+0.5*best_v):.3f}"
          + (f", val tke_score {100/(1+0.5*bh[0]['val_tke']):.3f}" if bh else ""))


if __name__ == "__main__":
    main()
