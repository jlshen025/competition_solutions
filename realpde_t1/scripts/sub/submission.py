"""RealPDE Track 1 (Sim2Real) submission.

Design notes:

* **Everything expensive happens outside the timed region.** Model
  construction and checkpoint loading are not timed; only inference is. The
  engine is therefore built, moved to the GPU and warmed up at import time and
  in ``SubmissionModel.__init__``; ``predict`` runs pure inference.
* **CUDA graphs per batch-size bucket.** The forward is captured once per
  bucket size in __init__ and replayed in predict, replacing a few hundred
  kernel launches with one replay. Any n is split into captured buckets
  (e.g. 5 -> 4 + 1), so no batch is padded. Each graph is checked against eager
  execution and dropped if it diverges.
* **Fused ensemble trunk** (config ``fused: 2``): the members share one
  architecture, so their trunks run as one grouped-convolution stack, with each
  block's 1x1 conv folded exactly into its 3x3 conv. The trunk precision is
  ``fwd_dtype``; every statistic after the trunk is computed in fp32.
* **Bounds are computed on the GPU.** The per-element half-width is a
  bucketised lookup on the predictive scale.
* **No cross-window information is used**: every window's prediction and bounds
  depend on that window only.
* Imports are limited to numpy + torch. Nothing outside this archive is read.
"""
from __future__ import annotations

import json
import os
import sys
import time as _time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

from rpde_model import Net  # noqa: E402

CFG = json.load(open(os.path.join(_HERE, "config.json")))
BATCH = int(CFG.get("batch", 16))       # largest captured graph shape
T_OUT = 20
BUCKETS = sorted({b for b in (1, 2, 3, 4, 5, 6, 8, 12, 16) if 0 < b <= BATCH})
# Pinned host output pool: HOST_SLOTS slots of up to HOST_N samples, allocated
# in __init__. Calls with larger n, or past the pool, use pageable host arrays
# (slower, same values). _MAX_CHUNK bounds the device staging buffer.
HOST_N = 64
HOST_SLOTS = 4
_MAX_CHUNK = max(2048, HOST_N)
# Output dtype of the returned arrays; the scorer casts them to float32.
_ODT = str(CFG.get("out_dtype", "float32"))
OUT_DT = {"float32": torch.float32, "float16": torch.float16}[_ODT]
OUT_NP = {"float32": np.float32, "float16": np.float16}[_ODT]
# Precision of the member trunks only; blending, variance head and bounds are fp32.
FWD_DT = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}[str(CFG.get("fwd_dtype", "float32"))]
FUSED = int(CFG.get("fused", 0))
if FUSED not in (0, 2):
    raise ValueError(f"unsupported fused mode {FUSED}")
# Archive-format fields that this code supports only at their neutral values.
if (float(CFG.get("tsharp", 0.0)) != 0.0 or CFG.get("alpha") is not None
        or CFG.get("zero_mask") or CFG.get("mask_width")
        or float(CFG.get("h_floor", 0.0)) != 0.0 or CFG.get("cascade")):
    raise ValueError("config uses options this submission does not implement")
# Variance-head gain clip range.
VH_LO, VH_HI = 0.2, 5.0

_FAST_HIT = [0]
_NCALL = [0]
_BUILD_S = [0.0]
_T_IMPORT = _time.time()
_T_LAST = [0.0]

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def _build_net(spec):
    """Net from a config["nets"] entry. Besides width/depth/delta/ar_m the
    entry carries architecture fields that are fixed in this release (m1/m2
    are the unused spectral mode counts)."""
    fixed = dict(spectral=False, multiscale=False, tshare=0, n_glob=0, n_aux=0,
                 ar_detach=0)
    for k, v in fixed.items():
        if spec.get(k, v) != v:
            raise ValueError(f"unsupported net option {k}={spec[k]!r}")
    return Net(width=spec["width"], depth=spec["depth"],
               delta=bool(spec.get("delta", False)), ar_m=int(spec.get("ar_m", 0)))


def _gconv(convs, shared_in=False):
    """Stack K same-shape Conv2d's into one grouped Conv2d (groups=K), or, with
    shared_in, into one plain Conv2d whose K*cout outputs read the same input."""
    c0 = convs[0]
    K = len(convs)
    cin = c0.in_channels * (1 if shared_in else K)
    g = torch.nn.Conv2d(cin, c0.out_channels * K, c0.kernel_size,
                        stride=c0.stride, padding=c0.padding,
                        dilation=c0.dilation, groups=1 if shared_in else K,
                        bias=c0.bias is not None).to(c0.weight.device)
    with torch.no_grad():
        g.weight.copy_(torch.cat([c.weight for c in convs], 0))
        if c0.bias is not None:
            g.bias.copy_(torch.cat([c.bias for c in convs], 0))
    return g


class _FusedBlock(torch.nn.Module):
    """K members' Blocks as grouped convs. pw(x) + lo(x) is one convolution
    with the 1x1 kernel added to the 3x3 centre tap, so pw is folded into lo
    (exact up to float summation order). GroupNorm(8, w) per member equals
    GroupNorm(8K, Kw) on the fused channels, since groups never straddle a
    member."""

    def __init__(self, blocks):
        super().__init__()
        b0 = blocks[0]
        K = len(blocks)
        pw = _gconv([b.pw for b in blocks])
        self.lo = _gconv([b.lo for b in blocks])
        self.lo2 = _gconv([b.lo2 for b in blocks])
        assert self.lo.kernel_size == (3, 3) and self.lo.padding == (1, 1)
        with torch.no_grad():
            self.lo.weight[:, :, 1, 1] += pw.weight[:, :, 0, 0]
            self.lo.bias += pw.bias
        n0 = b0.n
        self.n = torch.nn.GroupNorm(n0.num_groups * K, n0.num_channels * K,
                                    eps=n0.eps).to(n0.weight.device)
        with torch.no_grad():
            self.n.weight.copy_(torch.cat([b.n.weight for b in blocks], 0))
            self.n.bias.copy_(torch.cat([b.n.bias for b in blocks], 0))

    def forward(self, x):
        y = self.lo(x)
        y = y + self.lo2(x)
        return x + self.n(torch.nn.functional.gelu(y))


class _FusedTrunk(torch.nn.Module):
    """K members' lift + blocks; `shared_in` = every member reads one input."""

    def __init__(self, members, shared_in):
        super().__init__()
        m0 = members[0]
        self.shared, self.c = shared_in, m0.c
        self.lift = _gconv([m.lift for m in members], shared_in=shared_in)
        self.blocks = torch.nn.ModuleList(
            [_FusedBlock([m.blocks[d] for m in members]) for d in range(len(m0.blocks))])
        self._grid = m0._grid

    def forward(self, ctxs):
        cl = [ctxs] if self.shared else ctxs
        B, T, H, W, C = cl[0].shape
        gr = self._grid(B, H, W, cl[0].device, cl[0].dtype)
        z = torch.cat([t for xc in cl for t in
                       (xc.permute(0, 1, 4, 2, 3).reshape(B, T * self.c, H, W), gr)], 1)
        z = self.lift(z)
        for b in self.blocks:
            z = b(z)
        return z


class _FusedHeads(torch.nn.Module):
    """K members' mu/log-scale heads on their slice of a fused trunk output."""

    def __init__(self, members):
        super().__init__()
        m0 = members[0]
        self.K, self.c, self.h_out, self.delta = len(members), m0.c, m0.h_out, m0.delta
        self.h1 = _gconv([m.head[0] for m in members])
        self.h2 = _gconv([m.head[2] for m in members])
        self.s1 = _gconv([m.shead[0] for m in members])
        self.s2 = _gconv([m.shead[2] for m in members])

    def forward(self, z, lasts):
        # lasts: per member (B,1,H,W,c) last context frame (delta anchor)
        B, _, H, W = z.shape
        mu = self.h2(torch.nn.functional.gelu(self.h1(z)))
        ls = self.s2(torch.nn.functional.gelu(self.s1(z)))
        co = self.h_out * self.c
        outs = []
        for k in range(self.K):
            m = mu[:, k * co:(k + 1) * co].reshape(B, self.h_out, self.c, H, W).permute(0, 1, 3, 4, 2)
            if self.delta:
                m = m + lasts[k].expand(-1, self.h_out, -1, -1, -1)
            s_ = ls[:, k * co:(k + 1) * co].reshape(B, self.h_out, self.c, H, W).permute(0, 1, 3, 4, 2)
            outs.append((m, s_.clamp(-9.0, 2.0)))
        return outs


class _FusedEnsemble(torch.nn.Module):
    """Same outputs as [n(x) for n in nets], in the same order. Step 1 runs
    all members as one shared-input trunk (heads split by output width); each
    later rollout step runs the rollout (ar_m) members as one grouped trunk."""

    def __init__(self, nets):
        super().__init__()
        m0 = nets[0]
        for n in nets:
            assert n.lift.weight.shape == m0.lift.weight.shape and len(n.blocks) == len(m0.blocks)
        self.nets_n = len(nets)
        self.dir = [i for i, n in enumerate(nets) if not n.ar_m]
        self.ar = [i for i, n in enumerate(nets) if n.ar_m]
        if self.ar:
            assert len({nets[i].ar_m for i in self.ar}) == 1
        self.ar_m = nets[self.ar[0]].ar_m if self.ar else 0
        self.t_out = m0.t_out
        self.w = m0.lift.out_channels
        self.t1 = _FusedTrunk([nets[i] for i in self.dir + self.ar], True)
        self.hd = _FusedHeads([nets[i] for i in self.dir]) if self.dir else None
        self.ha = _FusedHeads([nets[i] for i in self.ar]) if self.ar else None
        self.ta = (_FusedTrunk([nets[i] for i in self.ar], False)
                   if self.ar and self.t_out // self.ar_m > 1 else None)

    def forward(self, x):
        xc = x[..., :2]
        last = xc[:, -1:]
        z = self.t1(xc)
        nd = len(self.dir) * self.w
        outs = [None] * self.nets_n
        if self.dir:
            for i, o in zip(self.dir, self.hd(z[:, :nd], [last] * len(self.dir))):
                outs[i] = o
        if self.ar:
            o1 = self.ha(z[:, nd:], [last] * len(self.ar))
            mus = [[o[0]] for o in o1]
            lss = [[o[1]] for o in o1]
            ctx = [torch.cat([xc[:, self.ar_m:], o[0]], 1) for o in o1]
            for _ in range(self.t_out // self.ar_m - 1):
                ok = self.ha(self.ta(ctx), [c[:, -1:] for c in ctx])
                for k, o in enumerate(ok):
                    mus[k].append(o[0])
                    lss[k].append(o[1])
                ctx = [torch.cat([c[:, self.ar_m:], o[0]], 1) for c, o in zip(ctx, ok)]
            for k, i in enumerate(self.ar):
                outs[i] = (torch.cat(mus[k], 1), torch.cat(lss[k], 1))
        return outs


class _Engine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = self.device
        # The ensemble averages the member means and folds member disagreement
        # into the predictive scale.
        self.nets = []
        for spec, ck in zip(CFG["nets"], CFG["checkpoints"]):
            n = _build_net(spec).to(d).eval()
            n.load_state_dict(torch.load(os.path.join(_HERE, ck), map_location="cpu"))
            for p in n.parameters():
                p.requires_grad_(False)
            # Cast after load_state_dict so the checkpoint stays a plain fp32
            # state_dict on disk.
            if FWD_DT is not torch.float32 and d.type == "cuda":
                n.to(FWD_DT)
            self.nets.append(n)

        self._fused = None
        if FUSED and len(self.nets) > 1:
            # weight surgery in fp32 (Module.float() is in place), then cast
            _src = [n.float() for n in self.nets]
            self._fused = _FusedEnsemble(_src).to(d).eval()
            if FWD_DT is not torch.float32:
                self._fused.to(FWD_DT)
            for p in self._fused.parameters():
                p.requires_grad_(False)
        self.mean = torch.as_tensor(np.asarray(CFG["mean"], np.float32), device=d)
        self.std = torch.as_tensor(np.asarray(CFG["std"], np.float32), device=d)
        wt = CFG["width_table"]
        e = np.asarray(wt["edges"], np.float32)
        self.centers = torch.as_tensor(0.5 * (e[:-1] + e[1:]), device=d)
        self.htab = torch.as_tensor(np.asarray(wt["h"], np.float32), device=d)
        self.lam = float(CFG.get("lam", 1.0))
        self.kappa = float(CFG.get("kappa", 1.0))
        self.beta = float(CFG.get("beta", 1.0))
        # Per-pixel variance head (torch transcription of rpde/vhead_core.py):
        # rescales the prediction's temporal anomaly toward an estimate of the
        # true anomaly variance. Absent, or both doses 0, is an exact no-op.
        _vh = CFG.get("vhead", None) or {}
        self.vh_dose = float(_vh.get("dose", 0.0))
        if _vh and _vh.get("kind") != "poly2":
            raise ValueError(f"unsupported vhead kind {_vh.get('kind')!r}")
        # Gate: apply the head only where the pixel's predicted anomaly
        # variance is at or above its within-window `gate` quantile (by rank).
        self.vh_gate = float(_vh.get("gate", 0.0))
        # Asymmetric dose: dose_up where the head amplifies (g >= 1), dose_dn
        # where it shrinks; both default to `dose`.
        self.vh_dose_up = float(_vh.get("dose_up", self.vh_dose))
        self.vh_dose_dn = float(_vh.get("dose_dn", self.vh_dose))
        self.vh_w = (torch.as_tensor(np.asarray(_vh["w"], np.float32), device=d)
                     if (self.vh_dose_up != 0.0 or self.vh_dose_dn != 0.0)
                     and _vh.get("w") else None)
        self.H, self.W = int(CFG.get("H", 32)), int(CFG.get("W", 64))
        self.out_buf = None
        self.host_pool = []
        self._call = 0
        _b0 = _time.time()
        self._warm()
        _BUILD_S[0] = _time.time() - _T_IMPORT
        print("[submission] diag build warm=%.2fs total_since_import=%.2fs "
              "graph=%d pool=%d host_n=%d batch=%d buckets=%s"
              % (_time.time() - _b0, _BUILD_S[0],
                 int(self.graph is not None), len(self.host_pool), HOST_N, BATCH,
                 ",".join(str(b) for b in sorted(self.graphs)) or "-"), flush=True)

    def _warm(self):
        z = torch.zeros(BATCH, T_OUT, self.H, self.W, 3, device=self.device)
        for _ in range(3):
            self._core(z)
        self.graph = None
        self.graphs = {}
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            try:
                self._capture(z)
            except Exception:      # any capture problem -> plain eager
                self.graph = None
                torch.cuda.synchronize()
            # Extra bucket shapes; a failed capture leaves that bucket absent
            # and every successfully captured one usable.
            if self.graph is not None:
                self.graphs[BATCH] = (self.graph, self.g_in, self.g_mu, self.g_sh)
                for b in BUCKETS:
                    if b == BATCH:
                        continue
                    try:
                        self._capture_at(b)
                    except Exception:
                        torch.cuda.synchronize()
        self._warm_finish()

    def _capture_at(self, b):
        """Capture one extra graph at batch size `b` and file it in self.graphs."""
        keep = (self.graph, self.g_in, self.g_mu, self.g_sh)
        z = torch.zeros(b, T_OUT, self.H, self.W, 3, device=self.device)
        try:
            self._capture(z)
            self.graphs[b] = (self.graph, self.g_in, self.g_mu, self.g_sh)
        finally:
            (self.graph, self.g_in, self.g_mu, self.g_sh) = keep

    def _warm_finish(self):
        """Pay the output path's first-use costs (device staging buffer, pinned
        host pool, the copy kernels) here, outside the timed region. Channel 2
        (pressure) of the staging buffer is zeroed once and never written."""
        try:
            if self.out_buf is None:
                self.out_buf = torch.zeros(
                    (3, _MAX_CHUNK, T_OUT, self.H, self.W, 3),
                    dtype=torch.float32, device=self.device)
            # Page-locking can be refused by the host; that degrades to the
            # pageable path, never fails.
            if self.device.type == "cuda" and not self.host_pool:
                try:
                    self.host_pool = [
                        torch.zeros((3, HOST_N, T_OUT, self.H, self.W, 3),
                                    dtype=OUT_DT, pin_memory=True)
                        for _ in range(HOST_SLOTS)]
                except Exception:
                    self.host_pool = []
            z = np.zeros((BATCH, T_OUT, self.H, self.W, 3), np.float32)
            for _ in range(2):
                _finish(*self.run(z), BATCH, self.out_buf, self._host())
            # also exercise the small-n route (bucket decomposition)
            for m in (5, 1):
                zm = np.zeros((m, T_OUT, self.H, self.W, 3), np.float32)
                _finish(*self.run(zm), m, self.out_buf, self._host())
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        except Exception:
            # predict then allocates its output buffers per call
            self.out_buf = None
        # warm-up must not consume the host pool
        self._call = 0

    def _host(self):
        """Next pinned destination, or None once the pool is spent, so a later
        call can never overwrite the arrays an earlier call returned."""
        if not self.host_pool or self._call >= len(self.host_pool):
            return None
        h = self.host_pool[self._call]
        self._call += 1
        return h

    def _capture(self, z):
        """Capture the whole forward as a CUDA graph at z's batch size, then
        check a replay on random input against eager execution."""
        self.g_in = z.clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._core(self.g_in)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.g_mu, self.g_sh = self._core(self.g_in)
        probe = torch.randn_like(self.g_in) * 0.1
        ref_mu, ref_sh = self._core(probe)
        self.g_in.copy_(probe)
        g.replay()
        torch.cuda.synchronize()
        if not (torch.allclose(self.g_mu, ref_mu, atol=1e-4, rtol=1e-3)
                and torch.allclose(self.g_sh, ref_sh, atol=1e-4, rtol=1e-3)):
            raise RuntimeError("cuda graph output diverged from eager")
        self.graph = g

    def _fwd(self, xb):
        g = self.graphs.get(xb.shape[0])
        if g is not None:
            gr, gin, gmu, gsh = g
            gin.copy_(xb)
            gr.replay()
            return gmu.clone(), gsh.clone()
        return self._core(xb)

    def _core(self, xb):
        """Members' forward -> ensemble mean and predictive scale
        sqrt(mean_m exp(logs_m)^2 + kappa * Var_m(mu_m)), physical units."""
        xn = (xb[..., :2] - self.mean) / self.std
        with torch.no_grad():
            if self._fused is not None:
                if FWD_DT is torch.float32:
                    outs = self._fused(xn)
                else:
                    outs = [[t.float() for t in o] for o in self._fused(xn.to(FWD_DT))]
            elif FWD_DT is torch.float32:
                outs = [n(xn) for n in self.nets]
            else:
                # cast back immediately so every statistic below is fp32
                xh = xn.to(FWD_DT)
                outs = [[t.float() for t in n(xh)] for n in self.nets]
        lg = torch.log(self.std).mean()
        mus = torch.stack([o[0] for o in outs], 0) * self.std + self.mean
        var = torch.stack([(o[1] + lg).exp() ** 2 for o in outs], 0).mean(0)
        mu = mus.mean(0)
        if mus.shape[0] > 1:      # member disagreement is extra predictive scale
            var = var + self.kappa * mus.var(0, unbiased=False)
        return mu, var.clamp_min(1e-12).sqrt()

    @staticmethod
    def _smooth3(a):
        o = torch.zeros_like(a)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                o = o + torch.roll(a, shifts=(dy, dx), dims=(2, 3))
        return o / 9.0

    def _vhead(self, mu, shat, xt):
        """Rescale mu's temporal anomaly by the fitted variance head. Features
        are read off this window and the model's own output on it only."""
        eps = 1e-12
        C = mu.shape[-1]
        m0 = mu.mean(dim=1, keepdim=True)
        ap = mu - m0
        vp = (ap * ap).mean(dim=1, keepdim=True)
        sh = shat[..., :C]
        msh2 = (sh * sh).mean(dim=1, keepdim=True)
        xin = xt[..., :C]
        xa = xin - xin.mean(dim=1, keepdim=True)
        vin = (xa * xa).mean(dim=1, keepdim=True)
        obsi = (xin == 0.0).all(dim=1, keepdim=True)
        H, W = vp.shape[2], vp.shape[3]
        yy = torch.linspace(-1.0, 1.0, H, device=mu.device,
                            dtype=mu.dtype).view(1, 1, H, 1, 1).expand_as(vp)
        xx = torch.linspace(-1.0, 1.0, W, device=mu.device,
                            dtype=mu.dtype).view(1, 1, 1, W, 1).expand_as(vp)
        vwin = vp.mean(dim=(2, 3, 4), keepdim=True).expand_as(vp)
        f = torch.stack([torch.log(vp + eps), torch.log(msh2 + eps),
                         torch.log(vin + eps),
                         torch.log(self._smooth3(vin) + eps),
                         yy, xx, torch.log(vwin + eps)], dim=-1)
        cols = [f] + [f[..., i:i + 1] * f[..., i:] for i in range(f.shape[-1])]
        f = torch.cat(cols, dim=-1)                       # degree-2 polynomial
        lv = (f * self.vh_w[:-1]).sum(-1) + self.vh_w[-1]
        g = torch.sqrt(torch.clamp(torch.exp(lv), min=eps) / (vp + eps))
        g = torch.clamp(g, VH_LO, VH_HI)
        g = torch.where(obsi.expand_as(g), torch.ones_like(g), g)
        if self.vh_gate > 0.0:
            N = g.shape[0]
            vf = vp.reshape(N, -1)
            okf = obsi.expand_as(vp).reshape(N, -1).logical_not()
            big = torch.where(okf, vf, torch.full_like(vf, float("inf")))
            srt, _ = torch.sort(big, dim=1)
            n_ok = okf.sum(dim=1)
            # threshold = the floor(gate * (n_ok - 1))-th smallest admissible
            # value; float64 so the index cannot round across an integer
            idx = torch.floor(self.vh_gate
                              * torch.clamp(n_ok - 1, min=0).double()).long()
            idx = torch.clamp(idx, 0, srt.shape[1] - 1)
            thr = srt.gather(1, idx.view(-1, 1))
            keep = (vf >= thr).reshape(vp.shape)
            g = torch.where(keep.expand_as(g), g, torch.ones_like(g))
        # gated-out and obstructed pixels carry g == 1, so either dose leaves
        # them unchanged
        t = torch.where(g >= 1.0,
                        torch.full_like(g, self.vh_dose_up),
                        torch.full_like(g, self.vh_dose_dn))
        return m0 + ap * (1.0 + t * (g - 1.0))

    def _halfwidth(self, shat):
        """lam * table(shat), linear between bucket centres, clamped at the ends."""
        idx = torch.bucketize(shat.contiguous(), self.centers)
        i1 = torch.clamp(idx, 1, self.centers.numel() - 1)
        i0 = i1 - 1
        c0, c1 = self.centers[i0], self.centers[i1]
        h0, h1 = self.htab[i0], self.htab[i1]
        w = torch.clamp((shat - c0) / torch.clamp(c1 - c0, min=1e-20), 0.0, 1.0)
        return torch.clamp(h0 + w * (h1 - h0), min=0.0) * self.lam

    def run(self, x: np.ndarray):
        d = self.device
        n = x.shape[0]
        if x.dtype != np.float32:
            # cast on the device: cheaper than a host cast, same rounding
            xt = torch.from_numpy(np.ascontiguousarray(x)).to(d).float()
        else:
            xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(d)
        mus, lss = [], []
        sizes = sorted(self.graphs)
        i = 0
        while i < n:
            r = n - i
            # largest captured bucket that fits -> no padding; with bucket 1
            # present every n decomposes exactly
            fit = [b for b in sizes if b <= r]
            if fit:
                step = fit[-1]
            else:
                # no graphs captured: eager at BATCH, padded
                step = sizes[0] if sizes else BATCH
            xb = xt[i:i + min(step, r)]
            k = xb.shape[0]
            if k < step:
                xb = torch.cat([xb, xb.new_zeros((step - k,) + xb.shape[1:])], 0)
            m, sh = self._fwd(xb)
            mus.append(m[:k]); lss.append(sh[:k])
            i += k
        mu = torch.cat(mus, 0)
        shat = torch.cat(lss, 0)

        if self.beta != 1.0:       # amplify the model's own temporal anomaly
            m0 = mu.mean(dim=1, keepdim=True)
            mu = m0 + self.beta * (mu - m0)
        if self.vh_w is not None:
            mu = self._vhead(mu, shat, xt)
        half = self._halfwidth(shat)
        return mu, mu - half, mu + half


_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _Engine()
    return _ENGINE


def _finish(pred, lo, hi, n, buf=None, host=None):
    """Enforce the output contract (shapes, finite values, lower <= upper) and
    copy to the host: prediction/lower/upper are written into one device
    buffer, with the pressure channel 0, then copied contiguously to host."""
    lo, hi = torch.minimum(lo, hi), torch.maximum(lo, hi)
    H, W = pred.shape[2], pred.shape[3]
    if host is not None and n <= host.shape[1] and buf is not None \
            and buf.shape[1] >= n:
        # Fast path: copy into a pinned, preallocated host slot and return
        # numpy views of it.
        _FAST_HIT[0] = 1
        out = buf[:, :n]
        out[0, ..., :2] = pred
        out[1, ..., :2] = lo
        out[2, ..., :2] = hi
        torch.nan_to_num_(out, nan=0.0, posinf=0.0, neginf=0.0)
        # narrow on the device so the transfer shrinks too
        src = out if OUT_DT is torch.float32 else out.to(OUT_DT)
        for k in range(3):
            host[k, :n].copy_(src[k], non_blocking=True)
        # the caller reads these pages as plain memory
        torch.cuda.synchronize()
        a = host.numpy()
        p, l, u = a[0][:n], a[1][:n], a[2][:n]
        if not (p.shape == l.shape == u.shape and p.shape[0] == n):
            raise RuntimeError(f"output contract violated: {p.shape} vs n={n}")
        return {"prediction": p, "lower": l, "upper": u}

    # General path: any n, in chunks of at most _MAX_CHUNK samples.
    outs = [np.empty((n, T_OUT, H, W, 3), OUT_NP) for _ in range(3)]
    for i in range(0, n, _MAX_CHUNK):
        j = min(i + _MAX_CHUNK, n)
        if buf is not None and buf.shape[1] >= j - i:
            out = buf[:, :j - i]
        else:
            out = torch.zeros((3, j - i, T_OUT, H, W, 3), dtype=torch.float32,
                              device=pred.device)
        out[0, ..., :2] = pred[i:j]
        out[1, ..., :2] = lo[i:j]
        out[2, ..., :2] = hi[i:j]
        torch.nan_to_num_(out, nan=0.0, posinf=0.0, neginf=0.0)
        src = out if OUT_DT is torch.float32 else out.to(OUT_DT)
        for k in range(3):
            torch.from_numpy(outs[k][i:j]).copy_(src[k])
    p, l, u = outs
    if not (p.shape == l.shape == u.shape and p.shape[0] == n):
        raise RuntimeError(f"output contract violated: {p.shape} vs n={n}")
    return {"prediction": p, "lower": l, "upper": u}


class SubmissionModel:
    def __init__(self, submission_dir=None, device="cpu"):
        self.engine = _engine()

    def load_checkpoint(self, path, device=None):
        # The engine loaded this archive's checkpoints at import time.
        return None

    def predict(self, input_array, metadata=None):
        return predict(input_array, metadata)


def predict(input_array, metadata=None):
    # no dtype here: a non-float32 input is cast on the device inside run()
    x = np.asarray(input_array)
    e = _engine()
    _FAST_HIT[0] = 0
    _NCALL[0] += 1
    t0 = _time.time()
    mu, lo, hi = e.run(x)
    if e.device.type == "cuda":
        torch.cuda.synchronize()
    t1 = _time.time()
    out = _finish(mu, lo, hi, x.shape[0], e.out_buf, e._host())
    t2 = _time.time()
    # One line per call with timings and shapes only (never data values);
    # gap = time since the previous call returned.
    gap = (t0 - _T_LAST[0]) * 1e3 if _T_LAST[0] else -1.0
    print("[submission] diag call=%d n=%d dt=%s fast=%d t_run=%.1fms "
          "t_out=%.1fms t_tot=%.1fms per_sample=%.3fms gap=%.1fms "
          "wall=%.3f build=%.3f"
          % (_NCALL[0], x.shape[0], x.dtype, _FAST_HIT[0], (t1 - t0) * 1e3,
             (t2 - t1) * 1e3, (t2 - t0) * 1e3, (t2 - t0) * 1e3 / max(1, x.shape[0]),
             gap, t0 - _T_IMPORT, _BUILD_S[0]), flush=True)
    _T_LAST[0] = _time.time()
    return out


try:                  # build eagerly: import time is outside the timed region
    _engine()
except Exception:     # degrade to a lazy build rather than failing the import
    _ENGINE = None
