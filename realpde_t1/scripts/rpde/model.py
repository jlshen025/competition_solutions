"""Fast 2D convolutional operator for RealPDE Track 1.

The field is 32x64 with T_in = T_out = 20, so folding time into channels turns
the problem into a 40 -> 40 channel map on a 2048-pixel grid. That is about
100x cheaper than a 3D operator, and inference time is a scored axis.

Two heads:
  mu    the conditional mean (drives rel_l2 / tke / mvpe)
  logs  a per-element log predictive scale (drives the SPS interval width)
"""
import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    """x + GroupNorm(GELU(pw1x1(x) + conv3x3(x) + conv5x5_dilated2(x)))."""

    def __init__(self, c):
        super().__init__()
        self.pw = nn.Conv2d(c, c, 1)
        self.lo = nn.Conv2d(c, c, 3, padding=1)
        self.lo2 = nn.Conv2d(c, c, 5, padding=4, dilation=2)
        self.n = nn.GroupNorm(8, c)

    def forward(self, x):
        y = self.pw(x) + self.lo(x)
        y = y + self.lo2(x)
        return x + self.n(F.gelu(y))


class Net(nn.Module):
    """(B,T_in,H,W,C) -> mu (B,T_out,H,W,C), logs (B,T_out,H,W,C).

    delta: predict the change from the last observed frame instead of the
        absolute field (in normalised space), so the persistence solution is
        free and capacity goes to the departure from it.
    ar_m: autoregressive rollout. 0 = one direct T_in -> T_out map. Otherwise a
        T_in -> ar_m operator is applied T_out/ar_m times; each step's context
        is the previous context shifted by ar_m frames plus that step's own
        prediction. Training and inference run the same recursion, and
        gradients flow through every step.
    """

    def __init__(self, t_in=20, t_out=20, c=2, width=96, depth=4, delta=False,
                 ar_m=0):
        super().__init__()
        self.t_in, self.t_out, self.c = t_in, t_out, c
        self.ar_m = int(ar_m)
        if self.ar_m and t_out % self.ar_m:
            raise ValueError(f"t_out {t_out} not divisible by ar_m {self.ar_m}")
        self.h_out = self.ar_m or t_out
        self.delta = bool(delta)
        cin, cout = t_in * c, self.h_out * c
        self.lift = nn.Conv2d(cin + 2, width, 1)          # +2 coordinate channels
        self.blocks = nn.ModuleList([Block(width) for _ in range(depth)])
        self.head = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(),
                                  nn.Conv2d(width, cout, 1))
        self.shead = nn.Sequential(nn.Conv2d(width, width // 2, 1), nn.GELU(),
                                   nn.Conv2d(width // 2, cout, 1))
        self.register_buffer("grid", torch.zeros(1), persistent=False)

    def _grid(self, B, H, W, device, dtype):
        ys = torch.linspace(0, 1, H, device=device, dtype=dtype)
        xs = torch.linspace(0, 1, W, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gy, gx], 0).unsqueeze(0).expand(B, -1, -1, -1)

    def forward(self, x):
        xc = x[..., :self.c]
        if not self.ar_m:
            return self._step(xc)
        mus, lss = [], []
        ctx = xc
        for _ in range(self.t_out // self.ar_m):
            mu, ls = self._step(ctx)
            mus.append(mu)
            lss.append(ls)
            ctx = torch.cat([ctx[:, self.ar_m:], mu], 1)
        return torch.cat(mus, 1), torch.cat(lss, 1)

    def _step(self, xc):
        """xc (B,T_in,H,W,c) -> mu, logs, each (B,h_out,H,W,c)."""
        B, T, H, W, _ = xc.shape
        z = xc.permute(0, 1, 4, 2, 3).reshape(B, T * self.c, H, W)
        z = torch.cat([z, self._grid(B, H, W, xc.device, z.dtype)], 1)
        z = self.lift(z)
        for b in self.blocks:
            z = b(z)
        mu = self.head(z).reshape(B, self.h_out, self.c, H, W).permute(0, 1, 3, 4, 2)
        if self.delta:                      # anchor on the last observed frame
            mu = mu + xc[:, -1:].expand(-1, self.h_out, -1, -1, -1)
        ls = self.shead(z).reshape(B, self.h_out, self.c, H, W).permute(0, 1, 3, 4, 2)
        return mu, ls.clamp(-9.0, 2.0)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
