"""Per-pixel variance head: numpy reference used by vfit.py.

A ridge-fitted estimator of log var_t(true temporal anomaly) at each pixel lets
the prediction's temporal anomaly be rescaled toward the truth's second moment.
The archive's submission.py carries a torch transcription of `features` and
`design`, plus the per-window gate and the asymmetric dose it applies at
inference.

Inputs, in this order of computation:
    mu    the ensemble mean after the beta amplification
    shat  the deployed scale sqrt(var + kappa * var_across_members)
    x     the raw input window (N, 20, H, W, >=2)
All features are functions of that one window and the model's output on it;
nothing is pooled across the samples of a predict() call.
"""
import numpy as np

EPS = 1e-12
FEATS = ["lvp", "lmsh2", "lvin", "lvin3", "yy", "xx", "lvwin"]


def _smooth3(a):
    """3x3 box mean over the H,W axes of an (N,1,H,W,C) array (periodic)."""
    o = np.zeros_like(a)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            o += np.roll(np.roll(a, dy, axis=2), dx, axis=3)
    return o / 9.0


def features(mu, shat, x):
    """-> feats (N,1,H,W,C,7), vp (N,1,H,W,C), obsi (N,1,H,W,C) bool.

    vp is the predicted anomaly variance; obsi marks pixels that are zero
    through the whole input window (PIV obstruction)."""
    mu = np.asarray(mu, np.float32)
    C = mu.shape[-1]
    ap = mu - mu.mean(axis=1, keepdims=True)
    vp = (ap ** 2).mean(axis=1, keepdims=True)
    msh2 = (np.asarray(shat, np.float32)[..., :C] ** 2).mean(axis=1, keepdims=True)
    xin = np.asarray(x, np.float32)[..., :C]
    vin = ((xin - xin.mean(axis=1, keepdims=True)) ** 2).mean(axis=1, keepdims=True)
    obsi = (xin == 0.0).all(axis=1)[:, None]
    H, W = vp.shape[2], vp.shape[3]
    yy = np.broadcast_to(np.linspace(-1, 1, H, dtype=np.float32
                                     ).reshape(1, 1, H, 1, 1), vp.shape)
    xx = np.broadcast_to(np.linspace(-1, 1, W, dtype=np.float32
                                     ).reshape(1, 1, 1, W, 1), vp.shape)
    vwin = np.broadcast_to(vp.mean(axis=(2, 3, 4), keepdims=True), vp.shape)
    feats = np.stack([np.log(vp + EPS), np.log(msh2 + EPS), np.log(vin + EPS),
                      np.log(_smooth3(vin) + EPS), yy, xx,
                      np.log(vwin + EPS)], axis=-1)
    return feats.astype(np.float32), vp, obsi


def design(F, kind="poly2"):
    """(M,7) -> degree-2 polynomial design matrix, WITHOUT the intercept."""
    if kind != "poly2":
        raise ValueError("unknown vhead kind: %r" % (kind,))
    cols = [F] + [F[:, i:i + 1] * F[:, i:] for i in range(F.shape[1])]
    return np.concatenate(cols, axis=1)


def gain(mu, shat, x, w, dose, lo=0.2, hi=5.0):
    """Multiplier 1 + dose*(g - 1) on mu's temporal anomaly, with
    g = sqrt(vhat / vp) clipped to [lo, hi] and g = 1 on obstructed pixels.
    dose 0 returns exactly ones."""
    feats, vp, obsi = features(mu, shat, x)
    if float(dose) == 0.0:
        return np.ones_like(vp)
    w = np.asarray(w, np.float64)
    G = design(feats.reshape(-1, feats.shape[-1]).astype(np.float64))
    lv = G @ w[:-1] + w[-1]
    vhat = np.exp(lv).reshape(vp.shape)
    g = np.sqrt(np.maximum(vhat, EPS) / (vp + EPS))
    g = np.clip(g, lo, hi)
    g = np.where(np.broadcast_to(obsi, g.shape), 1.0, g)
    return (1.0 + float(dose) * (g - 1.0)).astype(np.float32)


def apply_to(mu, shat, x, cfg):
    """Rescale mu with the `vhead` block {w, kind, dose} of a config. Absent,
    dose 0 or no coefficients => mu returned unchanged."""
    vh = (cfg or {}).get("vhead") or {}
    dose = float(vh.get("dose", 0.0))
    if dose == 0.0 or not vh.get("w"):
        return mu
    if vh.get("kind") != "poly2":
        raise ValueError("unknown vhead kind: %r" % (vh.get("kind"),))
    gg = gain(mu, shat, x, vh["w"], dose)
    m0 = mu.mean(axis=1, keepdims=True)
    return m0 + (mu - m0) * gg
