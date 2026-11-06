"""SPS interval-width table.

The scorer rewards each element independently:

    reward = exp(-(upper - lower) / sigma) * 1{lower <= target <= upper}

so for a symmetric interval of half-width h the choice decouples per element:
maximise exp(-2h/sigma) * P(|residual| <= h). Given a predicted per-element
scale `shat`, elements are bucketed by `shat` and each bucket takes the exact
empirical argmax (no Gaussian assumption): for the k-th smallest of n sorted
|residuals| r_k the objective is exp(-2 r_k / sigma) * k / n. The result is a
lookup table h(shat), interpolated linearly between bucket centres.
"""
import numpy as np

SIGMA = 0.0563870259     # the scorer's fixed sigma_global


def fit_width_table(resid, shat, scored, nbuck=64, sigma=SIGMA, min_count=200):
    """resid, shat, scored: same shape. Returns (edges, h_table)."""
    r = np.abs(resid[scored]).astype(np.float64).ravel()
    s = shat[scored].astype(np.float64).ravel()
    qs = np.linspace(0, 100, nbuck + 1)
    edges = np.unique(np.percentile(s, qs))
    if len(edges) < 3:
        edges = np.array([s.min(), s.max() + 1e-12])
    bid = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
    h = np.zeros(len(edges) - 1)
    for b in range(len(edges) - 1):
        rb = r[bid == b]
        if rb.size < min_count:
            h[b] = np.nan
            continue
        rb = np.sort(rb)
        n = rb.size
        k = np.arange(1, n + 1)
        obj = np.exp(-2.0 * rb / sigma) * (k / n)
        h[b] = rb[int(np.argmax(obj))]
    # fill under-populated buckets by interpolation over bucket centres
    ok = ~np.isnan(h)
    if ok.sum() == 0:
        h[:] = 0.01
    else:
        cen = 0.5 * (edges[:-1] + edges[1:])
        h = np.interp(cen, cen[ok], h[ok])
    return edges, h


def apply_width_table(shat, edges, h):
    """Piecewise-linear interpolation of the fitted table between bucket centres."""
    cen = 0.5 * (edges[:-1] + edges[1:])
    return np.interp(shat.ravel(), cen, h).reshape(shat.shape).astype(np.float32)
