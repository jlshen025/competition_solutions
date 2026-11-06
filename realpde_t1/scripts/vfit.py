"""Fit the variance-head coefficients on the ship nets.

The head is a ridge regression (degree-2 polynomial in the 7 features of
rpde/vhead_core.py) of log var_t(true anomaly) per pixel. It is fitted on the
all-data ship nets that go into the archive, over all training windows, after
the archive's beta amplification and with its kappa.

results.json carries the `vhead` block ready for assemble.py, plus two
sanity reads: the in-sample correlation of the fitted log-gain with the
oracle one, and max|dmu| at dose 0 (must be 0) and at --dose.

    EXP_OUTPUT_DIR=<dir> python scripts/vfit.py --runs r1+r2+... --zip base.zip
"""
import argparse, json, os, sys, zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rpde import data as D                                    # noqa: E402
from rpde import vhead_core as V                              # noqa: E402
from eval_ens import load, infer                              # noqa: E402

OUT = os.environ.get("EXP_OUTPUT_DIR", ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="'+'-separated ship run dirs")
    ap.add_argument("--zip", required=True, help="archive supplying beta/kappa")
    ap.add_argument("--kind", default="poly2", choices=["poly2"])
    ap.add_argument("--dose", type=float, default=0.3)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--stride", type=int, default=20)
    a = ap.parse_args()

    cfg = json.loads(zipfile.ZipFile(a.zip).read("config.json"))
    beta = float(cfg.get("beta", 1.0))
    kap = float(cfg.get("kappa", 1.0))
    if cfg.get("zero_mask") or cfg.get("alpha") is not None:
        raise ValueError("zero_mask / alpha archives are not supported")

    arr, idx = D.load("real")
    wins = D.windows(idx, stride=a.stride)
    print(f"{len(wins)} training windows at stride {a.stride}", flush=True)
    nets = [load(r) for r in a.runs.split("+")]

    XtX = None
    Xty = None
    cs, co = [], []
    B = 64
    for s in range(0, len(wins), B):
        x, y = D.gather(arr, idx, wins[s:s + B])
        MU, VAR = infer(nets, x)
        mu = MU.mean(0)
        m0 = mu.mean(axis=1, keepdims=True)
        mu = m0 + beta * (mu - m0)
        vmem = MU.var(0) if MU.shape[0] > 1 else np.zeros_like(VAR)
        shat = np.sqrt(np.maximum(VAR + kap * vmem, V.EPS))

        feats, vp, obsi = V.features(mu, shat, x)
        yt = y[..., :mu.shape[-1]]
        ay = yt - yt.mean(axis=1, keepdims=True)
        vt = (ay ** 2).mean(axis=1, keepdims=True)
        obst = (yt == 0.0).all(axis=1)[:, None]
        ok = (~obst & ~obsi & (vt > 0) & (vp > 0)).ravel()

        G = V.design(feats.reshape(-1, feats.shape[-1])[ok].astype(np.float64),
                     a.kind)
        G = np.column_stack([G, np.ones(len(G))])
        t = np.log(vt + V.EPS).ravel()[ok]
        XtX = G.T @ G if XtX is None else XtX + G.T @ G
        Xty = G.T @ t if Xty is None else Xty + G.T @ t
        cs.append(np.log(np.clip(np.sqrt(vt / (vp + V.EPS)), 1e-3, 50)
                         ).ravel()[ok])
        co.append((feats, vp, obsi, ok))
        if s == 0:
            print(f"  block0: {ok.sum()} of {ok.size} pixels usable", flush=True)

    n = XtX[-1, -1]
    w = np.linalg.solve(XtX + a.ridge * n * np.eye(XtX.shape[0]), Xty)
    print(f"fitted {a.kind}: {len(w)} coefficients over {int(n)} pixels")

    # in-sample correlation with the oracle log-gain (sanity only)
    num = []
    for (feats, vp, obsi, ok), Lo in zip(co, cs):
        Gf = V.design(feats.reshape(-1, feats.shape[-1])[ok].astype(np.float64),
                      a.kind)
        lv = Gf @ w[:-1] + w[-1]
        g = np.sqrt(np.maximum(np.exp(lv), V.EPS)
                    / (vp.ravel()[ok] + V.EPS))
        num.append((np.log(np.clip(g, 1e-3, 50)), Lo))
    Le = np.concatenate([u for u, _ in num])
    Lo = np.concatenate([v for _, v in num])
    corr = float(np.corrcoef(Le, Lo)[0, 1])
    print(f"CORR in-sample log-gain vs oracle = {corr:+.4f}")

    # dose 0 must leave mu unchanged; dose --dose must move it
    x, y = D.gather(arr, idx, wins[:8])
    MU, VAR = infer(nets, x)
    mu = MU.mean(0)
    m0 = mu.mean(axis=1, keepdims=True)
    mu = m0 + beta * (mu - m0)
    vmem = MU.var(0) if MU.shape[0] > 1 else np.zeros_like(VAR)
    shat = np.sqrt(np.maximum(VAR + kap * vmem, V.EPS))
    vh = dict(w=[float(v) for v in w], kind=a.kind, dose=a.dose)
    same = V.apply_to(mu.copy(), shat, x, {"vhead": dict(vh, dose=0.0)})
    noop = float(np.abs(same - mu).max())
    live = V.apply_to(mu.copy(), shat, x, {"vhead": vh})
    moved = float(np.abs(live - mu).max())
    print(f"NOOP max|dmu| at dose 0 = {noop:.3e}   "
          f"LIVE max|dmu| at dose {a.dose} = {moved:.3e}")

    res = dict(vhead=vh, vfit_corr_insample=corr, vfit_noop_maxabs=noop,
               vfit_live_maxabs=moved, n_pixels=int(n), runs=a.runs,
               zip=a.zip)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("VFIT DONE")


if __name__ == "__main__":
    main()
