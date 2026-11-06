"""Fit the SPS interval-width table on held-out residuals.

Each CV run (trained with Re 13950 and 22875 held out) is loaded separately,
not blended, and evaluated on the held-out windows after the archive's beta
amplification. Its residuals there are out-of-sample; the table
fitted on them is applied to the all-data ship nets.

Diagnostic cross-fit: fit on one held-out Re and score the expected SPS element
factor F on the other's hardest windows (both directions). The hard slice is the
share of windows whose pooled rel_l2_score matches --target. The reference
table read from --zip (the identity h = shat in the base archive) is scored
beside it at lam 1.00 and 0.65, with the |r| oracle as an upper bound. The
deployed table is then fitted on both Re together.

    EXP_OUTPUT_DIR=<dir> python scripts/newtab.py --runs cv1,cv2,... --zip base.zip
"""
import argparse, json, os, sys, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rpde import data as D
from rpde import sps as S
from eval_ens import load, infer

OUT = os.environ.get("EXP_OUTPUT_DIR", ".")
HOLDOUT_RE = (13950, 22875)
SIG = S.SIGMA


def F_of(r, h, sc):
    """Mean SPS element factor exp(-2h/sigma) * 1{|r| <= h} over scored elements."""
    el = np.where(np.abs(r) <= h, np.exp(-2.0 * h / SIG), 0.0)
    return float(np.sum(el, where=sc, dtype=np.float64) / np.count_nonzero(sc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="comma-separated CV run dirs, each evaluated separately")
    ap.add_argument("--zip", required=True,
                    help="archive supplying beta/kappa and the reference table")
    ap.add_argument("--nbuck", type=int, default=64)
    ap.add_argument("--target", type=float, default=94.951,
                    help="rel_l2_score the hard diagnostic slice is matched to")
    a = ap.parse_args()

    cfg = json.loads(zipfile.ZipFile(a.zip).read("config.json"))
    edges0 = np.asarray(cfg["width_table"]["edges"], np.float32)
    htab0 = np.asarray(cfg["width_table"]["h"], np.float32)
    kappa = float(cfg["kappa"]); beta = float(cfg.get("beta", 1.0))
    if cfg.get("zero_mask") or cfg.get("mask_width"):
        raise ValueError("zero_mask / mask_width archives are not supported")
    print(f"archive {a.zip}: kappa={kappa} beta={beta}", flush=True)

    arr, idx = D.load("real")
    pool = {re_: dict(r=[], sh=[], sc=[], e=[]) for re_ in HOLDOUT_RE}
    for run in [r for r in a.runs.split(",") if r]:
        net = load(run)
        for re_ in HOLDOUT_RE:
            w = D.windows(idx, stride=20, keep=lambda r: r["re"] == re_)
            x, y = D.gather(arr, idx, w)
            MU, VAR = infer([net], x)
            mu = MU.mean(0)
            m0 = mu.mean(axis=1, keepdims=True)
            mu = m0 + beta * (mu - m0)
            pool[re_]["r"].append(y[..., :2] - mu)
            pool[re_]["sh"].append(np.sqrt(VAR))
            pool[re_]["sc"].append(y[..., :2] != 0.0)
            pool[re_]["e"].append(D.rel_l2_ps(mu, y[..., :2], 2))
        print(f"  {run}: done", flush=True)
        del net
    for re_ in HOLDOUT_RE:
        for k in ("r", "sh", "sc", "e"):
            pool[re_][k] = np.concatenate(pool[re_][k], 0)

    lams = np.round(np.arange(0.30, 2.01, 0.05), 3)

    def hard_of(d, frac):
        kk = max(2, int(round(frac * len(d["e"]))))
        o = np.argsort(-d["e"])[:kk]
        return {k: d[k][o] for k in ("r", "sh", "sc", "e")}

    allE = np.concatenate([pool[k]["e"] for k in HOLDOUT_RE])
    cum = np.cumsum(np.sort(allE)[::-1]) / np.arange(1, len(allE) + 1)
    kk = int(np.argmin(np.abs(100.0 / (1 + 0.5 * cum) - a.target))) + 1
    frac = kk / len(allE)
    hard = {re_: hard_of(pool[re_], frac) for re_ in HOLDOUT_RE}
    print(f"difficulty match: frac {frac:.3f} "
          f"(rel_l2_score {100/(1+0.5*cum[kk-1]):.3f})", flush=True)

    # Cross-fit: fit on one Re (all windows), score on the other's hard slice.
    rows = []
    for fit_re in HOLDOUT_RE:
        chk_re = HOLDOUT_RE[1] if fit_re == HOLDOUT_RE[0] else HOLDOUT_RE[0]
        f, c = pool[fit_re], hard[chk_re]
        ed, h = S.fit_width_table(f["r"], f["sh"], f["sc"], nbuck=a.nbuck)
        bn = S.apply_width_table(c["sh"], ed, h)
        bo = S.apply_width_table(c["sh"], edges0, htab0)
        Fn = [F_of(c["r"], bn * l, c["sc"]) for l in lams]
        Fo = [F_of(c["r"], bo * l, c["sc"]) for l in lams]
        i, j = int(np.argmax(Fn)), int(np.argmax(Fo))
        Fn1 = Fn[int(np.where(lams == 1.0)[0][0])]
        Fo65 = F_of(c["r"], bo * 0.65, c["sc"])
        Fo100 = Fo[int(np.where(lams == 1.0)[0][0])]
        Fe = F_of(c["r"], np.abs(c["r"]) + 1e-12, c["sc"])
        print(f"\n[fit Re{fit_re} -> hard Re{chk_re}]  n={np.count_nonzero(c['sc'])}")
        print(f"  fitted table    lam=1.00 F {Fn1:.5f} | best lam {lams[i]:.2f} "
              f"F {Fn[i]:.5f}")
        print(f"  reference table lam=1.00 F {Fo100:.5f} | lam=0.65 F {Fo65:.5f} "
              f"| best lam {lams[j]:.2f} F {Fo[j]:.5f}")
        print(f"  fitted@1.00 - reference@0.65 = {Fn1 - Fo65:+.5f}   "
              f"(|r| oracle {Fe:.5f})")
        # key names are part of the results.json format read downstream
        rows.append(dict(fit=fit_re, chk=chk_re, F_new1=Fn1, F_new=Fn[i],
                         lam_new=float(lams[i]), F_ship100=Fo100,
                         F_ship65=Fo65, F_ship=Fo[j], lam_ship=float(lams[j]),
                         d_new_vs_ship65=Fn1 - Fo65, F_oracle_r=Fe,
                         n=int(np.count_nonzero(c["sc"]))))

    # Deployed table: fitted on both held-out Re together.
    R = np.concatenate([pool[re_]["r"] for re_ in HOLDOUT_RE], 0)
    SH = np.concatenate([pool[re_]["sh"] for re_ in HOLDOUT_RE], 0)
    SC = np.concatenate([pool[re_]["sc"] for re_ in HOLDOUT_RE], 0)
    ed, h = S.fit_width_table(R, SH, SC, nbuck=a.nbuck)
    q = np.percentile(SH[SC], [10, 30, 50, 70, 90, 99]).astype(np.float32)
    print("\nDEPLOYMENT table  reference/fitted h(shat) at shat pct 10/30/50/70/90/99: "
          + " ".join(f"{x/y:.2f}" for x, y in
                     zip(S.apply_width_table(q, edges0, htab0),
                         S.apply_width_table(q, ed, h))))
    d_mean = float(np.mean([r["d_new_vs_ship65"] for r in rows]))
    out = dict(rows=rows, frac=frac, nbuck=a.nbuck,
               newtab_d_vs_ship65=d_mean,
               lam=1.0, kappa=kappa, source=a.runs, emulator=False,
               raw=dict(edges=[float(v) for v in ed], h=[float(v) for v in h]))
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nfitted table vs reference@lam0.65, mean over both fold directions: "
          f"{d_mean:+.5f} F")
    print("NEWTAB DONE")


if __name__ == "__main__":
    main()
