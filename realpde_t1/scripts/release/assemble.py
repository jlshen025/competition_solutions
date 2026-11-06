"""Assemble the Track-1 submission archive from trained run dirs + fitted calibration.

This is the last step of the release pipeline (`scripts/release/run_all.sh`).
It writes `config.json` directly -- every setting is an explicit argument whose
default is the shipped value -- and packages, at the archive root:

    submission.py      (scripts/sub/submission.py)
    rpde_model.py      (scripts/rpde/model.py)
    model<i>.pth       (each run dir's plain tensor state_dict, copied byte-for-byte)
    config.json

Inputs:
  --runs    comma-separated run dirs of the ship nets (all-data training), in order
  --table   newtab.py results.json  -> the per-element width table ("raw" block)
  --vfit    vfit.py   results.json  -> the variance-head ridge coefficients
  --base    write a calibration-free base archive (identity table, no variance
            head); newtab.py / vfit.py read beta/kappa from it.
  --expect  compare against a reference archive: config keys, checkpoint bytes
            and code files.

    python scripts/release/assemble.py --runs a,b,c,d --table T.json --vfit V.json \
        --out release_work/submission.zip [--expect reference.zip]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

# Fixed global normalisation: real-data (u, v) mean/std, the same constants as
# train.py and eval_ens.py (not a per-window statistic).
MEAN_R = [0.15496085584163666, -0.0005139928543940187]
STD_R = [0.09680565446615219, 0.015960684046149254]


def spec_of(run):
    """Per-net entry of config["nets"]. Besides width/depth/delta/ar_m the
    archive format carries architecture fields that are fixed in this release
    (no spectral layers, single resolution, ...); they are written with their
    fixed values so the config schema stays the same."""
    rj = json.load(open(os.path.join(run, "results.json")))["args"]
    for k in ("cascade", "spectral", "multiscale", "tshare", "n_glob", "n_aux",
              "aux_fine", "ar_detach"):
        assert not rj.get(k), f"{run}: unsupported option {k}={rj.get(k)!r}"
    return dict(width=rj["width"], depth=rj["depth"], m1=12, m2=24,
                spectral=False, multiscale=False,
                delta=bool(rj.get("delta") or 0),
                tshare=0, n_glob=0, n_aux=0,
                ar_m=int(rj.get("ar_m") or 0), ar_detach=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", default="")
    ap.add_argument("--vfit", default="")
    ap.add_argument("--base", action="store_true",
                    help="calibration-free base archive for newtab.py/vfit.py to read")
    ap.add_argument("--beta", type=float, default=1.05,
                    help="temporal-anomaly amplification about the window mean")
    ap.add_argument("--kappa", type=float, default=0.75,
                    help="weight of member disagreement in the predictive variance")
    ap.add_argument("--lam", type=float, default=0.95,
                    help="scale on the width table's half-width")
    ap.add_argument("--vh-dose", type=float, default=0.3)
    ap.add_argument("--vh-gate", type=float, default=0.7,
                    help="variance head acts only above this per-window "
                         "quantile of predicted anomaly variance")
    ap.add_argument("--vh-dose-up", type=float, default=0.7,
                    help="dose where the head amplifies (g >= 1)")
    ap.add_argument("--vh-dose-dn", type=float, default=0.2,
                    help="dose where the head shrinks (g < 1)")
    ap.add_argument("--out-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--batch", type=int, default=16,
                    help="largest CUDA-graph batch size")
    ap.add_argument("--expect", default="", help="zip whose config/checkpoints must match exactly")
    # Inference layout: 2 = the members run as one grouped-convolution trunk
    # with each block's 1x1 conv folded into its 3x3 conv; 0 = members run one
    # after another. --fwd-dtype sets the trunk precision ('' = float32).
    ap.add_argument("--fused", type=int, default=0, choices=[0, 2],
                    help="0 = key omitted (members run serially)")
    ap.add_argument("--fwd-dtype", default="", help="'' = key omitted (float32 trunk)")
    a = ap.parse_args()

    runs = [r.strip() for r in a.runs.split(",") if r.strip()]
    specs = [spec_of(r) for r in runs]
    if a.base:
        # identity single-bucket table (h = shat), lam 1, no variance head
        table = dict(edges=[0.0, 1e9], h=[1.0])
    else:
        table = json.load(open(a.table))["raw"]
    # tsharp / alpha / zero_mask / h_floor / mask_width are fixed fields of the
    # archive format; this release always writes them at their neutral values.
    cfg = dict(nets=specs, checkpoints=[f"model{i}.pth" for i in range(len(runs))],
               mean=MEAN_R, std=STD_R, H=32, W=64, batch=a.batch,
               beta=a.beta, tsharp=0.0, alpha=None, zero_mask=False,
               lam=1.0 if a.base else a.lam, kappa=a.kappa, h_floor=0.0,
               mask_width=False, width_table=dict(edges=table["edges"], h=table["h"]))
    if not a.base:
        vf = json.load(open(a.vfit))["vhead"]
        assert vf["kind"] == "poly2", vf["kind"]
        cfg["vhead"] = dict(w=vf["w"], kind=vf["kind"], dose=a.vh_dose, gate=a.vh_gate,
                            dose_up=a.vh_dose_up, dose_dn=a.vh_dose_dn)
        cfg["out_dtype"] = a.out_dtype
        if a.fused:
            cfg["fused"] = a.fused
        if a.fwd_dtype.strip():
            cfg["fwd_dtype"] = a.fwd_dtype.strip()

    stage = tempfile.mkdtemp(prefix="rpde_assemble_")
    shutil.copy(os.path.join(SCRIPTS, "sub", "submission.py"), os.path.join(stage, "submission.py"))
    shutil.copy(os.path.join(SCRIPTS, "rpde", "model.py"), os.path.join(stage, "rpde_model.py"))
    for r, name in zip(runs, cfg["checkpoints"]):
        shutil.copy(os.path.join(r, "model.pth"), os.path.join(stage, name))
    json.dump(cfg, open(os.path.join(stage, "config.json"), "w"), indent=1)
    names = sorted(os.listdir(stage))
    extracted = sum(os.path.getsize(os.path.join(stage, n)) for n in names)
    assert extracted < 256e6, f"extracted {extracted/1e6:.1f} MB exceeds the 256 MB cap"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(os.path.join(stage, n), n)            # archive root
    shutil.rmtree(stage)
    sha = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(f"ASSEMBLED {a.out} sha {sha[:8]} files {names} extracted {extracted/1e6:.1f} MB")
    out = dict(zip=a.out, sha256=sha, files=names, extracted_mb=round(extracted / 1e6, 2),
               runs=runs, table=a.table, vfit=a.vfit, base=a.base)

    if a.expect:
        ze, zn = zipfile.ZipFile(a.expect), zipfile.ZipFile(a.out)
        ce = json.loads(ze.read("config.json"))
        cn = json.loads(zn.read("config.json"))
        diff = sorted(k for k in set(ce) | set(cn) if ce.get(k) != cn.get(k))
        ck = {n: hashlib.sha256(ze.read(n)).hexdigest() == hashlib.sha256(zn.read(n)).hexdigest()
              for n in cfg["checkpoints"]}
        code = {n: ze.read(n) == zn.read(n) for n in ("submission.py", "rpde_model.py")}
        out["expect"] = dict(ref=a.expect, config_keys_differing=diff,
                             checkpoints_identical=ck, code_identical=code,
                             config_identical=not diff)
        print("EXPECT " + json.dumps(out["expect"]))
    od = os.environ.get("EXP_OUTPUT_DIR")
    if od:
        json.dump(out, open(os.path.join(od, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
