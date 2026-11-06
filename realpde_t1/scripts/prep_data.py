"""Preprocess the RealPDE release into 32x64 float32 memmaps.

Real: u,v at top level, (T,64,128) float64  -> (T,32,64,2) float32
Sim : u,v,p at top level, (T,64,128) float32 -> (T,32,64,3) float32

Subsampling is the official [..., ::2, ::2]. Excludes train_real/7575_0.h5,
which the organizers' data notice says duplicates 6300_0.h5.
Writes <out>/{real,sim}.npy (concatenated along time) + {real,sim}_index.json.

    python scripts/prep_data.py [real|sim|both]      (default both)
"""
import json, os, sys
import h5py
import numpy as np

D = os.environ.get("RPDE_RAW", "data/realpde")
OUT = os.environ.get("RPDE_DATA32", "data32")
SUB = 2
BAD = {("7575", "0")}


def build(split, chans):
    src = os.path.join(D, f"train_{split}")
    files = sorted(f for f in os.listdir(src) if f.endswith(".h5"))
    recs, blocks, off = [], [], 0
    for fn in files:
        re_s, aoa_s = fn[:-3].split("_")
        if split == "real" and (re_s, aoa_s) in BAD:
            print("SKIP", fn, "(organizer data notice)")
            continue
        with h5py.File(os.path.join(src, fn), "r") as f:
            arr = np.stack(
                [np.asarray(f[c][:, ::SUB, ::SUB], dtype=np.float32) for c in chans],
                axis=-1)
        T = arr.shape[0]
        recs.append(dict(file=fn, re=int(re_s), aoa=int(aoa_s), start=off, T=T))
        blocks.append(arr)
        off += T
        print(f"{split} {fn} T={T} shape={arr.shape[1:]}", flush=True)
    full = np.concatenate(blocks, axis=0)
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, f"{split}.npy"), full)
    json.dump(recs, open(os.path.join(OUT, f"{split}_index.json"), "w"), indent=1)
    print(f"[{split}] trajectories={len(recs)} total_frames={off} array={full.shape} "
          f"{full.nbytes/1e9:.2f} GB")
    return full, recs


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "real"):
        r, rec = build("real", ["u", "v"])
        # channel statistics of real (u, v), for reference
        print("real mean", r.reshape(-1, 2).mean(0), "std", r.reshape(-1, 2).std(0))
        z = (r == 0.0)
        print("real zero-fraction per channel", z.reshape(-1, 2).mean(0))
    if which in ("both", "sim"):
        s, _ = build("sim", ["u", "v", "p"])
        print("sim mean", s.reshape(-1, 3).mean(0), "std", s.reshape(-1, 3).std(0))
