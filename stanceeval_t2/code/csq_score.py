"""
csq_score.py — LOGIN-node LLM signal for the "status-quo criticism" (CSQ) aim pattern.

Targets the ONE persistent residual: the blend over-fires AGAINST on tweets that complain
about the OLD system / status-quo / slow rollout while actually FAVORING the target
(frustration aimed at the status quo, not the target). This scores, per tweet, a narrow
aim question — is the negativity aimed at the OLD situation (author supports the target),
NOT at the target itself? — as a SOFT [0,1] signal to MODULATE the ensemble's Against mass
(posterior-modulating op-point lever; distinct from the ruled-out posterior-replacing aim-CoT).

Output npz aligned to input rows: csq[N] in [0,1], idx, target (+ y_true if labeled).
Reuses llm_predict's proven login HTTP path.
"""
import os, sys, json, argparse, time, re
import numpy as np, pandas as pd
REPO = os.environ.get("LLM_REGISTRY_ROOT", ".")  # dir holding llm_provider_registry.py + .env
sys.path.insert(0, REPO)
from aakit import llm_provider_registry as _reg
from pathlib import Path
_reg.load_dotenv(Path(REPO) / ".env")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S
from llm_predict import call_api, _provider_for

SYS = ("You are an expert Arabic-tweet analyst. Some tweets express NEGATIVITY or complaints that are "
       "NOT aimed at the given TARGET but at the OLD SYSTEM / current situation / slow implementation / "
       "bureaucracy — and the author actually SUPPORTS the target (wants the change).\n"
       "For each numbered Arabic tweet + its TARGET, answer ONLY whether this specific pattern holds:\n"
       "- YES: the tweet criticizes the old/current situation (or slow rollout) while the author FAVORS the target.\n"
       "- WEAK: partially/ambiguously so.\n"
       "- NO: the negativity is aimed at the target itself, OR there is no such old-vs-new pattern (any other case, "
       "including plain favor, plain against, or neutral).\n"
       "Output EXACTLY one line per tweet '<number>|<YES|WEAK|NO>'. No other text.")
VAL = {"YES": 1.0, "WEAK": 0.5, "NO": 0.0}
RE = re.compile(r'^\s*(\d+)\s*[|\.\):\-]\s*(yes|weak|no)\b', re.I)


def score(rows, cfg, model, batch=40):
    csq = np.zeros(len(rows))
    for b0 in range(0, len(rows), batch):
        chunk = rows[b0:b0 + batch]
        user = "Analyze each tweet's negativity aim toward its TARGET:\n\n" + "\n".join(
            f"{i+1}| [TARGET: {t}] {x}" for i, (t, x) in enumerate(chunk))
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
        content = call_api(cfg, model, msgs, "none", 20 * len(chunk) + 200, temperature=0.0)
        got = {}
        for ln in (content or "").splitlines():
            m = RE.match(ln)
            if m:
                got[int(m.group(1))] = VAL[m.group(2).upper()]
        miss = [k for k in range(1, len(chunk) + 1) if k not in got]
        if len(miss) > max(2, len(chunk) // 5):
            content = call_api(cfg, model, msgs, "none", 30 * len(chunk) + 300, temperature=0.0)
            for ln in (content or "").splitlines():
                m = RE.match(ln)
                if m:
                    got.setdefault(int(m.group(1)), VAL[m.group(2).upper()])
        for j in range(len(chunk)):
            csq[b0 + j] = got.get(j + 1, 0.0)
        print(f"  {b0+len(chunk)}/{len(rows)}", flush=True)
    return csq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--csv", default=S.TRAIN_CSV)
    ap.add_argument("--order", default="loto", choices=["loto", "file"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pname, cfg = _provider_for(args.model)
    print(f"[provider] {pname} wire={cfg['wire_model']}", flush=True)
    df = pd.read_csv(args.csv, keep_default_na=False, dtype=str, encoding="utf-8-sig")
    for c in ("text", "target"):
        df[c] = df[c].astype(str).str.strip()
    if args.order == "loto":
        order_idx = []
        for t in dict.fromkeys(df["target"].tolist()):
            order_idx.extend(df.index[df["target"] == t].tolist())
    else:
        order_idx = df.index.tolist()
    rows = [(df.at[i, "target"], df.at[i, "text"]) for i in order_idx]
    t0 = time.time()
    csq = score(rows, cfg, cfg["wire_model"])
    out = {"csq": csq.astype(np.float32), "idx": np.array(order_idx),
           "target": np.array([df.at[i, "target"] for i in order_idx], dtype=object).astype(str)}
    labeled = "stance" in df.columns and (df["stance"].str.strip() != "").all()
    if labeled:
        out["y_true"] = np.array([S.LABEL2ID[df.at[i, "stance"].strip()] for i in order_idx])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **out)
    frac = {k: round(float((csq == v).mean()), 3) for k, v in [("YES", 1.0), ("WEAK", 0.5), ("NO", 0.0)]}
    print(f"[RESULT] {len(rows)} rows -> {args.out} | csq frac {frac} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
