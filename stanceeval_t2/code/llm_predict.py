"""
llm_predict.py — Bulk zero-shot Arabic stance classification via a frontier LLM.

Runs on the LOGIN NODE (compute nodes have no internet). Calls the DeepSeek chat
API directly with `requests` (openai SDK not installed), batching ~N tweets per
request. Saves predictions aligned to the input rows (idx, target, y_true, pred_id,
one-hot proba) for offline ensembling with encoder OOF. Prints per-target + pooled
Favg2 if labels are present.

Config from repo .env via providers.load_dotenv + provider_for(model).
"""
import os, sys, json, argparse, time, re
import numpy as np, pandas as pd, requests
REPO = os.environ.get("LLM_REGISTRY_ROOT", ".")  # dir holding llm_provider_registry.py + .env
sys.path.insert(0, REPO)
from aakit import llm_provider_registry as _reg
from pathlib import Path
_reg.load_dotenv(Path(REPO) / ".env")


def _provider_for(model, provider=""):
    """Resolve a model id -> (name, cfg{base_url,api_key,wire_model}).
    Prefer the STABLE leg (official API) for bulk-inference reliability, unless
    `provider` names a specific leg to force (e.g. route gpt-5.6-luna/terra to
    the working sub2apigpt relay instead of the real openai endpoint)."""
    chain = _reg.chain_for(model)
    if provider:
        legs = [(n, c) for n, c in chain if n == provider]
        if not legs:
            raise SystemExit(f"[provider] {provider!r} not serving {model!r}; chain={[n for n, _ in chain]}")
        name, cfg = legs[0]
    else:
        stable = [(n, c) for n, c in chain if c.get("tier", "stable") == "stable"]
        name, cfg = (stable or chain)[0]
    wire = cfg.get("model_map", {}).get(model, model)
    return name, {"base_url": cfg["base_url"], "api_key": cfg["api_key"], "wire_model": wire}


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S

SYS = ("You are an expert annotator for Arabic stance detection. For each numbered Arabic tweet and its "
       "TARGET topic, output the author's stance TOWARD THE TARGET as exactly one of: Favor, Against, None.\n"
       "- Favor: the author supports, endorses, or is glad about the target.\n"
       "- Against: the author opposes, criticizes, distrusts, mocks, or fears the target (or its mandate).\n"
       "- None: no clear stance, off-topic, purely factual/news, or ambiguous.\n"
       "Sarcasm and rhetorical questions usually signal Against. Judge the AUTHOR's own stance.\n"
       "Output EXACTLY one line per tweet formatted '<number>|<Label>' (Label = Favor, Against, or None). No other text.")

LBL_RE = re.compile(r'^\s*(\d+)\s*[|\.\):\-]\s*(favor|against|none)\b', re.I)


def call_api(cfg, model, messages, reasoning, max_tokens, temperature=0.0):
    body = {"model": model, "messages": messages}
    if reasoning and reasoning != "none":
        body["reasoning_effort"] = "max" if reasoning in ("xhigh", "max") else "high"
        body["max_completion_tokens"] = max_tokens
    else:
        body["temperature"] = temperature
        body["max_tokens"] = max_tokens
        if "deepseek" in model.lower():
            body["thinking"] = {"type": "disabled"}  # DeepSeek reasons by default; disable for fast bulk
    ep = cfg["base_url"].rstrip("/") + "/chat/completions"
    h = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            r = requests.post(ep, headers=h, json=body, timeout=240)
            r.raise_for_status()
            _m = r.json()["choices"][0]["message"]
            return _m.get("content") or _m.get("reasoning_content") or ""
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def classify(rows, cfg, model, reasoning, batch, samples=1, temperature=0.0):
    """rows: list of (target, text). Returns SOFT proba [N,3] = vote fractions over `samples`."""
    votes = np.zeros((len(rows), 3), dtype=float)
    for s in range(samples):
        temp = temperature if samples > 1 else 0.0
        for b0 in range(0, len(rows), batch):
            chunk = rows[b0:b0 + batch]
            user = "Classify each tweet's stance toward its TARGET:\n\n" + "\n".join(
                f"{i+1}| [TARGET: {tgt}] {txt}" for i, (tgt, txt) in enumerate(chunk))
            msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
            content = call_api(cfg, model, msgs, reasoning, 30 * len(chunk) + 200, temp)
            got = {}
            for line in content.splitlines():
                m = LBL_RE.match(line)
                if m:
                    got[int(m.group(1))] = m.group(2).capitalize()
            miss = [k for k in range(1, len(chunk) + 1) if k not in got]
            if len(miss) > max(2, len(chunk) // 5):  # too many missing -> one retry
                content = call_api(cfg, model, msgs, reasoning, 40 * len(chunk) + 300, temp)
                for line in content.splitlines():
                    m = LBL_RE.match(line)
                    if m:
                        got.setdefault(int(m.group(1)), m.group(2).capitalize())
            for j in range(len(chunk)):
                votes[b0 + j, S.LABEL2ID.get(got.get(j + 1, "None"), 2)] += 1.0
        print(f"  sample {s+1}/{samples} done", flush=True)
    return votes / samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--csv", default=S.TRAIN_CSV)
    ap.add_argument("--per_target", type=int, default=0, help="subsample N/target (0=all)")
    ap.add_argument("--order", default="loto", choices=["loto", "file"])
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--reasoning", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=1, help=">1 enables self-consistency (soft proba)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pname, cfg = _provider_for(args.model)
    wire_model = cfg["wire_model"]
    print(f"[provider] {pname} base={cfg['base_url']} wire={wire_model}", flush=True)

    df = pd.read_csv(args.csv, keep_default_na=False, dtype=str)
    for c in ("text", "target"):
        df[c] = df[c].astype(str).str.strip()
    labeled = "stance" in df.columns and (df["stance"].str.strip() != "").all()

    # build ordered row list
    order_idx = []
    if args.order == "loto":
        targets = list(dict.fromkeys(df["target"].tolist()))
        rng = np.random.default_rng(args.seed)
        for t in targets:
            pos = df.index[df["target"] == t].to_numpy()
            if args.per_target and len(pos) > args.per_target:
                pos = np.sort(rng.choice(pos, args.per_target, replace=False))
            order_idx.extend(pos.tolist())
    else:
        order_idx = df.index.tolist()
    rows = [(df.at[i, "target"], df.at[i, "text"]) for i in order_idx]
    print(f"[cfg] model={args.model} n={len(rows)} order={args.order} batch={args.batch} reasoning={args.reasoning} samples={args.samples} temp={args.temperature}", flush=True)

    t0 = time.time()
    proba = classify(rows, cfg, wire_model, args.reasoning, args.batch, args.samples, args.temperature)
    pred_ids = proba.argmax(1)
    tgt = np.array([df.at[i, "target"] for i in order_idx], dtype=object)
    out = {"proba": proba, "pred_id": pred_ids, "target": tgt.astype(str), "idx": np.array(order_idx)}
    res = {"model": args.model, "n": len(rows), "reasoning": args.reasoning, "runtime_sec": round(time.time() - t0, 1)}
    if labeled:
        y = np.array([S.LABEL2ID[df.at[i, "stance"].strip()] for i in order_idx])
        out["y_true"] = y
        pooled = S.compute_metrics(y, pred_ids)
        res["pooled_Favg2"] = round(pooled["Favg2"] * 100, 2)
        res["pooled_Favg3"] = round(pooled["Favg3"] * 100, 2)
        res["per_target"] = {}
        for t in dict.fromkeys(tgt.tolist()):
            mask = tgt == t
            m = S.compute_metrics(y[mask], pred_ids[mask])
            res["per_target"][t] = {"Favg2": round(m["Favg2"] * 100, 2), "Favg3": round(m["Favg3"] * 100, 2), "n": int(mask.sum())}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **out)
    json.dump(res, open(os.path.splitext(args.out)[0] + "_metrics.json", "w"), indent=2, ensure_ascii=False)
    print("[RESULT] " + json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
