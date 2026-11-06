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
from aakit import llm_provider_registry as providers
from pathlib import Path
providers.load_dotenv(Path(REPO) / ".env")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S

SYS = ("You are an expert annotator for Arabic stance detection. For each numbered Arabic tweet and its "
       "TARGET topic, output the author's stance TOWARD THE TARGET as exactly one of: Favor, Against, None.\n"
       "- Favor: the author supports, endorses, or is glad about the target.\n"
       "- Against: the author opposes, criticizes, distrusts, mocks, or fears the target (or its mandate).\n"
       "- None: no clear stance, off-topic, purely factual/news, or ambiguous.\n"
       "Sarcasm and rhetorical questions usually signal Against. Judge the AUTHOR's own stance.\n"
       "Output EXACTLY one line per tweet formatted '<number>|<Label>' (Label = Favor, Against, or None). No other text.")

# Prompt B: a semantic reframe (want/support vs reject) that DROPS the "sarcasm->Against"
# bias (WD sarcasm cuts both ways) -> decorrelated errors for prompt-diverse ensembling.
SYS_B = ("You are an expert Arabic stance-detection annotator. For each numbered Arabic tweet and its TARGET, "
         "decide the author's own position ON THE TARGET ISSUE and output exactly one of: Favor, Against, None.\n"
         "- Favor: the author WANTS / SUPPORTS / celebrates the target (supports the action or policy, is glad it happened).\n"
         "- Against: the author REJECTS / OPPOSES / condemns the target (opposes the action or policy, warns against it, wants it stopped).\n"
         "- None: the author states no personal position — pure news/fact, a bare question, off-topic, or genuinely unclear.\n"
         "Mockery or sarcasm can support EITHER side — judge what the author actually WANTS, not the tone. "
         "A tweet may discuss the target without taking a side (None).\n"
         "Output EXACTLY one line per tweet formatted '<number>|<Label>'. No other text.")

# Prompt C: anti-None "commit" reframe. Frontier LLMs over-default to None on ambiguous rows,
# but the winning ensemble is None-sparse/Against-committed (C4 07-22: fewer None scored higher).
# Push the model to COMMIT to Favor/Against whenever any leaning is present; reserve None strictly.
SYS_C = ("You are an expert annotator for Arabic stance detection. For each numbered Arabic tweet and its "
         "TARGET topic, output the author's stance TOWARD THE TARGET as exactly one of: Favor, Against, None.\n"
         "- Favor: the author supports, endorses, defends, celebrates, or is glad about the target.\n"
         "- Against: the author opposes, criticizes, distrusts, mocks, fears, or warns against the target.\n"
         "- None: use ONLY when the tweet truly takes no side — pure factual news with no opinion, off-topic, "
         "or a bare question. Most tweets that mention the target DO take a side.\n"
         "COMMIT: if the author reveals ANY leaning — through tone, sarcasm, mockery, rhetorical questions, "
         "word choice, or implicit endorsement/criticism — choose Favor or Against, NOT None. Sarcasm, mockery, "
         "and rhetorical questions usually signal Against. Do not retreat to None just because a tweet is subtle.\n"
         "Output EXACTLY one line per tweet formatted '<number>|<Label>' (Label = Favor, Against, or None). No other text.")

# Prompt G: CONVENTION-GROUNDED — written from MawqifV2's own annotation-scheme columns
# (verified on train.csv): none_reason = "Not clear" 313 / "Not Related" 11, i.e. gold None means
# the ANNOTATOR COULD NOT DETERMINE the stance (or the tweet isn't about the target) — NOT "the
# author is neutral". against_reason A_Explicit 857 / A_Implicit 147 and favor_reason F_Explicit
# 1996 / F_Implicit 110 ⇒ explicit-signal-first: implicit stance is a rare, marked category.
# sarcasm×stance = Against 79 / Favor 49 / None 36 ⇒ "sarcasm ⇒ Against" holds only 48% of the
# time, and sarcastic tweets are 2.5x MORE likely to be labelled Not-clear (22.0% vs 8.9%).
SYS_G = ("You are applying a FIXED annotation scheme for Arabic stance detection toward a TARGET. "
         "Label each tweet exactly as this scheme's annotators would.\n"
         "- Favor: the author supports the target. In ~95% of Favor cases this is stated EXPLICITLY; "
         "implicit support is a rare, marked case.\n"
         "- Against: the author opposes the target. In ~85% of Against cases this is stated EXPLICITLY; "
         "implicit opposition is a rare, marked case.\n"
         "- None: you genuinely CANNOT DETERMINE the author's position toward the target, or the tweet "
         "is NOT ABOUT the target. None does NOT mean 'the author is neutral' or 'takes no side' — it "
         "means the stance is NOT RECOVERABLE from the text.\n"
         "Decision rule: if an explicit statement of support or opposition is present, label it. "
         "If you must rely on a chain of inference, irony, or guessing what the author 'really' means, "
         "the annotators most often recorded that as None (Not clear).\n"
         "Sarcasm is NOT a reliable Against signal (it splits roughly evenly between Against and Favor) "
         "and sarcastic tweets are 2.5x MORE likely to be labelled None. Do NOT invert a sarcastic tweet "
         "to its 'ironic' reading; judge what is actually stated, else use None.\n"
         "Output EXACTLY one line per tweet formatted '<number>|<Label>' (Favor, Against, or None). No other text.")

LBL_RE = re.compile(r'^\s*(\d+)\s*[|\.\):\-]\s*(favor|against|none)\b', re.I)


def call_api(cfg, model, messages, reasoning, max_tokens, temperature=0.0):
    body = {"model": model, "messages": messages}
    is_ds = "deepseek" in model.lower()
    # `thinking` is a DeepSeek-NATIVE-endpoint param; the NVIDIA relay rejects it (400).
    is_native_ds = is_ds and "deepseek" in (cfg.get("base_url") or "").lower()
    if reasoning and reasoning != "none":
        if is_native_ds:
            body["thinking"] = {"type": "enabled"}   # DeepSeek reasons via this toggle
            body["max_tokens"] = max(max_tokens, 8000)
        else:
            body["reasoning_effort"] = "max" if reasoning in ("xhigh", "max") else "high"
            body["max_completion_tokens"] = max(max_tokens, 8000)
    else:
        body["temperature"] = temperature
        body["max_tokens"] = max_tokens
        if is_native_ds:
            body["thinking"] = {"type": "disabled"}  # DeepSeek reasons by default; disable for fast bulk
    ep = cfg["base_url"].rstrip("/") + "/chat/completions"
    h = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
    for attempt in range(6):
        try:
            r = requests.post(ep, headers=h, json=body, timeout=240)
            r.raise_for_status()
            return (r.json()["choices"][0]["message"].get("content") or "")
        except Exception as e:
            if attempt == 5:
                raise
            is429 = "429" in str(e) or "Too Many" in str(e)
            time.sleep(40 if is429 else 2 * (attempt + 1))  # per-minute TPM windows need ~40s


def call_any(legs, messages, reasoning, max_tokens, temperature=0.0):
    """Try each (cfg, wire) leg in order until one returns non-empty content."""
    last = None
    for cfg, model in legs:
        try:
            c = call_api(cfg, model, messages, reasoning, max_tokens, temperature)
            if c and c.strip():
                return c
        except Exception as e:
            last = e
    if last:
        raise last
    return ""


def build_demos(k, seed=0):
    """K balanced labeled train examples (seen targets) as few-shot demos."""
    df = pd.read_csv(S.TRAIN_CSV, keep_default_na=False, dtype=str)
    for c in ("text", "target", "stance"):
        df[c] = df[c].astype(str).str.strip()
    rng = np.random.default_rng(seed)
    per = max(1, k // 3)
    lines = ["Study these labeled examples, then label the new tweets by the same standard:"]
    for st in ["Favor", "Against", "None"]:
        sub = df[df["stance"] == st]
        for i in rng.choice(sub.index.to_numpy(), min(per, len(sub)), replace=False):
            lines.append(f"[TARGET: {df.at[i,'target']}] {df.at[i,'text']} => {st}")
    return "\n".join(lines) + "\n\n"


def classify(rows, legs, reasoning, batch, samples=1, temperature=0.0, demos=""):
    """rows: list of (target, text). Returns SOFT proba [N,3] = vote fractions over `samples`."""
    votes = np.zeros((len(rows), 3), dtype=float)
    for s in range(samples):
        temp = temperature if samples > 1 else 0.0
        for b0 in range(0, len(rows), batch):
            chunk = rows[b0:b0 + batch]
            user = demos + "Classify each tweet's stance toward its TARGET:\n\n" + "\n".join(
                f"{i+1}| [TARGET: {tgt}] {txt}" for i, (tgt, txt) in enumerate(chunk))
            msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
            content = call_any(legs, msgs, reasoning, 30 * len(chunk) + 200, temp)
            got = {}
            for line in content.splitlines():
                m = LBL_RE.match(line)
                if m:
                    got[int(m.group(1))] = m.group(2).capitalize()
            miss = [k for k in range(1, len(chunk) + 1) if k not in got]
            if len(miss) > max(2, len(chunk) // 5):  # too many missing -> one retry
                content = call_any(legs, msgs, reasoning, 40 * len(chunk) + 300, temp)
                for line in content.splitlines():
                    m = LBL_RE.match(line)
                    if m:
                        got.setdefault(int(m.group(1)), m.group(2).capitalize())
            for j in range(len(chunk)):
                votes[b0 + j, S.LABEL2ID.get(got.get(j + 1, "None"), 2)] += 1.0
            if os.environ.get("AA_BATCH_SLEEP"):
                time.sleep(float(os.environ["AA_BATCH_SLEEP"]))  # dodge relay RPM (e.g. groq)
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
    ap.add_argument("--prompt_variant", default="a", choices=["a", "b", "c", "g"])
    ap.add_argument("--fewshot", type=int, default=0, help="K few-shot demos (0=zero-shot)")
    ap.add_argument("--target_ctx", default="", help="knowledge-injection: background anchoring the FAVOR/AGAINST poles for the (unseen) target")
    args = ap.parse_args()
    if args.prompt_variant == "b":
        globals()["SYS"] = SYS_B  # classify() reads the module-global SYS
    elif args.prompt_variant == "c":
        globals()["SYS"] = SYS_C  # anti-None "commit" reframe
    elif args.prompt_variant == "g":
        globals()["SYS"] = SYS_G  # convention-grounded (MawqifV2 annotation scheme)
    if args.target_ctx:  # knowledge injection (Zhang 2024): ground the target's favor/against axis
        globals()["SYS"] = globals()["SYS"] + (
            "\n\nTARGET CONTEXT (authoritative — use ONLY to disambiguate which direction is FAVOR vs "
            "AGAINST for this target; still judge each author's OWN stance from their tweet):\n" + args.target_ctx)
    demos = build_demos(args.fewshot) if args.fewshot > 0 else ""
    chain = providers.chain_for(args.model)  # [(name,cfg)] unstable->stable
    legs = [(c, c.get("model_map", {}).get(args.model, args.model))
            for _, c in reversed(chain) if c.get("api") == "chat" and c.get("base_url")]
    only = os.environ.get("AA_ONLY_URL")
    if only:
        legs = [(c, m) for c, m in legs if only in (c.get("base_url") or "")]
    if not legs:
        raise SystemExit(f"no HTTP chat endpoint for {args.model}; chain={[n for n,_ in chain]}")
    print(f"[provider] {args.model} -> {len(legs)} chat leg(s): {[c['base_url'] for c,_ in legs]}", flush=True)

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
    proba = classify(rows, legs, args.reasoning, args.batch, args.samples, args.temperature, demos=demos)
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
