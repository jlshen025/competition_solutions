#!/usr/bin/env python
"""none_detect.py — a DEDICATED None-vs-stance detector for MawqifV2.

WHY THIS EXISTS.  The champion ensemble (sub 869442, Favg2 0.891160) is a strong
F/A system bolted to a crippled None head: its top-14 None rows are only 43%
correct, BELOW the 55%/57% break-even, while 28 gold-None rows sit inside the
stance pools as pure false positives dragging both Favg2 terms.  Favg2 excludes
None as a CLASS but not as a ROW.  Fixing None detection is the only axis with
double-digit-row headroom left (perfect detection => 0.9299).

WHAT IT ASKS.  Not the 3-way label — a single graded judgment:

    "How likely is it that a careful annotator would mark this tweet's stance
     toward the TARGET as NOT RECOVERABLE?"   -> integer 0..100

That is MawqifV2's actual None convention, verified from the corpus columns:
none_reason = "Not clear" (313) / "Not Related" (11).  Gold None means the
annotator could not determine the stance, NOT that the author is neutral.
A graded score (not a binary) is what lets us rank and pick an operating point
instead of accepting whatever count the model happens to emit.

DECORRELATION.  Two prompt framings x N labs.  Framing R asks for
non-recoverability directly; framing E asks the complement (how EXPLICIT the
stance signal is) and is inverted on read, so the two disagree on different rows.
Cross-LAB diversity is this project's deepest measured axis (+0.0065, the single
largest gain of the run).

VALIDATION.  Unlike the F/A residual, this axis HAS a local gauge: train/dev carry
333 gold-None rows over the 3 seen targets.  Run with --csv dev.csv to get
precision@k / lift directly.  The unseen-target transfer risk is far lower here
than for F/A polarity: "is a stance expressed at all" is largely target-agnostic.

Usage
-----
    python none_detect.py --model claude-opus-5 --csv <test|dev>.csv \
                          --variant r --out _eval/nd_opus5_test.npz

Output npz: score[N] in 0..100 (higher = more likely gold None), idx, target,
plus y_true/gold-None diagnostics when the csv is labeled.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

REPO = os.environ.get("LLM_REGISTRY_ROOT", ".")  # dir holding llm_provider_registry.py + .env
sys.path.insert(0, REPO)
from aakit import llm_provider_registry as providers  # noqa: E402
from pathlib import Path  # noqa: E402

providers.load_dotenv(Path(REPO) / ".env")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S  # noqa: E402
from llm_predict import call_any  # reuse the retrying multi-leg caller  # noqa: E402

# ---------------------------------------------------------------- prompts
# Framing R: score NON-RECOVERABILITY directly (the gold convention verbatim).
SYS_R = (
    "You are applying MawqifV2's FIXED annotation scheme for Arabic stance detection.\n"
    "In this scheme a tweet is labelled None when the annotator CANNOT DETERMINE the author's "
    "position toward the TARGET from the text, or the tweet is NOT ABOUT the target. "
    "None does NOT mean 'the author is neutral' or 'balanced' — it means the stance is NOT RECOVERABLE.\n"
    "Annotator behaviour, measured on this corpus:\n"
    "  - ~95% of Favor and ~85% of Against labels rest on an EXPLICIT statement. Implicit stance is rare.\n"
    "  - If the label needs a chain of inference, irony, or guessing what the author 'really' means, "
    "the annotators most often recorded None (Not clear).\n"
    "  - Sarcastic tweets are 2.5x MORE likely to be labelled None. Do NOT invert sarcasm to an "
    "'ironic' stance — that is exactly the inference annotators refused to make.\n"
    "  - Pure news/reporting, bare questions, jokes with no side, replies about something else, "
    "and tweets that only mention the topic in a hashtag are None.\n"
    "For each numbered tweet output how likely a careful annotator of THIS scheme would mark it None.\n"
    "Scale: 0 = an explicit, unmistakable Favor or Against statement; "
    "50 = genuinely borderline; 100 = certainly None (no recoverable stance, or off-topic).\n"
    "Output EXACTLY one line per tweet formatted '<number>|<0-100>'. No other text."
)

# Framing E: the COMPLEMENT — score how explicit the stance signal is. Inverted on
# read (100 - score). Asking the opposite question moves the model's errors around.
SYS_E = (
    "You are an expert Arabic-tweet analyst. For each numbered tweet and its TARGET topic, rate how "
    "EXPLICITLY the author states a position (support or opposition) toward that target.\n"
    "Scale: 100 = the author plainly and directly states support or opposition, in words a reader "
    "cannot misread; 50 = a position is implied but you would have to infer it; "
    "0 = no position toward the target is stated at all — the tweet is news, a bare question, "
    "off-topic, a joke with no side, or about something else entirely.\n"
    "Judge only how CLEARLY a position is expressed. Do NOT judge which side it is on, and do NOT "
    "reward a tweet for being emotional, sarcastic or vivid — heat is not clarity. "
    "A sarcastic tweet whose target of mockery is unclear scores LOW.\n"
    "Output EXACTLY one line per tweet formatted '<number>|<0-100>'. No other text."
)

# Framing P: SENTIMENT, not stance. Mawqif ships a gold sentiment column that no
# component uses at inference. Measured on train+dev over Favor/Against rows:
# P(Favor | Positive) = 0.984 and the direction is stable across ALL three seen targets
# (0.979/0.982/0.989), while stance polarity itself transfers badly to an unseen target.
# So sentiment is a target-agnostic bridge INTO the F/A decision, and the cross-class
# flip only needs ~50% precision to pay (gold-None rows caught in the flip are a wash).
# Deliberately never mentions the target's stance poles -- asking about stance is what
# the six incumbent voters already do, and a correlated opinion adds nothing.
SYS_P = (
    "You are an expert annotator of Arabic tweet SENTIMENT. For each numbered tweet, rate the "
    "author's overall emotional tone on a 0-100 scale.\n"
    "  100 = clearly POSITIVE — approving, enthusiastic, celebratory, grateful, hopeful, proud.\n"
    "   50 = NEUTRAL — factual, reporting, a bare question, or genuinely mixed.\n"
    "    0 = clearly NEGATIVE — angry, mocking, disgusted, fearful, contemptuous, complaining.\n"
    "Judge the author's TONE ONLY. Do NOT judge whether they support or oppose any topic, and "
    "do not try to infer a position — a tweet can be warmly positive while opposing something, "
    "or coldly negative while supporting it. Sarcasm expresses NEGATIVE tone even when the "
    "literal words are praise.\n"
    "Output EXACTLY one line per tweet formatted '<number>|<0-100>'. No other text."
)

VARIANTS = {"r": (SYS_R, False), "e": (SYS_E, True), "p": (SYS_P, False)}  # (prompt, invert?)
SCORE_RE = re.compile(r"^\s*(\d+)\s*[|.):\-]\s*(\d{1,3})\b")


def build_demos(k, seed=0):
    """K anchored demos, half gold-None half explicit-stance, drawn from the SEEN targets.

    Gold None here means the annotator could not recover a stance, so the convention is
    carried by examples far better than by prose — the same reason few-shot k=12 won as
    an ensemble member on the F/A axis. Demos are scored on the same 0-100 scale the
    model is asked to emit, and are anchored at the poles (95 / 5) rather than 100 / 0 so
    the model keeps headroom to rank within each side.
    """
    df = pd.read_csv(S.TRAIN_CSV, keep_default_na=False, dtype=str)
    for c in ("text", "target", "stance"):
        df[c] = df[c].astype(str).str.strip()
    rng = np.random.default_rng(seed)
    lines = ["Calibration — how annotators of this scheme scored these tweets:"]
    half = max(1, k // 2)
    none_pool = df.index[df["stance"] == "None"].to_numpy()
    st_pool = df.index[df["stance"] != "None"].to_numpy()
    picks = [(i, 95) for i in rng.choice(none_pool, min(half, len(none_pool)), replace=False)]
    picks += [(i, 5) for i in rng.choice(st_pool, min(k - half, len(st_pool)), replace=False)]
    rng.shuffle(picks)
    for i, sc in picks:
        lines.append(f"[TARGET: {df.at[i, 'target']}] {df.at[i, 'text']} => {sc}")
    return "\n".join(lines) + "\n\n"


def score_rows(rows, legs, sys_prompt, reasoning, batch, invert, demos=""):
    """rows: [(target, text)]. Returns float[N] in 0..100 (higher = more None-like)."""
    out = np.full(len(rows), np.nan)
    for b0 in range(0, len(rows), batch):
        chunk = rows[b0:b0 + batch]
        user = demos + "Rate each tweet:\n\n" + "\n".join(
            f"{i + 1}| [TARGET: {t}] {x}" for i, (t, x) in enumerate(chunk))
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]
        got = {}
        for attempt in range(2):
            content = call_any(legs, msgs, reasoning, 20 * len(chunk) + 400, 0.0)
            for line in content.splitlines():
                m = SCORE_RE.match(line)
                if m:
                    got.setdefault(int(m.group(1)), min(100, int(m.group(2))))
            if len(got) >= len(chunk) - max(1, len(chunk) // 10):
                break
        for j in range(len(chunk)):
            v = got.get(j + 1)
            if v is not None:
                out[b0 + j] = (100 - v) if invert else v
        print(f"  batch {b0 // batch + 1}/{(len(rows) + batch - 1) // batch}: "
              f"{len(got)}/{len(chunk)} parsed", flush=True)
    # Un-parsed rows default to the median (no information => no ranking signal).
    med = np.nanmedian(out) if np.isfinite(out).any() else 50.0
    n_miss = int(np.isnan(out).sum())
    out[np.isnan(out)] = med
    return out, n_miss


def report(score, y, label=""):
    """Precision@k of the None ranking against gold — the number that decides everything."""
    gold_none = (y == S.LABEL2ID["None"])
    n_gold = int(gold_none.sum())
    if n_gold == 0 or len(y) == 0:
        print(f"\n=== None-ranking quality {label}: no gold None in slice, skipped ===")
        return {"n": len(y), "gold_none": 0, "base_rate": 0.0, "prec_at_gold_n": None}
    base = n_gold / len(y)
    order = np.argsort(-score)
    print(f"\n=== None-ranking quality {label} ===")
    print(f"  n={len(y)}  gold None={n_gold}  base rate={base:.3f}")
    print(f"  {'k':>5} {'prec@k':>8} {'lift':>6} {'recall':>7}   (break-even ~0.56)")
    for k in [10, 20, 30, n_gold, 50, 75, 100]:
        if k > len(y):
            continue
        hit = int(gold_none[order[:k]].sum())
        p = hit / k
        flag = "  <== ABOVE break-even" if p >= 0.56 else ""
        print(f"  {k:5d} {p:8.3f} {p / base:6.2f} {hit / max(n_gold,1):7.3f}{flag}")
    return {"n": len(y), "gold_none": n_gold, "base_rate": round(base, 4),
            "prec_at_gold_n": round(float(gold_none[order[:n_gold]].sum()) / max(n_gold, 1), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--variant", default="r", choices=list(VARIANTS))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--reasoning", default="none")
    ap.add_argument("--fewshot", type=int, default=0, help="K calibration demos (0=zero-shot)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys_prompt, invert = VARIANTS[args.variant]
    chain = providers.chain_for(args.model)
    legs = [(c, c.get("model_map", {}).get(args.model, args.model))
            for _, c in reversed(chain) if c.get("api") == "chat" and c.get("base_url")]
    if not legs:
        raise SystemExit(f"no HTTP chat endpoint for {args.model}; chain={[n for n, _ in chain]}")
    print(f"[provider] {args.model} -> {[c['base_url'] for c, _ in legs]}", flush=True)

    df = pd.read_csv(args.csv, keep_default_na=False, dtype=str)
    tcol = "target" if "target" in df.columns else "Target"
    for c in ("text", tcol):
        df[c] = df[c].astype(str).str.strip()
    rows = [(df.at[i, tcol], df.at[i, "text"]) for i in df.index]
    demos = build_demos(args.fewshot) if args.fewshot > 0 else ""
    print(f"[cfg] model={args.model} variant={args.variant} n={len(rows)} batch={args.batch} "
          f"reasoning={args.reasoning} fewshot={args.fewshot}", flush=True)

    t0 = time.time()
    score, n_miss = score_rows(rows, legs, sys_prompt, args.reasoning, args.batch, invert, demos)
    res = {"model": args.model, "variant": args.variant, "fewshot": args.fewshot,
           "n": len(rows), "n_unparsed": n_miss, "runtime_sec": round(time.time() - t0, 1)}

    save = {"score": score, "idx": df.index.to_numpy(),
            "target": df[tcol].to_numpy().astype(str)}
    if "stance" in df.columns and (df["stance"].str.strip() != "").all():
        y = np.array([S.LABEL2ID[df.at[i, "stance"].strip()] for i in df.index])
        save["y_true"] = y
        if args.variant == "p":
            # Sentiment bridge: the number that decides a flip is P(gold Favor) among the
            # top-scoring rows, measured over F/A rows only (None rows are a wash, ~-1e-4).
            fa = y != S.LABEL2ID["None"]
            sf, yf = score[fa], y[fa]
            order = np.argsort(-sf)
            base_f = float((yf == S.LABEL2ID["Favor"]).mean())
            print(f"\n=== sentiment bridge {args.model} === F/A rows={fa.sum()} "
                  f"base P(Favor)={base_f:.3f}  (break-even ~0.51)")
            for k in [20, 40, 60, 100, 150]:
                if k > len(sf):
                    continue
                p = float((yf[order[:k]] == S.LABEL2ID["Favor"]).mean())
                res[f"pFavor@{k}"] = round(p, 4)
                print(f"    top-{k:<4} P(Favor)={p:.3f}"
                      + ("   <== CLEARS" if p >= 0.51 else ""))
            # And the low end, which is the symmetric Favor->Against flip.
            for k in [20, 40, 60]:
                p = float((yf[order[-k:]] == S.LABEL2ID["Against"]).mean())
                res[f"pAgainst@bottom{k}"] = round(p, 4)
                print(f"    bottom-{k:<3} P(Against)={p:.3f}"
                      + ("   <== CLEARS" if p >= 0.51 else ""))
            if "sentiment" in df.columns:
                gs = df["sentiment"].str.strip().to_numpy()[fa]
                for lab in ("Positive", "Negative", "Neutral"):
                    m = gs == lab
                    if m.sum() >= 20:
                        print(f"    [vs gold sentiment] {lab:<9} mean score={sf[m].mean():6.1f} n={m.sum()}")
        else:
            res.update(report(score, y, f"{args.model}/{args.variant}"))
        if args.variant != "p":
            for t in dict.fromkeys(save["target"].tolist()):
                m = save["target"] == t
                res[f"prec_{t}"] = report(score[m], y[m],
                                          f"{args.model}/{args.variant} @ {t}")["prec_at_gold_n"]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **save)
    json.dump(res, open(os.path.splitext(args.out)[0] + "_metrics.json", "w"), indent=2)
    print("[RESULT] " + json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
