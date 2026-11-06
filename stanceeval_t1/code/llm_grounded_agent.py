"""
llm_grounded_agent.py — grounded Arabic stance voter over llm_client (agent_inproc models).

Same method/prompts as llm_grounded.py (imports CARD_SYS / sys_prompt / LBL_RE from it),
but calls through the shared repo-root llm_client.get_llm — unlocking agent_inproc models
(claude-sonnet-5, claude-opus-4-8) that have no HTTP endpoint. One CLI spawn per batch call.

RUN WITH A VENV THAT HAS THE CLI-BACKED CLIENT INSTALLED:
  <venv>/bin/python llm_grounded_agent.py ...

Output npz identical schema to llm_grounded.py (proba/pred_id/target/idx[/y_true]) for
drop-in blend_eval.py use. --limit N supports a cheap paired probe before a full run.
"""
import os, sys, json, argparse, time, asyncio
import numpy as np, pandas as pd
REPO = os.environ.get("LLM_REGISTRY_ROOT", ".")  # dir holding llm_provider_registry.py + .env
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aakit import llm_provider_registry as _reg
from pathlib import Path
_reg.load_dotenv(Path(REPO) / ".env")
from aakit.llm_client import get_llm
import stance_lib as S
from grounded_prompts import CARD_SYS, sys_prompt  # ported from t2 (see module docstring)
from llm_predict import LBL_RE


def agen(model, msgs, max_tokens):
    async def go():
        llm = get_llm(model)
        try:
            r = await llm.generate(msgs, temperature=0.0, max_tokens=max_tokens)
        finally:
            await llm.close()
        return getattr(r, "text", None) or getattr(r, "content", None) or str(r)
    for attempt in range(3):
        try:
            return asyncio.run(go())
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  [retry {attempt+1}] {type(e).__name__}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))


def make_card(model, target):
    import re as _re
    msgs = [{"role": "system", "content": CARD_SYS},
            {"role": "user", "content": f"TARGET: {target}"}]
    try:
        txt = agen(model, msgs, 2000)
        m = _re.search(r"\{.*\}", txt, _re.S)
        d = json.loads(m.group(0)) if m else {}
        defn = str(d.get("definition", "")).strip()
        asp = d.get("aspects", "")
        asp = ", ".join(asp) if isinstance(asp, list) else str(asp).strip()
        if defn:
            return {"definition": defn, "aspects": asp}
    except Exception as e:
        print(f"[card] FAILED for {target!r}: {e}", flush=True)
    return {"definition": "", "aspects": ""}


def classify_target(texts, model, sysp, batch):
    proba = np.zeros((len(texts), 3), dtype=float)
    for b0 in range(0, len(texts), batch):
        chunk = texts[b0:b0 + batch]
        user = "Classify each tweet's stance toward the TARGET:\n\n" + "\n".join(
            f"{i+1}| {t}" for i, t in enumerate(chunk))
        msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": user}]
        content = agen(model, msgs, 100 * len(chunk) + 1000)
        got = {}
        for line in content.splitlines():
            mm = LBL_RE.match(line)
            if mm:
                got[int(mm.group(1))] = mm.group(2).capitalize()
        miss = [k for k in range(1, len(chunk) + 1) if k not in got]
        if len(miss) > max(2, len(chunk) // 5):
            content = agen(model, msgs, 100 * len(chunk) + 1000)
            for line in content.splitlines():
                mm = LBL_RE.match(line)
                if mm:
                    got.setdefault(int(mm.group(1)), mm.group(2).capitalize())
        for j in range(len(chunk)):
            proba[b0 + j, S.LABEL2ID.get(got.get(j + 1, "None"), 2)] += 1.0
        print(f"  batch {b0//batch+1}/{(len(texts)+batch-1)//batch} done", flush=True)
    return proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--csv", default=S.DEV_CSV)
    ap.add_argument("--order", default="file", choices=["loto", "file"])
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="cap rows (paired probe); 0=all")
    ap.add_argument("--cards_from", default="", help="reuse cards.json from a prior run's meta (identical cards A/B)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, keep_default_na=False, dtype=str)
    for c in ("text", "target"):
        df[c] = df[c].astype(str).str.strip()
    labeled = "stance" in df.columns and (df["stance"].str.strip() != "").all()
    if args.order == "loto":
        order_idx = []
        for tg in dict.fromkeys(df["target"].tolist()):
            order_idx.extend(df.index[df["target"] == tg].tolist())
    else:
        order_idx = df.index.tolist()
    if args.limit:
        order_idx = order_idx[: args.limit]
    targets = list(dict.fromkeys(df.loc[order_idx, "target"].tolist()))

    cards = {}
    if args.cards_from:
        cards = {k: v for k, v in json.load(open(args.cards_from))["cards"].items()}
        print(f"[cards] reusing {list(cards)} from {args.cards_from}", flush=True)
    t0 = time.time()
    for tg in targets:
        if tg not in cards:
            cards[tg] = make_card(args.model, tg)
        print(f"[card] {tg!r}: def={cards[tg]['definition'][:60]!r}", flush=True)

    proba = np.zeros((len(order_idx), 3), dtype=float)
    pos_of = {gi: k for k, gi in enumerate(order_idx)}
    for tg in targets:
        idxs = [i for i in order_idx if df.at[i, "target"] == tg]
        texts = [df.at[i, "text"] for i in idxs]
        pr = classify_target(texts, args.model, sys_prompt(tg, cards[tg]), args.batch)
        for k, i in enumerate(idxs):
            proba[pos_of[i]] = pr[k]
        print(f"[done] {tg!r}: {len(idxs)} tweets", flush=True)

    pred_ids = proba.argmax(1)
    tgt = np.array([df.at[i, "target"] for i in order_idx], dtype=object)
    out = {"proba": proba, "pred_id": pred_ids, "target": tgt.astype(str), "idx": np.array(order_idx)}
    res = {"model": args.model, "grounded": True, "n": len(order_idx), "limit": args.limit,
           "runtime_sec": round(time.time() - t0, 1)}
    if labeled:
        y = np.array([S.LABEL2ID[df.at[i, "stance"].strip()] for i in order_idx])
        out["y_true"] = y
        res["pooled_Favg2"] = round(S.compute_metrics(y, pred_ids)["Favg2"] * 100, 2)
        res["per_target"] = {tg: round(S.compute_metrics(y[tgt == tg], pred_ids[tgt == tg])["Favg2"] * 100, 2) for tg in targets}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **out)
    json.dump({"cards": cards, "res": res}, open(os.path.splitext(args.out)[0] + "_meta.json", "w"),
              indent=2, ensure_ascii=False)
    print("[RESULT] " + json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
