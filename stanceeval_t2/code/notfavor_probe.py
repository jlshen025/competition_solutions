#!/usr/bin/env python
"""displace_probe.py — TARGET-DISPLACEMENT probe over a pool of tweets.

Mechanism being tested (named from an eyeball read of the champion's Ecars Favor pool):
some tweets carry a clear evaluative stance, but it is aimed at a DIFFERENT object
(the government, the country, a company, a person) while the TARGET is used only as a
benchmark / example / backdrop. A generic stance prompt sees target-positive wording and
votes Favor; the corpus annotator marks None because no stance is directed at the target.

This is deliberately NOT the "pure spec/news sharing = None" hypothesis, which this project
measured DEAD on 2026-07-30 (0/6 at conf 82). Spec-sharing and growth-reporting are
convention-Favor here; displacement is a different question and is asked as such.

BUILT-IN FREE VALIDATION: run over the whole pool including the rows the champion lineage
already moved None->Favor (decode-measured majority-correct, i.e. gold-Favor). A good probe
must stay QUIET on those. Firing rate there is an upper bound on its false-positive rate,
paid for with zero submission slots.

Usage:
  <repo>/.venv/bin/python displace_probe.py \
      --model claude-opus-5 --target Ecars --pool favor --out _llm/displace_ec.json
"""
import os, sys, json, argparse, time, asyncio, re
import numpy as np, pandas as pd

REPO = os.environ.get("LLM_REGISTRY_ROOT", ".")  # dir holding llm_provider_registry.py + .env
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aakit import llm_provider_registry as _reg
from pathlib import Path
_reg.load_dotenv(Path(REPO) / ".env")
from aakit.llm_client import get_llm

SYS = """You are replicating the labelling decisions of ONE specific Arabic annotation project.
You are NOT giving your own opinion. Each tweet below has been machine-translated to English.

TARGET: {target_desc}

A strong classifier already labelled every tweet below FAVOR. Your ONLY job is to find the ones
where THIS project's annotators would NOT have written Favor — they would have written either
Against or None. **You do NOT need to say which.** Answer only Favor vs NOT-FAVOR.

MEASURED CONVENTIONS of this annotation project (from 4121 labelled rows — follow these, not
your own intuition):
  - The annotators resolve ambiguity TOWARD stance and toward Favor. Even tweets with purely
    NEUTRAL tone were labelled Favor 50.7% of the time and None only 22.9%.
  - Sharing product news, specifications, sales figures, market growth or launch reports about
    the target = FAVOR. National-pride and national-aspiration framing = FAVOR (odds-ratio 4.8).
    "The world is advanced and we cannot even build these" = FAVOR.
  - Praise is taken at FACE VALUE. Do NOT read irony or sarcasm into praise — sarcasm carries
    almost no information here (measured odds-ratio 1.22) and reading it in is a known error.
  - Favor is 95% EXPLICIT. Their None class is 97% "the annotators could not tell", not
    "off-topic".
So: only answer NOT-FAVOR when the tweet carries a genuine criticism, complaint, distrust,
rejection or mockery OF THE TARGET ITSELF, or when it is so devoid of any evaluative content
that two annotators would have failed to agree it was Favor. Be strict — most of these ARE Favor.

Output EXACTLY one line per tweet: '<number>|<FAVOR or NOT-FAVOR>|<confidence 0-100>|<=5-word reason'
No other text."""

DESC = {
    "Ecars": "السيارات الكهربائية (electric cars) — the vehicles themselves, as a technology and a product.",
    "Trimester": "الفصل الدراسي الثالث / نظام الثلاثة فصول (the three-term school year system in Saudi Arabia).",
}


def agen(model, msgs, max_tokens=3000):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--target", default="Ecars")
    ap.add_argument("--pool", default="favor", choices=["favor", "against", "none"])
    ap.add_argument("--base", default="staging/e_ch869013.txt")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t = pd.read_csv("test_norm.csv", keep_default_na=False, encoding="utf-8-sig")
    _en = pd.read_csv("_llm/test_en.csv", keep_default_na=False, encoding="utf-8-sig")
    _c = [c for c in _en.columns if c.lower() in ("en","text_en","english","text")][0]
    t = t.assign(text=_en[_c].values)
    ch = [l.strip() for l in open(a.base, encoding="utf-8") if l.strip()]
    want = {"favor": "Favor", "against": "Against", "none": "None"}[a.pool]
    idx = [i for i in range(644) if t.at[i, "target"] == a.target and ch[i] == want]
    print(f"pool: {a.target} champion-{want} -> {len(idx)} rows", flush=True)

    sysp = SYS.format(target_desc=DESC[a.target])
    res = {}
    for b0 in range(0, len(idx), a.batch):
        chunk = idx[b0:b0 + a.batch]
        user = "Audit each tweet:\n\n" + "\n".join(
            f"{k+1}| {t.at[i,'text']}" for k, i in enumerate(chunk))
        txt = agen(a.model, [{"role": "system", "content": sysp},
                             {"role": "user", "content": user}])
        got = 0
        for line in txt.splitlines():
            m = re.match(r"\s*(\d+)\s*\|\s*(NOT-FAVOR|FAVOR)\s*\|\s*(\d+)\s*\|?\s*(.*)", line.strip(), re.I)
            if not m:
                continue
            k = int(m.group(1)) - 1
            if not (0 <= k < len(chunk)):
                continue
            res[int(chunk[k])] = {"verdict": m.group(2).upper(), "conf": int(m.group(3)),
                                  "aimed_at": m.group(4).strip()[:40]}
            got += 1
        print(f"  batch {b0//a.batch+1}/{(len(idx)-1)//a.batch+1}: parsed {got}/{len(chunk)}", flush=True)

    json.dump({"model": a.model, "target": a.target, "pool": a.pool, "n": len(idx),
               "res": {str(k): v for k, v in res.items()}}, open(a.out, "w"),
              indent=1, ensure_ascii=False)
    d = sum(1 for v in res.values() if v["verdict"] == "NOT-FAVOR")
    print(f"[RESULT] parsed {len(res)}/{len(idx)}  NOTFAVOR={d}  -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
