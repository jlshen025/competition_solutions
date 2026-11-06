"""
gen_paraphrase.py — minority-class-targeted PARAPHRASE augmentation for the SEEN
Digital-Against slice (the extreme minority: 142 Against vs 879 Favor for Digital
Transformation → drives the Digital residual, blend Favg2 81.9 vs 90+ elsewhere).

Literature-backed (targeted minority aug + implicit oversampling): paraphrase each
existing Digital-Against tweet into K variants that PRESERVE the against-the-target
stance & meaning but vary surface form. Then a STANCE-CONSISTENCY QUALITY GATE:
re-classify every candidate with the SAME verified zero-shot deepseek classifier
(llm_predict.classify) and KEEP only those judged Against — so stance-drifted or
degenerate paraphrases (the failure mode that washed the prior untargeted aug) are
dropped before they ever touch training.

LOGIN NODE only (needs internet). Output CSV columns: ID,text,target,stance
(the 3 columns S.load_labeled needs; aux labels absent → aug rows train stance only).
"""
import os, sys, json, argparse, time, re
import numpy as np, pandas as pd
REPO = os.environ.get("LLM_REGISTRY_ROOT", ".")  # dir holding llm_provider_registry.py + .env
sys.path.insert(0, REPO)
from aakit import llm_provider_registry as providers
from pathlib import Path
providers.load_dotenv(Path(REPO) / ".env")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stance_lib as S
import llm_predict as L

TARGET = "Digital Transformation"
TARGET_AR = "التحول الرقمي"

PARA_SYS = (
    "أنت خبير في إعادة صياغة التغريدات العربية. المهمة: أعِد صياغة تغريدة تُعارض هدفاً "
    "معيّناً، مع الحفاظ التام على (1) الموقف المُعارِض للهدف و(2) المعنى الأساسي، "
    "لكن بأسلوب وكلمات مختلفة (فصحى أو لهجة خليجية، بطول وأسلوب متنوّع كتغريدات مستخدمين "
    "حقيقيين مختلفين). لا تُحوّل الموقف إلى تأييد أو حياد. عربية فقط، بدون ترجمة."
)


def paraphrase_batch(legs, rows, k):
    """rows: list of tweet strings (all Digital-Against). Returns {row_idx: [variants]}."""
    numbered = "\n".join(f"{i+1}| {t}" for i, t in enumerate(rows))
    user = (
        f"الهدف الذي تعارضه كل تغريدة: «{TARGET_AR}».\n"
        f"لكل تغريدة مرقّمة أدناه، اكتب {k} إعادات صياغة مختلفة تحافظ على المعارضة للهدف.\n"
        f"التنسيق: سطر واحد لكل إعادة صياغة بالشكل «<رقم التغريدة>.<رقم الصياغة>| <النص>». "
        f"بدون أي كلام آخر.\n\nالتغريدات:\n{numbered}"
    )
    msgs = [{"role": "system", "content": PARA_SYS}, {"role": "user", "content": user}]
    content = L.call_any(legs, msgs, "none", 60 * k * len(rows) + 400, temperature=0.9)
    out = {i: [] for i in range(len(rows))}
    rx = re.compile(r"^\s*(\d+)\s*[.\-]\s*(\d+)\s*[|\):\-]\s*(.+)$")
    for line in content.splitlines():
        m = rx.match(line)
        if not m:
            continue
        ri = int(m.group(1)) - 1
        txt = re.sub(r"\s+", " ", m.group(3)).strip()
        if 0 <= ri < len(rows) and len(txt) >= 12 and re.search(r"[؀-ۿ]", txt):
            out[ri].append(txt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--k", type=int, default=3, help="paraphrases requested per source tweet")
    ap.add_argument("--cap", type=int, default=300, help="max kept aug rows (targeted, avoid over-tilt)")
    ap.add_argument("--batch", type=int, default=8, help="source tweets per generation call")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "_llm", "aug_digital_against.csv"))
    args = ap.parse_args()

    chain = providers.chain_for(args.model)
    legs = [(c, c.get("model_map", {}).get(args.model, args.model))
            for _, c in reversed(chain) if c.get("api") == "chat" and c.get("base_url")]
    if not legs:
        raise SystemExit(f"no chat endpoint for {args.model}; chain={[n for n,_ in chain]}")
    print(f"[provider] {args.model} -> {len(legs)} leg(s): {[c['base_url'] for c,_ in legs]}", flush=True)

    df = S.load_labeled(S.TRAIN_CSV, preprocess="light")
    src = df[(df[S.TARGET_COL] == TARGET) & (df["label"] == S.LABEL2ID["Against"])].copy()
    tweets = src["text"].tolist()          # ORIGINAL text (not preproc) so paraphrases stay natural
    print(f"[src] {len(tweets)} Digital-Against source tweets; requesting k={args.k} each", flush=True)

    # ---- phase 1: generate ----
    t0 = time.time()
    cands = []      # (source_idx, paraphrase_text)
    for b0 in range(0, len(tweets), args.batch):
        chunk = tweets[b0:b0 + args.batch]
        try:
            got = paraphrase_batch(legs, chunk, args.k)
        except Exception as e:
            print(f"  [warn] batch {b0} failed: {e}", flush=True)
            continue
        for ri, variants in got.items():
            for v in variants:
                cands.append((b0 + ri, v))
        print(f"  gen {b0+len(chunk)}/{len(tweets)} src -> {len(cands)} candidates", flush=True)
    # dedup exact + drop any that equal a source tweet
    srcset = set(t.strip() for t in tweets)
    seen, uniq = set(), []
    for si, v in cands:
        key = v.strip()
        if key and key not in seen and key not in srcset:
            seen.add(key); uniq.append((si, v))
    print(f"[gen] {len(cands)} raw -> {len(uniq)} unique non-source paraphrases in {time.time()-t0:.0f}s", flush=True)

    # ---- phase 2: stance-consistency QUALITY GATE (keep only Against) ----
    rows_for_clf = [(TARGET, v) for _, v in uniq]
    proba = L.classify(rows_for_clf, legs, "none", batch=40, samples=1, temperature=0.0)
    pred = proba.argmax(1)
    kept = [(si, v) for (si, v), p in zip(uniq, pred) if p == S.LABEL2ID["Against"]]
    n_drop = len(uniq) - len(kept)
    print(f"[gate] kept {len(kept)} Against-consistent / dropped {n_drop} stance-drifted", flush=True)

    # cap (keep a spread across distinct source tweets first)
    if len(kept) > args.cap:
        by_src = {}
        for si, v in kept:
            by_src.setdefault(si, []).append(v)
        ordered, rr = [], True
        while len(ordered) < args.cap and rr:
            rr = False
            for si in list(by_src):
                if by_src[si]:
                    ordered.append((si, by_src[si].pop(0))); rr = True
                    if len(ordered) >= args.cap:
                        break
        kept = ordered
    print(f"[cap] final {len(kept)} aug rows (cap={args.cap})", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = pd.DataFrame({
        "ID": [f"aug_dig_ag_{i}" for i in range(len(kept))],
        "text": [v for _, v in kept],
        "target": [TARGET] * len(kept),
        "stance": ["Against"] * len(kept),
    })
    out.to_csv(args.out, index=False)
    meta = {"src_n": len(tweets), "requested_k": args.k, "raw_cands": len(cands),
            "unique": len(uniq), "kept_against": len([1 for _ in kept]), "dropped_drift": n_drop,
            "cap": args.cap, "out": args.out}
    json.dump(meta, open(os.path.splitext(args.out)[0] + "_meta.json", "w"), indent=2, ensure_ascii=False)
    print("[RESULT] " + json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
