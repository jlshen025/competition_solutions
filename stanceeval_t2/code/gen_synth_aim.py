"""
gen_synth_aim.py — CONSTRUCTED aim-targeted synthetic Arabic stance tweets (LOGIN node).

Idea-4: the Digital residual is an AIM error — the model fires Against on tweets that
complain about the OLD system / status-quo / slow rollout while actually SUPPORTING the
target (56 Favor->Against + 38 None->Against on Digital-LOTO). Hand-written aim-CoT rules
washed. Here the label is fixed BY CONSTRUCTION (the generation scenario, NOT an LLM
annotation), populating the confusable aim boundary with reliable labels, across many
DIVERSE FAKE Arabic targets (never Covid/Digital/WomenEmp -> LOTO stays clean) so the
aim-convention is learned target-agnostically and transfers to blind unseen targets.

Contrastive by design: for EACH target we emit both
  * complaint-about-old-system + support-target  -> Favor  (the hard negative), and
  * complaint/criticism-of-the-target-itself     -> Against (true against),
so the discriminative signal is WHAT the complaint targets, not "is there a complaint"
(guards the aim-CoT failure mode: fixing false-Against must not free true-Against).

Runs on LOGIN (internet); deepseek-v4-flash via the live registry. MUST launch with
Output CSV: ID,text,target,stance,scenario.
"""
import os, sys, re, csv, time, argparse
REPO = os.environ.get("LLM_REGISTRY_ROOT", ".")  # dir holding llm_provider_registry.py + .env
sys.path.insert(0, REPO)
from aakit import llm_provider_registry as _reg
from pathlib import Path
_reg.load_dotenv(Path(REPO) / ".env")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_predict import call_api  # proven login HTTP path

MODEL = "deepseek-v4-flash"


def provider():
    chain = _reg.chain_for(MODEL)
    stable = [(n, c) for n, c in chain if c.get("tier", "stable") == "stable"]
    name, cfg = (stable or chain)[0]
    wire = cfg.get("model_map", {}).get(MODEL, MODEL)
    return {"base_url": cfg["base_url"], "api_key": cfg["api_key"], "wire": wire}


# 20 diverse Arabic fake targets (tech / policy / social / environment / economy).
# NONE is Covid Vaccine / Digital Transformation / Women empowerment.
TARGETS = [
    "السيارات الكهربائية", "العمل عن بُعد", "التعليم الإلكتروني", "الطاقة المتجددة",
    "القطار فائق السرعة", "المدفوعات الإلكترونية", "الذكاء الاصطناعي", "إعادة التدوير",
    "النقل العام", "السياحة الداخلية", "ريادة الأعمال", "الزواج المبكر",
    "التسوق الإلكتروني", "منع التدخين في الأماكن العامة", "العملات الرقمية", "الوجبات السريعة",
    "الدراسة في الخارج", "تربية الحيوانات الأليفة", "الرياضة النسائية", "القيادة الذاتية للسيارات",
]

SUFFIX = ("نوّع الصياغة والطول واللهجة (خليجي/سعودي أو فصحى) بأسلوب تويتر (يمكن استخدام هاشتاقات ورموز تعبيرية). "
          "كل تغريدة في سطر مستقل، عربي فقط، بدون ترقيم وبدون علامات اقتباس وبدون أي شرح أو مقدمة.")

# (scenario_key, stance_label, K, prompt_template)
SCENARIOS = [
    ("oldsys_pro", "Favor", 6,
     "اكتب {k} تغريدات عربية واقعية: الكاتب يشتكي بوضوح من النظام القديم أو الوضع الحالي أو بطء التطبيق أو "
     "الروتين الورقي أو البيروقراطية أو الطوابير، لكنه في نفس الوقت **يؤيد ويطالب بوضوح** بـ«{tgt}». "
     "اجعل الاستياء موجّهاً نحو القديم/التأخير وليس نحو «{tgt}» — موقف الكاتب تجاه «{tgt}» مؤيد وإيجابي. {sfx}"),
    ("direct_pro", "Favor", 3,
     "اكتب {k} تغريدات عربية واقعية: الكاتب **يؤيد ويشجّع بوضوح** «{tgt}» ويعبّر عن حماسه له وعن فوائده. {sfx}"),
    ("oldsys_anti", "Against", 3,
     "اكتب {k} تغريدات عربية واقعية: الكاتب **يعارض** «{tgt}» ويفضّل الطريقة القديمة أو الوضع الحالي، "
     "ويرى أن «{tgt}» غير ضروري أو مبالغ فيه أو أن القديم كان أفضل. موقف الكاتب تجاه «{tgt}» معارض. {sfx}"),
    ("direct_anti", "Against", 3,
     "اكتب {k} تغريدات عربية واقعية: الكاتب **يعارض أو ينتقد أو يتوجّس أو يخاف** من «{tgt}» بشكل مباشر "
     "(مخاطر، أضرار، عدم ثقة، آثار سلبية). موقف الكاتب تجاه «{tgt}» معارض. {sfx}"),
    ("sarcasm_anti", "Against", 3,
     "اكتب {k} تغريدات عربية واقعية بأسلوب **ساخر وتهكّمي** يسخر من «{tgt}» أو يستهزئ به ويقلّل من شأنه "
     "(السخرية تدل على المعارضة). موقف الكاتب تجاه «{tgt}» معارض. {sfx}"),
    ("neutral_none", "None", 4,
     "اكتب {k} تغريدات عربية واقعية تذكر «{tgt}» بشكل **محايد فقط**: خبر أو معلومة أو إحصائية أو سؤال استفساري، "
     "دون أن يُظهر الكاتب أي تأييد أو معارضة شخصية. الموقف: محايد/بلا موقف. {sfx}"),
    ("report_none", "None", 3,
     "اكتب {k} تغريدات عربية واقعية يذكر فيها الكاتب «{tgt}» لكنه **لا يتبنّى موقفاً شخصياً**: ينقل رأي غيره، "
     "أو يطرح احتمالاً، أو يذكره بشكل عابر ضمن حديث عن موضوع آخر. الموقف: محايد/بلا موقف. {sfx}"),
]

_ARABIC = re.compile(r"[؀-ۿ]")
_NUM = re.compile(r"^\s*[\d\-\.\)•\*٠-٩]+\s*")


def parse(content, k, seen):
    out = []
    for ln in content.splitlines():
        s = _NUM.sub("", ln).strip().strip('"“”').strip()
        s = re.sub(r"\s+", " ", s)
        if len(s) >= 15 and _ARABIC.search(s) and s not in seen:
            seen.add(s); out.append(s)
        if len(out) >= k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "_synth", "synth_aim.csv"))
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--mult", type=float, default=1.0, help="scale K per scenario")
    args = ap.parse_args()
    cfg = provider()
    print(f"[provider] base={cfg['base_url']} wire={cfg['wire']} model={MODEL}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    seen = set()
    n = 0; t0 = time.time()
    counts = {"Favor": 0, "Against": 0, "None": 0}
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "text", "target", "stance", "scenario"])
        for ti, tgt in enumerate(TARGETS):
            for sk, lbl, k0, tmpl in SCENARIOS:
                k = max(1, round(k0 * args.mult))
                prompt = tmpl.format(k=k, tgt=tgt, sfx=SUFFIX)
                try:
                    content = call_api(cfg, cfg["wire"], [{"role": "user", "content": prompt}],
                                       "none", 70 * k + 300, temperature=args.temperature)
                except Exception as e:
                    print(f"  [ERR] {tgt}/{sk}: {e}", flush=True); content = ""
                tweets = parse(content or "", k, seen)
                for i, tw in enumerate(tweets):
                    w.writerow([f"syn_{ti}_{sk}_{i}", tw, tgt, lbl, sk])
                    counts[lbl] += 1; n += 1
                f.flush()
            print(f"  [{ti+1}/{len(TARGETS)}] {tgt}: total={n} {counts}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[RESULT] wrote {n} synthetic tweets -> {args.out}  dist={counts}  in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
