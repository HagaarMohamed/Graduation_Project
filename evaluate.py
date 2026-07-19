"""
evaluate.py — Reference-Free Pipeline Evaluation
Usage: python evaluate.py --image "photo.jpg"
"""

import argparse
import json
import os
import sys


def run_evaluation(image_path: str):
    # ── Load result ───────────────────────────────────────────
    result_path = "outputs/translation/translation_result.json"
    if not os.path.exists(result_path):
        print("ERROR: No translation found.")
        print("Run: python main.py translate --image ... --save")
        sys.exit(1)

    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    from stage4_translation.translator import GARDINER_DICT

    print("\n" + "═" * 65)
    print("  📊  HIEROGLYPH PIPELINE EVALUATION REPORT")
    print("═" * 65)

    # ── 1. Sign Coverage ──────────────────────────────────────
    codes    = result["raw_codes"]
    found    = [c for c in codes if c in GARDINER_DICT]
    missing  = [c for c in codes if c not in GARDINER_DICT]
    coverage = len(found) / len(codes) * 100 if codes else 0

    print(f"\n  [1] Sign Coverage")
    print(f"  Total detected   : {len(codes)}")
    print(f"  In dictionary    : {len(found)}  ({coverage:.1f}%)")
    if missing:
        print(f"  Missing signs    : {', '.join(missing)}")

    # ── 2. Word Analysis ──────────────────────────────────────
    word_anal = result.get("word_analysis", [])
    complete  = [w for w in word_anal if w.get("transliteration") and w.get("english")]
    w_cov     = len(complete) / len(word_anal) * 100 if word_anal else 0

    print(f"\n  [2] Word Analysis")
    print(f"  Word groups      : {len(word_anal)}")
    print(f"  Words identified : {len(complete)}  ({w_cov:.1f}%)")
    for w in complete:
        codes_str = "+".join(w.get("codes", []))
        print(f"    [{codes_str}] {w.get('transliteration','')} → {w.get('english','')} / {w.get('arabic','')}")

    # ── 3. Known Pattern Matching ────────────────────────────
    KNOWN_PATTERNS = {
        "Htp di nsw" : "Royal offering formula",
        "anx wDAs"   : "Life, prosperity, health",
        "nb tAwy"    : "Lord of the Two Lands",
        "nTr nfr"    : "The Good God (pharaoh)",
        "ra nb"      : "Ra, lord of...",
        "di anx"     : "Given life",
        "nswt bity"  : "King of Upper and Lower Egypt",
        "zA ra"      : "Son of Ra",
        "nfr"        : "Good/beautiful (common sign)",
        "nb"         : "Lord/all",
    }

    translit = result.get("transliteration", "").lower()
    matches  = []
    for pattern, meaning in KNOWN_PATTERNS.items():
        if pattern.lower() in translit:
            matches.append(f"{pattern} = {meaning}")

    print(f"\n  [3] Known Egyptian Patterns Found")
    if matches:
        for m in matches:
            print(f"    ✅ {m}")
    else:
        print("    ⚠️  No common patterns matched")

    # ── 4. LLM Self-Evaluation ───────────────────────────────
    print(f"\n  [4] LLM Self-Evaluation")
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("  ⚠️  GROQ_API_KEY not set — skipping self-evaluation")
    else:
        client = Groq(api_key=api_key)
        import yaml
        with open("config/config.yaml", "r") as f:
            cfg   = yaml.safe_load(f)
        model = cfg["translation"]["llm_model"]

        eval_prompt = f"""You are an expert Egyptologist.
Evaluate this hieroglyph translation WITHOUT needing a reference.

Signs: {' '.join(codes)}
Transliteration: {result['transliteration']}
English: {result['english_translation']}
Arabic: {result['arabic_translation']}

Score each 1-5 and respond ONLY in JSON:
{{
  "phonetic_accuracy": 1-5,
  "determinative_logic": 1-5,
  "translation_fluency": 1-5,
  "historical_plausibility": 1-5,
  "overall_score": 1-5,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "suggestions": "..."
}}"""

        resp = client.chat.completions.create(
            model    = model,
            messages = [{"role": "user", "content": eval_prompt}],
            max_tokens = 500, temperature = 0.1,
        )
        raw = resp.choices[0].message.content.strip()
        try:
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            scores = json.loads(raw)
            print(f"  Phonetic accuracy      : {'⭐' * scores.get('phonetic_accuracy',0)}")
            print(f"  Determinative logic    : {'⭐' * scores.get('determinative_logic',0)}")
            print(f"  Translation fluency    : {'⭐' * scores.get('translation_fluency',0)}")
            print(f"  Historical plausibility: {'⭐' * scores.get('historical_plausibility',0)}")
            print(f"  Overall score          : {'⭐' * scores.get('overall_score',0)} / ⭐⭐⭐⭐⭐")
            if scores.get("strengths"):
                print(f"  Strengths  : {', '.join(scores['strengths'])}")
            if scores.get("weaknesses"):
                print(f"  Weaknesses : {', '.join(scores['weaknesses'])}")
            if scores.get("suggestions"):
                print(f"  Suggestions: {scores['suggestions']}")
        except Exception as e:
            print(f"  Raw evaluation: {raw}")

    # ── 5. Final Summary ──────────────────────────────────────
    print(f"\n  [5] Final Summary")
    print(f"  LLM Confidence   : {result.get('confidence','—').upper()}")
    print(f"  English          : {result['english_translation']}")
    print(f"  Arabic           : {result['arabic_translation']}")
    if result.get("caveats"):
        print(f"  Caveats          : {result['caveats']}")

    print("\n" + "═" * 65 + "\n")


    # استبدلي الجزء ده في آخر evaluate.py

if __name__ == "__main__":
    import sys
    print("evaluate.py starting...", flush=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    
    print(f"Image: {args.image}", flush=True)
    
    # Check result file first
    result_path = "outputs/translation/translation_result.json"
    if not os.path.exists(result_path):
        print(f"ERROR: File not found: {result_path}", flush=True)
        print("Run this first:", flush=True)
        print('  python main.py translate --image "your_image.png" --save', flush=True)
        sys.exit(1)
    
    print(f"Found result file: {result_path}", flush=True)
    run_evaluation(args.image)