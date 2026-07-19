"""
translator.py — Stage 4
Full translation pipeline using Groq API + Llama 3.
Input : StructuredSequence from Stage 3
Output: TranslationResult (English + Arabic + word analysis)
"""

import os
import sys
import json
from dataclasses import dataclass, field


# ── Config loader ─────────────────────────────────────────────────────────────

def _load_config():
    import yaml
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path  = os.path.join(project_root, "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Gardiner Dictionary ───────────────────────────────────────────────────────

GARDINER_DICT = {
    # Section A
    "A1":  {"phonetic": "",     "type": "determinative", "meaning": "man, male person, I, me"},
    "A2":  {"phonetic": "",     "type": "determinative", "meaning": "eat, drink, speak, think"},
    "A3":  {"phonetic": "",     "type": "determinative", "meaning": "sit, be seated"},
    "A4":  {"phonetic": "",     "type": "determinative", "meaning": "pray, worship, adore"},
    "A17": {"phonetic": "",     "type": "determinative", "meaning": "child, young"},
    "A40": {"phonetic": "",     "type": "determinative", "meaning": "god, seated deity"},
    "A41": {"phonetic": "",     "type": "determinative", "meaning": "king, pharaoh"},
    # Section B
    "B1":  {"phonetic": "",     "type": "determinative", "meaning": "woman, female person"},
    # Section C
    "C1":  {"phonetic": "",     "type": "determinative", "meaning": "Ra (sun god, falcon head)"},
    "C3":  {"phonetic": "",     "type": "determinative", "meaning": "Thoth (ibis head)"},
    "C4":  {"phonetic": "",     "type": "determinative", "meaning": "Anubis (jackal head)"},
    # Section D
    "D1":  {"phonetic": "tp",   "type": "ideogram",      "meaning": "head; top, upon"},
    "D2":  {"phonetic": "Hr",   "type": "ideogram",      "meaning": "face, upon, concerning"},
    "D4":  {"phonetic": "ir",   "type": "ideogram",      "meaning": "eye; do, make"},
    "D21": {"phonetic": "r",    "type": "phonogram",     "meaning": "mouth → phonetic r"},
    "D28": {"phonetic": "kA",   "type": "ideogram",      "meaning": "two arms raised, Ka spirit"},
    "D35": {"phonetic": "n",    "type": "phonogram",     "meaning": "arms in negation → not, no"},
    "D36": {"phonetic": "a",    "type": "phonogram",     "meaning": "forearm → phonetic a (ayin)"},
    "D37": {"phonetic": "di",   "type": "phonogram",     "meaning": "arm holding bread → give"},
    "D40": {"phonetic": "",     "type": "determinative", "meaning": "arm with stick, force, strength"},
    "D46": {"phonetic": "d",    "type": "phonogram",     "meaning": "hand → phonetic d"},
    "D54": {"phonetic": "",     "type": "determinative", "meaning": "legs walking, go, come"},
    "D58": {"phonetic": "b",    "type": "phonogram",     "meaning": "foot → phonetic b"},
    "D60": {"phonetic": "wab",  "type": "phonogram",     "meaning": "foot in water → pure, clean"},
    # Section E
    "E1":  {"phonetic": "kA",   "type": "ideogram",      "meaning": "bull, ox"},
    "E17": {"phonetic": "sAb",  "type": "phonogram",     "meaning": "jackal → sAb"},
    "E22": {"phonetic": "mAi",  "type": "ideogram",      "meaning": "lion lying"},
    # Section F
    "F13": {"phonetic": "db",   "type": "phonogram",     "meaning": "horn → db"},
    "F31": {"phonetic": "ms",   "type": "phonogram",     "meaning": "three fox skins → give birth"},
    "F34": {"phonetic": "ib",   "type": "ideogram",      "meaning": "heart"},
    "F35": {"phonetic": "nfr",  "type": "phonogram",     "meaning": "heart and trachea → good, beautiful"},
    # Section G
    "G1":  {"phonetic": "A",    "type": "phonogram",     "meaning": "vulture → Egyptian Aleph"},
    "G5":  {"phonetic": "Hr",   "type": "ideogram",      "meaning": "falcon (Horus)"},
    "G14": {"phonetic": "",     "type": "determinative", "meaning": "vulture (Nekhbet)"},
    "G17": {"phonetic": "m",    "type": "phonogram",     "meaning": "owl → phonetic m"},
    "G18": {"phonetic": "mm",   "type": "phonogram",     "meaning": "two owls → mm"},
    "G26": {"phonetic": "Dhwty","type": "ideogram",      "meaning": "ibis (Thoth)"},
    "G26a":{"phonetic": "Dsr",  "type": "phonogram",     "meaning": "sacred ibis variant → sacred, holy"},
    "G29": {"phonetic": "bA",   "type": "ideogram",      "meaning": "jabiru stork → soul (Ba)"},
    "G36": {"phonetic": "wr",   "type": "phonogram",     "meaning": "swallow → great, elder"},
    "G37": {"phonetic": "nDs",  "type": "phonogram",     "meaning": "sparrow → small, bad"},
    "G39": {"phonetic": "sA",   "type": "phonogram",     "meaning": "pintail duck → son"},
    "G41": {"phonetic": "pA",   "type": "phonogram",     "meaning": "duck flying → fly, that"},
    "G43": {"phonetic": "w",    "type": "phonogram",     "meaning": "quail chick → phonetic w"},
    # Section I
    "I9":  {"phonetic": "f",    "type": "phonogram",     "meaning": "horned viper → phonetic f"},
    "I10": {"phonetic": "D",    "type": "phonogram",     "meaning": "cobra upright → phonetic D (dj)"},
    # Section M
    "M1":  {"phonetic": "iAm",  "type": "ideogram",      "meaning": "tree"},
    "M3":  {"phonetic": "xt",   "type": "ideogram",      "meaning": "branch, wood"},
    "M13": {"phonetic": "wAD",  "type": "ideogram",      "meaning": "papyrus stem → green, fresh, flourish"},
    "M17": {"phonetic": "i",    "type": "phonogram",     "meaning": "reed → phonetic i (yod)"},
    "M23": {"phonetic": "sw",   "type": "phonogram",     "meaning": "sedge plant → Upper Egypt"},
    # Section N
    "N1":  {"phonetic": "pt",   "type": "ideogram",      "meaning": "sky"},
    "N5":  {"phonetic": "ra",   "type": "ideogram",      "meaning": "sun disk → Ra, time, day"},
    "N14": {"phonetic": "sbA",  "type": "ideogram",      "meaning": "star → teach, star"},
    "N16": {"phonetic": "tA",   "type": "ideogram",      "meaning": "flat land → earth"},
    "N35": {"phonetic": "n",    "type": "phonogram",     "meaning": "water ripple → phonetic n"},
    "N37": {"phonetic": "S",    "type": "phonogram",     "meaning": "pool → phonetic S (sh)"},
    # Section O
    "O1":  {"phonetic": "pr",   "type": "ideogram",      "meaning": "house plan → house, go out"},
    "O6":  {"phonetic": "Hwt",  "type": "ideogram",      "meaning": "enclosure → temple, estate"},
    "O22": {"phonetic": "",     "type": "determinative", "meaning": "shrine with naos"},
    "O34": {"phonetic": "z",    "type": "phonogram",     "meaning": "door bolt → phonetic s/z"},
    "O49": {"phonetic": "niwt", "type": "ideogram",      "meaning": "town with crossroads → city"},
    # Section Q
    "Q1":  {"phonetic": "st",   "type": "ideogram",      "meaning": "throne/seat → Isis"},
    "Q3":  {"phonetic": "p",    "type": "phonogram",     "meaning": "stool → phonetic p"},
    # Section R
    "R4":  {"phonetic": "Htp",  "type": "ideogram",      "meaning": "offering table → offer, peace"},
    "R8":  {"phonetic": "nTr",  "type": "ideogram",      "meaning": "flag/standard → god, divine"},
    "R11": {"phonetic": "Dd",   "type": "ideogram",      "meaning": "djed pillar → stability, endurance"},
    # Section S
    "S29": {"phonetic": "s",    "type": "phonogram",     "meaning": "folded cloth → phonetic s"},
    "S34": {"phonetic": "anx",  "type": "ideogram",      "meaning": "sandal strap → life (ankh)"},
    "S40": {"phonetic": "wAs",  "type": "ideogram",      "meaning": "was-scepter → power, dominion"},
    # Section T
    "T14": {"phonetic": "",     "type": "determinative", "meaning": "throw stick → foreign, Libyan"},
    # Section V
    "V13": {"phonetic": "T",    "type": "phonogram",     "meaning": "hobble rope → phonetic T"},
    "V28": {"phonetic": "H",    "type": "phonogram",     "meaning": "twisted wick → phonetic H"},
    "V30": {"phonetic": "nb",   "type": "ideogram",      "meaning": "basket → lord, all, every"},
    "V31": {"phonetic": "k",    "type": "phonogram",     "meaning": "basket with handle → phonetic k"},
    # Section W
    "W3":  {"phonetic": "Hb",   "type": "phonogram",     "meaning": "alabaster basin → festival"},
    "W24": {"phonetic": "nw",   "type": "phonogram",     "meaning": "pot → phonetic nw"},
    # Section X
    "X1":  {"phonetic": "t",    "type": "phonogram",     "meaning": "bread loaf → phonetic t"},
    "X8":  {"phonetic": "di",   "type": "phonogram",     "meaning": "conical bread → give"},
    # Section Y
    "Y1":  {"phonetic": "mDAt", "type": "ideogram",      "meaning": "papyrus scroll → write, book"},
    # Section Z
    "Z1":  {"phonetic": "",     "type": "determinative", "meaning": "single stroke — numeral 1 or ideogram marker"},
    "Z2":  {"phonetic": "",     "type": "determinative", "meaning": "three strokes — plural marker"},
    "Z4":  {"phonetic": "y",    "type": "phonogram",     "meaning": "two diagonal strokes → phonetic y"},
    "Z4a": {"phonetic": "y",    "type": "phonogram",     "meaning": "hieratic dual strokes → phonetic y"},
    # Section Aa
    "Aa1": {"phonetic": "x",    "type": "phonogram",     "meaning": "placenta → phonetic x (kh)"},
    "Aa27":{"phonetic": "nD",   "type": "phonogram",     "meaning": "unclassified → protect"},

        # ── Missing signs discovered from real inference ──────────
    "N36": {"phonetic": "mr",  "type": "phonogram",    "meaning": "canal, water channel → mr"},
    "N33": {"phonetic": "",    "type": "determinative", "meaning": "grain of sand, mineral"},
    "V23": {"phonetic": "wp",  "type": "phonogram",    "meaning": "cord on stick → wp (open)"},
    "V4":  {"phonetic": "wA",  "type": "phonogram",    "meaning": "lasso → wA"},
    "M22": {"phonetic": "nn",  "type": "phonogram",    "meaning": "rush with root → nn"},
    "O22": {"phonetic": "",    "type": "determinative", "meaning": "shrine with naos"},
    "F13": {"phonetic": "db",  "type": "phonogram",    "meaning": "horn → db"},
    "G26a":{"phonetic": "Dsr", "type": "phonogram",    "meaning": "sacred ibis → sacred, holy"},
    "Z4a": {"phonetic": "y",   "type": "phonogram",    "meaning": "dual strokes → phonetic y"},
    "D60": {"phonetic": "wab", "type": "phonogram",    "meaning": "foot in water → pure, clean"},
    "M13": {"phonetic": "wAD", "type": "ideogram",     "meaning": "papyrus stem → green, flourish"},
    "D37": {"phonetic": "di",  "type": "phonogram",    "meaning": "arm with bread → give"},




}



def lookup(code: str) -> dict:
    entry = GARDINER_DICT.get(code)
    if entry:
        return {**entry, "code": code, "found": True}
    return {
        "code": code, "phonetic": "", "type": "unknown",
        "meaning": f"Sign {code} (not in local dictionary)",
        "found": False
    }


# ── Transliteration builder ───────────────────────────────────────────────────

def build_transliteration(codes: list, sign_types: list) -> str:
    """Convert Gardiner codes to MdC transliteration string."""
    tokens = []
    for code, stype in zip(codes, sign_types):
        entry    = lookup(code)
        phonetic = entry["phonetic"]
        if stype in ("phonogram", "ideogram") and phonetic:
            tokens.append(phonetic)
        elif stype == "determinative":
            tokens.append(f"[{code}]")
        elif phonetic:
            tokens.append(phonetic)
        else:
            tokens.append(f"<{code}>")
    return " ".join(tokens)


# ── Groq LLM client ───────────────────────────────────────────────────────────

def get_groq_client():
    try:
        from groq import Groq
    except ImportError:
        print("[Stage4] ERROR: groq not installed.")
        print("         Run: pip install groq")
        sys.exit(1)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[Stage4] ERROR: GROQ_API_KEY not set.")
        print("         1. Go to: https://console.groq.com")
        print("         2. Create a free API key")
        print("         3. Run: set GROQ_API_KEY=gsk_...")
        sys.exit(1)

    return Groq(api_key=api_key)


SYSTEM_PROMPT = """You are a strict hieroglyph translator. 

ABSOLUTE RULES:
1. Translate EVERY sign group in order — do not skip any
2. Do NOT summarize — give the FULL translation word by word
3. Do NOT say "fragmentary" unless more than 50% of signs are unknown
4. Show the complete translation as a sequence, not a summary

OUTPUT FORMAT — JSON only:
{
  "english_translation": "full sequential translation of every word group in order",
  "arabic_translation": "الترجمة الكاملة لكل مجموعة رموز بالترتيب",
  "transliteration_refined": "complete MdC transliteration",
  "word_analysis": [
    {
      "codes": ["G17", "N35"],
      "transliteration": "mn",
      "english": "to endure/remain",
      "arabic": "يدوم/يبقى"
    }
  ],
  "context": "type of text (offering formula/royal title/temple inscription/etc)",
  "confidence": "high/medium/low",
  "caveats": "only list genuinely unknown signs"
}

EXAMPLE of correct output:
english_translation: "The pure house of Ra endures. The good god, lord of all. Peace offering to Osiris. The son of Ra, beloved of Amun."

EXAMPLE of WRONG output (never do this):
english_translation: "The text contains mn, nfr, Htp..." ← THIS IS WRONG, never summarize
"""

def build_prompt(seq) -> str:
    """Build the user prompt from a StructuredSequence."""
    lines = []

    # Sign table
    lines.append("## Detected Hieroglyphic Signs")
    lines.append(f"Direction: {seq.direction.upper()} | Total signs: {len(seq.all_codes)}\n")
    lines.append("| # | Code | Phonetic | Type | Meaning |")
    lines.append("|---|------|----------|------|---------|")

    for i, (code, stype) in enumerate(zip(seq.all_codes, seq.all_types)):
        entry    = lookup(code)
        phonetic = entry["phonetic"] or "—"
        meaning  = entry["meaning"][:45]
        lines.append(f"| {i+1:2d} | {code:8s} | {phonetic:8s} | {stype:15s} | {meaning} |")

    # Transliteration
    translit = build_transliteration(seq.all_codes, seq.all_types)
    lines.append(f"\n## Transliteration\n{translit}")

    # Word groups
    lines.append("\n## Word Groups (from spatial analysis)")
    for i, w in enumerate(seq.words):
        ph = "+".join(w.phonetic_signs) or "(no phonetic signs)"
        dt = ",".join(w.determinatives) or "—"
        lines.append(f"Word {i+1}: signs={'+'.join(w.signs)} | phonetic={ph} | det={dt}")

    # Visual lines
    lines.append("""
## TRANSLATION INSTRUCTIONS
- Group consecutive phonetic signs into Egyptian words FIRST
- Common groupings from this text:
  * m+n = mn (to endure/remain)
  * n+f+r = nfr (good/beautiful)  
  * H+t+p = Htp (peace/offering)
  * w+a+b = wab (pure/clean)
  * p+r = pr (house)
  * n+T+r = nTr (god/divine)
  * r+a = ra (sun/Ra)
- THEN translate the grouped words
- Determinatives in [] classify the word before them
- Write: phonetic_group [determinative] = meaning
""")  


    for i, line in enumerate(seq.lines):
        lines.append(f"Line {i+1}: {' '.join(line)}")
      
    return "\n".join(lines)

# أضيفي الجزء ده في آخر build_prompt قبل return


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class TranslationResult:
    english_translation   : str
    arabic_translation    : str
    transliteration       : strلإ
    transliteration_refined: str
    word_analysis         : list
    context               : str
    confidence            : str
    caveats               : str
    raw_codes             : list
    direction             : str

    def pretty_print(self):
        w = 65
        print("\n" + "═" * w)
        print("  🏛  HIEROGLYPH TRANSLATION RESULT")
        print("═" * w)
        print(f"\n  Direction        : {self.direction.upper()}")
        print(f"  Signs            : {' '.join(self.raw_codes)}")
        print(f"\n  Transliteration  : {self.transliteration}")
        if self.transliteration_refined:
            print(f"  Refined          : {self.transliteration_refined}")
        print(f"\n  🇬🇧 English  :\n  {self.english_translation}")
        print(f"\n  🇪🇬 Arabic   :\n  {self.arabic_translation}")

        if self.word_analysis:
            print(f"\n  Word-by-Word Analysis:")
            print(f"  {'Codes':<20} {'Translit':<12} English / Arabic")
            print("  " + "-" * 55)
            for w_entry in self.word_analysis:
                codes    = "+".join(w_entry.get("codes", []))
                translit = w_entry.get("transliteration", "")
                eng      = w_entry.get("english", "")
                arb      = w_entry.get("arabic", "")
                print(f"  [{codes:<18}] {translit:<12} {eng} / {arb}")

        if self.context:
            print(f"\n  Context    : {self.context}")
        print(f"\n  Confidence : {self.confidence}")
        if self.caveats:
            print(f"  Caveats    : {self.caveats}")
        print("═" * w + "\n")

    def to_dict(self) -> dict:
        return {
            "english_translation"    : self.english_translation,
            "arabic_translation"     : self.arabic_translation,
            "transliteration"        : self.transliteration,
            "transliteration_refined": self.transliteration_refined,
            "word_analysis"          : self.word_analysis,
            "context"                : self.context,
            "confidence"             : self.confidence,
            "caveats"                : self.caveats,
            "raw_codes"              : self.raw_codes,
            "direction"              : self.direction,
        }


# ── Main pipeline ─────────────────────────────────────────────────────────────

class Stage4Pipeline:

    def __init__(self):
        cfg         = _load_config()
        tr          = cfg.get("translation", {})
        self.model  = tr.get("llm_model", "llama-3.3-70b-versatile")
        self.max_tok= int(tr.get("max_tokens", 2000))
        self.out_dir= tr.get("output_dir", "outputs/translation")
        os.makedirs(self.out_dir, exist_ok=True)
        self.client = get_groq_client()

    def translate(self, seq) -> TranslationResult:
        """Full Stage 4: seq → TranslationResult."""

        translit   = build_transliteration(seq.all_codes, seq.all_types)
        user_prompt= build_prompt(seq)

        print(f"[Stage4] Sending {len(seq.all_codes)} signs to Groq ({self.model})...")

        response = self.client.chat.completions.create(
            model    = self.model,
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens  = self.max_tok,
            temperature = 0.0,   # low temp for more deterministic Egyptology output
        )

        raw_text = response.choices[0].message.content.strip()

        # Parse JSON
        try:
            if "```" in raw_text:
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            llm = json.loads(raw_text)
        except json.JSONDecodeError as e:
            print(f"[Stage4] Warning: JSON parse failed ({e}) — using raw text")
            llm = {
                "english_translation"    : raw_text,
                "arabic_translation"     : "",
                "transliteration_refined": "",
                "word_analysis"          : [],
                "context"                : "",
                "confidence"             : "low",
                "caveats"                : "JSON parsing failed; raw LLM output shown.",
            }

        result = TranslationResult(
            english_translation    = llm.get("english_translation", ""),
            arabic_translation     = llm.get("arabic_translation", ""),
            transliteration        = translit,
            transliteration_refined= llm.get("transliteration_refined", translit),
            word_analysis          = llm.get("word_analysis", []),
            context                = llm.get("context", ""),
            confidence             = llm.get("confidence", "low"),
            caveats                = llm.get("caveats", ""),
            raw_codes              = seq.all_codes,
            direction              = seq.direction,
        )

        result.pretty_print()
        return result

    def save(self, result: TranslationResult, out_path: str = None):
        if out_path is None:
            out_path = os.path.join(self.out_dir, "translation_result.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"[Stage4] Result saved: {out_path}")