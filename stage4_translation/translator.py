"""
translator.py — Stage 4
Full translation pipeline using Groq API + Llama 3.
Input : StructuredSequence from Stage 3
Output: TranslationResult (English + Arabic + word analysis)
"""

import os
import sys
import json
import re
import base64
import mimetypes
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


SYSTEM_PROMPT = """You are an expert Egyptologist and ancient Egyptian translation assistant.

Examine the actual attached image first. Then use the image and the compact Stage 1–3 evidence to produce the closest natural translation or semantic reconstruction of the inscription and scene. The user wants the meaning directly in complete sentences, not a technical report, OCR output, image-analysis notes, or a list of signs. Never describe the material, color, surface, carving technique, layout, columns, shapes, birds, eyes, or general artistic quality unless that detail is itself part of the translated message.

Use the visible image as the primary evidence. Use detected signs, classifications, reading direction, and word-group candidates only as supporting evidence. Treat detector codes such as D12, R8, N35, A40, G17, and Z1 as machine labels, not as Egyptian words. Never turn isolated phonetic fragments into invented sentences.

When some words or grammatical links are missing, reconstruct the meaning naturally with the smallest suitable connecting words. Preserve the reliable subject, action, people, objects, and relationships. This is a contextual reconstruction, so make the result fluent and meaningful, but do not add a specific name, deity, ritual, prayer, historical period, or event unless it is supported by the image or reliable grouped meanings. If the exact wording cannot be recovered, give the closest supported semantic message in one or two clear sentences; do not replace it with a description of the stone, ink, colors, columns, symbols, craftsmanship, or image quality.

Never assume Papyrus of Ani, the Book of the Dead, Ani, Osiris, Anubis, Thoth, Maat, the weighing of the heart, the Hall of Judgment, the Fields of Iaru, or any other fixed story unless the actual image or reliable evidence supports it. If a visual identification is uncertain, use cautious wording such as “appears to” or “likely,” but still provide a useful complete translation.

The English and Arabic translations must begin directly with the meaning, action, person, object, or event. Do not begin with or mention “The text”, “The inscription”, “The image”, “Image description”, “Description”, “Visually”, “This scene”, “يُصوّر النص”, “النص”, “النقش”, “الصورة”, “وصف الصورة”, “الوصف”, “بصريًا”, “هذا المشهد”, “المعنى العام هو”, “يُظهر”, or any similar meta-introduction. Do not say that something is being shown, depicted, described, written, carved, arranged, or visible. Start with the actual content, such as “The deceased stands before the great scales...” or “يقف المتوفى أمام الميزان العظيم...”.

The English translation must be a coherent paragraph of approximately 80–180 words for a complex image. The Arabic translation must be a complete equivalent in formal Classical Arabic (العربية الفصحى), with the same information and similar detail. Do not make the Arabic a short summary. The Arabic section must contain Arabic script only, apart from normal Arabic punctuation and numerals when necessary. Never include English letters, English words, Latin transliteration, Gardiner codes, or mixed forms in the Arabic section. Translate all names and terms into Arabic script: Maat → ماعت, Osiris → أوزير, Anubis → أنوبيس, Thoth → تحوت, Ra → رع, and Iaru/Aaru → يارو.

Never output Gardiner codes, detector labels, coordinates, confidence percentages from detection, line numbers, classification names, raw OCR, raw transliteration, isolated phonetic fragments, Markdown lists, Markdown bold markers, or internal reasoning. Never output a <think> block. Do not mention the model, the pipeline, Stage 1, Stage 2, Stage 3, the prompt, tokens, API, image processing, or parsing. Never output generic visual-description phrases such as “the inscription is carved on dark stone,” “white symbols stand out,” “the signs are arranged in columns,” “the image contains birds and eyes,” “the craftsmanship is detailed,” “the scene appears official,” or equivalent wording. Do not describe what the image is made of; translate what it means.

Do not write Context, Caveats, analysis, reasoning, a visual-analysis plan, a draft, a final-check section, or any explanation outside the two translations and the confidence value. The translation fields must contain only direct semantic sentences, not labels or commentary about the source material.

Return ONLY these three sections in exactly this order and with exactly these headings:

Translation in English:
Write the complete natural English translation or semantic reconstruction here. Start directly with the meaning or action, person, object, or event. Never start with “The text”, “The inscription”, “The image”, “This scene”, or “Description”. If the readable evidence is insufficient for a literal sentence, provide only the closest supported meaning and do not write a physical or artistic description.

Translation in Arabic:
اكتب الترجمة أو إعادة البناء الدلالية بالعربية الفصحى هنا. ابدأ بالفعل أو الشخص أو الشيء أو الحدث مباشرة، ولا تبدأ بكلمة «النص» أو «النقش» أو «الصورة» أو «الوصف» أو «يُظهر» أو «يُصوّر». يجب أن يحتوي هذا القسم على حروف عربية فقط، من دون أي كلمة إنجليزية أو ترانسلِتريشن أو أكواد أو حروف لاتينية. اكتب «ماعت» بدل Maat، و«أوزير» بدل Osiris، و«أنوبيس» بدل Anubis، و«تحوت» بدل Thoth، و«رع» بدل Ra، و«يارو» بدل Iaru أو Aaru. إذا تعذرت القراءة الحرفية، اكتب أقرب معنى تدعمه العلامات والمشهد فقط، ولا تصف الحجر أو الألوان أو الأعمدة أو شكل الرموز أو جودة الحفر.

Confidence:
Write exactly one word: high, medium, or low.

Before responding, silently verify that the response contains only these three sections, that both translations are complete sentences, that no raw symbols or technical commentary appear, and that the English and Arabic convey the same meaning."""















def build_prompt(seq) -> str:
    """Build a compact prompt; the attached image is the primary evidence."""
    lines = [
        "Analyze the attached image first and explain the specific object or scene in complete sentences.",
        "The image is the primary source. Use the Stage 3 data only as secondary, noisy evidence.",
        f"Reading direction={seq.direction.upper()}; detected signs={len(seq.all_codes)}.",
        "Describe the visible figures, objects, actions, layout, and likely subject of the scene.",
        "Then provide an evidence-based English and formal Arabic explanation.",
        "Do not list signs, codes, phonetic fragments, or raw transliteration.",
        "Do not assume Ani, the Book of the Dead, a specific god, ritual, or funerary formula unless the image supports it.",
        "If the inscription is unclear, explain the visible scene instead of refusing or repeating the sign sequence.",
    ]

    # Send only a small number of compact word-group hints. The complete sign
    # table and transliteration duplicate information already visible in image
    # processing and can push a Vision request over Groq's TPM limit.
    reliable_groups = []
    for word in getattr(seq, "words", [])[:12]:
        signs = "+".join(getattr(word, "signs", [])[:5])
        phonetics = "+".join(getattr(word, "phonetic_signs", [])[:5])
        if signs or phonetics:
            reliable_groups.append(f"{signs}|{phonetics}")
    if reliable_groups:
        lines.append("Secondary Stage 3 hints (do not treat uncertain fragments as translation):")
        lines.extend(reliable_groups)
    else:
        lines.append("No reliable textual word groups are available; rely on the image for the scene explanation.")

    return "\n".join(lines)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class TranslationResult:
    english_translation   : str
    arabic_translation    : str
    transliteration       : str
    transliteration_refined: str
    word_analysis         : list
    context               : str
    confidence            : str
    caveats               : str
    raw_codes             : list
    direction             : str

    def pretty_print(self):
        """Display only the user-facing translations and confidence."""
        print("\n" + "=" * 72)
        print("Translation in English:")
        print(self.english_translation.strip())
        print("\nTranslation in Arabic:")
        print(self.arabic_translation.strip())
        print(f"\nConfidence: {self.confidence}")
        print("=" * 72 + "\n")

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

    def __init__(self, image_path=None):
        cfg = _load_config()
        tr = cfg.get("translation", {})
        self.model = tr.get("llm_model", "qwen/qwen3.6-27b")
        self.max_tok = int(tr.get("max_tokens", 1500))
        self.out_dir = tr.get("output_dir", "outputs/translation")
        self.image_path = image_path
        os.makedirs(self.out_dir, exist_ok=True)
        self.client = get_groq_client()

    @staticmethod
    def _image_data_url(image_path: str):
        """Encode a local image in the data-URL format required by Groq Vision."""
        if not image_path or not os.path.isfile(image_path):
            return None
        mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _clean_arabic_translation(value: str) -> str:
        """Keep the Arabic field free of English letters and transliteration."""
        text = (value or "").strip()
        replacements = {
            "Maat": "ماعت",
            "Ma'at": "ماعت",
            "Osiris": "أوزير",
            "Anubis": "أنوبيس",
            "Thoth": "تحوت",
            "Ra": "رع",
            "Re": "رع",
            "Iaru": "يارو",
            "Aaru": "يارو",
            "Horus": "حورس",
            "Isis": "إيزيس",
            "Seth": "ست",
            "Amun": "آمون",
            "Ptah": "بتاح",
            "Aten": "آتون",
            "the deceased": "المتوفى",
            "the heart": "القلب",
            "the gods": "الآلهة",
            "the god": "الإله",
            "heart": "القلب",
            "truth": "الحق",
            "life": "الحياة",
            "death": "الموت",
        }
        for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
            text = re.sub(rf"(?<![A-Za-z]){re.escape(source)}(?![A-Za-z])", target, text, flags=re.IGNORECASE)
        # Remove any remaining Latin words/letters, including accidental transliteration.
        text = re.sub(r"[A-Za-z]+", "", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\s+([،؛:.)،؟])", r"\1", text)
        text = re.sub(r"([،؛:(])\s+", r"\1", text)
        return text.strip(" \t\r\n-–—:؛،")

    @staticmethod
    def _parse_text_response(raw_text: str) -> dict:
        """Extract only the English translation, Arabic translation, and confidence."""
        text = (raw_text or "").strip()
        # Qwen may return an open <think> block when the request is truncated.
        # Remove both closed and unclosed reasoning blocks before parsing.
        text = re.sub(r"<think>.*?(?:</think>|\Z)", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

        english_match = re.search(
            r"(?:Translation in English|English Translation|English)\s*:\s*(.*?)(?=\n\s*(?:Translation in Arabic|Arabic Translation|Arabic)\s*:|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        arabic_match = re.search(
            r"(?:Translation in Arabic|Arabic Translation|Arabic|الترجمة بالعربية|الترجمة العربية)\s*:\s*(.*?)(?=\n\s*Confidence\s*:|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        confidence_match = re.search(
            r"(?:Confidence|الثقة)\s*:\s*(high|medium|low|مرتفع|متوسط|منخفض)",
            text,
            flags=re.IGNORECASE,
        )

        english = english_match.group(1).strip() if english_match else ""
        arabic = arabic_match.group(1).strip() if arabic_match else ""
        arabic = Stage4Pipeline._clean_arabic_translation(arabic)
        confidence = confidence_match.group(1).lower() if confidence_match else "medium"
        confidence = {"مرتفع": "high", "متوسط": "medium", "منخفض": "low"}.get(confidence, confidence)

        # If the model omitted headings, keep the raw response as Arabic only
        # rather than exposing technical content or losing the result entirely.
        if not english and not arabic:
            arabic = text

        return {
            "english_translation": english,
            "arabic_translation": arabic,
            "context": "",
            "confidence": confidence,
            "caveats": "",
        }

    @staticmethod
    def _parse_json(raw_text: str) -> dict:
        """Parse Groq JSON and safely handle an unexpected fallback."""
        raw_text = (raw_text or "").strip()
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            raw_text = "\n".join(lines).strip()

        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("response is JSON but not an object")
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[Stage4] Warning: JSON parse failed ({exc})")
            print(f"[Stage4] Raw response preview: {raw_text[:300]!r}")
            return {
                "english_translation": raw_text,
                "arabic_translation": "",
                "transliteration_refined": "",
                "word_analysis": [],
                "context": "",
                "confidence": "low",
                "caveats": "The model returned invalid JSON.",
            }

    def translate(self, seq) -> TranslationResult:
        """Translate the complete sequence in one Groq request."""
        translit = build_transliteration(seq.all_codes, seq.all_types)
        user_prompt = build_prompt(seq)

        print(f"[Stage4] Sending {len(seq.all_codes)} signs to Groq ({self.model})...")

        # Vision mode: the image is the primary source and Stage 3 data is
        # supplied only as compact secondary evidence.
        image_url = self._image_data_url(self.image_path)
        if not image_url:
            raise FileNotFoundError(
                "Stage 4 requires a readable image for Vision translation. "
                f"Received image_path={self.image_path!r}. "
                "Check the --image argument and main.py."
            )

        user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        print(f"[Stage4] Vision image attached: {self.image_path}")

        # Groq's free/on-demand limit is 8000 TPM in the user's error.
        # Estimate the compact text request plus a conservative image reserve.
        estimated_input_chars = len(SYSTEM_PROMPT) + len(user_prompt)
        estimated_prompt_tokens = max(1, estimated_input_chars // 3)
        estimated_image_tokens = 1500
        estimated_total_input = estimated_prompt_tokens + estimated_image_tokens
        request_budget = max(800, 7200 - estimated_total_input)
        request_max_tokens = min(self.max_tok, request_budget)
        print(
            f"[Stage4] Estimated input={estimated_total_input} tokens; "
            f"max_completion_tokens={request_max_tokens}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=request_max_tokens,
            temperature=0.0,
            # Qwen 3.6 supports none; this prevents <think> from consuming
            # the completion budget before the two translations are returned.
            reasoning_effort="none",
        )

        choice = response.choices[0]
        raw_text = (choice.message.content or "").strip()
        usage = getattr(response, "usage", None)
        print(
            f"[Stage4] finish_reason={choice.finish_reason}; "
            f"prompt_tokens={getattr(usage, 'prompt_tokens', '?')}; "
            f"completion_tokens={getattr(usage, 'completion_tokens', '?')}"
        )
        if choice.finish_reason == "length":
            print("[Stage4] WARNING: response truncated; reduce prompt or increase max_tokens.")

        llm = self._parse_text_response(raw_text)

        result = TranslationResult(
            english_translation=llm.get("english_translation", ""),
            arabic_translation=llm.get("arabic_translation", ""),
            transliteration=translit,
            transliteration_refined=translit,
            word_analysis=[],
            context=llm.get("context", ""),
            confidence=llm.get("confidence", "low"),
            caveats=llm.get("caveats", ""),
            raw_codes=seq.all_codes,
            direction=seq.direction,
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