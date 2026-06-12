"""
Script-specific typography rules for the 10 supported Indic languages
(+ Latin for mixed-script pages).

Each entry carries everything the renderer and generator need:
    font_file       : filename under data/fonts/
    hb_script       : HarfBuzz/ISO-15924 script tag (informational; HarfBuzz
                      auto-detects from the codepoints, but we keep it for
                      logging and the RTL decision)
    direction       : "ltr" or "rtl"
    has_shirorekha  : whether the script has the headline stroke (affects QC)
    script_head     : label for the 9-way auxiliary script-classification head
    unicode_ranges  : (start, end) inclusive codepoint ranges for the script's
                      letters, used to synthesise placeholder words when a real
                      corpus is not supplied.

NOTE on placeholder text: random codepoints are fine for *layout* pretraining
(the model learns where text blocks sit, not what they say). For publishable
quality, drop a real word list into data/text_corpus/<lang>_words.txt and the
corpus loader will prefer it automatically.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class ScriptSpec:
    name: str
    font_file: str
    hb_script: str
    direction: str
    has_shirorekha: bool
    script_head: str
    unicode_ranges: List[Tuple[int, int]] = field(default_factory=list)


# Unicode consonant/vowel ranges chosen to avoid combining-mark-only codepoints
# at word start (which would render as dotted circles). We sample a base letter
# first, then optionally a matra, in corpus.py.
REGISTRY = {
    "Hindi": ScriptSpec(
        "Hindi", "NotoSansDevanagari.ttf", "Deva", "ltr", True, "Devanagari",
        [(0x0905, 0x0939), (0x0958, 0x0961)],
    ),
    "Bengali": ScriptSpec(
        "Bengali", "NotoSansBengali.ttf", "Beng", "ltr", True, "Bengali",
        [(0x0985, 0x098C), (0x098F, 0x0990), (0x0993, 0x09B9)],
    ),
    "Tamil": ScriptSpec(
        "Tamil", "NotoSansTamil.ttf", "Taml", "ltr", False, "Tamil",
        [(0x0B85, 0x0B8A), (0x0B8E, 0x0B90), (0x0B92, 0x0BB9)],
    ),
    "Telugu": ScriptSpec(
        "Telugu", "NotoSansTelugu.ttf", "Telu", "ltr", False, "Telugu",
        [(0x0C05, 0x0C0C), (0x0C0E, 0x0C10), (0x0C12, 0x0C39)],
    ),
    "Kannada": ScriptSpec(
        "Kannada", "NotoSansKannada.ttf", "Knda", "ltr", False, "Kannada",
        [(0x0C85, 0x0C8C), (0x0C8E, 0x0C90), (0x0C92, 0x0CB9)],
    ),
    "Malayalam": ScriptSpec(
        "Malayalam", "NotoSansMalayalam.ttf", "Mlym", "ltr", False, "Malayalam",
        [(0x0D05, 0x0D0C), (0x0D0E, 0x0D10), (0x0D12, 0x0D39)],
    ),
    "Gujarati": ScriptSpec(
        "Gujarati", "NotoSansGujarati.ttf", "Gujr", "ltr", False, "Gujarati",
        [(0x0A85, 0x0A8B), (0x0A8F, 0x0A91), (0x0A93, 0x0AB9)],
    ),
    "Punjabi": ScriptSpec(
        "Punjabi", "NotoSansGurmukhi.ttf", "Guru", "ltr", True, "Gurmukhi",
        [(0x0A05, 0x0A0A), (0x0A0F, 0x0A10), (0x0A13, 0x0A39)],
    ),
    "Odia": ScriptSpec(
        "Odia", "NotoSansOriya.ttf", "Orya", "ltr", False, "Odia",
        [(0x0B05, 0x0B0C), (0x0B0F, 0x0B10), (0x0B13, 0x0B39)],
    ),
    "Urdu": ScriptSpec(
        "Urdu", "NotoNaskhArabic.ttf", "Arab", "rtl", False, "Arabic",
        [(0x0627, 0x063A), (0x0641, 0x064A)],
    ),
    "English": ScriptSpec(
        "English", "NotoSans.ttf", "Latn", "ltr", False, "Latin",
        [(0x0041, 0x005A), (0x0061, 0x007A)],
    ),
}


def get_spec(language: str) -> ScriptSpec:
    if language not in REGISTRY:
        raise KeyError(f"Unknown language '{language}'. Known: {list(REGISTRY)}")
    return REGISTRY[language]


def is_rtl(language: str) -> bool:
    return get_spec(language).direction == "rtl"
