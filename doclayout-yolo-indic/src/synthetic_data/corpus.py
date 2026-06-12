"""
Text corpus: provides sample words per language for filling layout regions.

Priority:
  1. If data/text_corpus/<lang>_words.txt exists, sample real words from it.
  2. Otherwise synthesise plausible words from the script's Unicode ranges.

Synthetic words are adequate for *layout* pretraining (the detector learns
region geometry, not semantics) but for the released dataset and any
linguistic claims you should drop in a real corpus (e.g. AI4Bharat IndicCorp).
"""

import random
from typing import Dict, List

from src.logger import get_logger
from src.synthetic_data.indic_typography import REGISTRY, get_spec
from src.utils.paths import get_corpus_dir

logger = get_logger(__name__)

# lang -> filename stem used in data/text_corpus/
_FILE_STEM = {
    "Hindi": "hindi", "Bengali": "bengali", "Tamil": "tamil",
    "Telugu": "telugu", "Kannada": "kannada", "Malayalam": "malayalam",
    "Gujarati": "gujarati", "Punjabi": "punjabi", "Odia": "odia",
    "Urdu": "urdu", "English": "english",
}


def _codepoints(language: str) -> List[str]:
    chars = []
    for lo, hi in get_spec(language).unicode_ranges:
        chars.extend(chr(c) for c in range(lo, hi + 1))
    return chars


def _synth_word(language: str, rng: random.Random) -> str:
    chars = _codepoints(language)
    if not chars:
        return "x"
    return "".join(rng.choice(chars) for _ in range(rng.randint(2, 7)))


class Corpus:
    """Loads/synthesises words and samples blocks of text on demand."""

    def __init__(self, seed: int = 0, synth_vocab: int = 4000):
        self.rng = random.Random(seed)
        self.words: Dict[str, List[str]] = {}
        corpus_dir = get_corpus_dir()
        for lang in REGISTRY:
            f = corpus_dir / f"{_FILE_STEM[lang]}_words.txt"
            if f.exists():
                ws = [w.strip() for w in f.read_text(encoding="utf-8").splitlines()
                      if w.strip()]
                if ws:
                    self.words[lang] = ws
                    logger.info(f"{lang}: loaded {len(ws)} real words")
                    continue
            # synthesise a vocabulary once, then sample from it
            self.words[lang] = [_synth_word(lang, self.rng)
                                for _ in range(synth_vocab)]
        logger.info("Corpus ready (real where available, else synthetic)")

    def sample_text(self, language: str, num_words: int) -> str:
        pool = self.words[language]
        return " ".join(self.rng.choice(pool) for _ in range(num_words))
