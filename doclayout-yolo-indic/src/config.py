"""
Global configuration for DocLayout-YOLO-Indic.

All paths, the script registry, class mapping, and Phase 2 hyperparameters
live here so every module imports a single source of truth.

Example:
    >>> from src.config import COCO_CLASSES, INDIC_SCRIPTS, FONTS_DIR
    >>> COCO_CLASSES[1]
    'headline'
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Phase indicator (1=Outline, 2=Synthetic, 3=SelfTraining, 4=Eval, 5=Submit)
# ---------------------------------------------------------------------------
CURRENT_PHASE = 2

# ---------------------------------------------------------------------------
# Paths (repo-root relative; resolved in src/utils/paths.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"
FONTS_DIR = DATA_DIR / "fonts"
TEXT_CORPUS_DIR = DATA_DIR / "text_corpus"

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
NUM_SYNTHETIC_DOCS = 150_000
SYNTHETIC_IMG_SIZE = (1280, 1280)          # (width, height)
SYNTHETIC_FONT_SIZES = [16, 18, 20, 22, 24, 28]
MIXED_SCRIPT_PROB = 0.20                   # 20% of pages mix an Indic script + Latin
PAGE_BG = 255                              # white background (grayscale value)
PAGE_FG = 30                               # near-black text

# ---------------------------------------------------------------------------
# Script registry
#   key   -> human language name used throughout the project
#   These map to fonts + HarfBuzz script tags in indic_typography.py.
# Sampling weights are rough IndicDLP-style frequencies (sum need not be 1;
# they are normalised at sample time). Tune once IndicDLP stats are in.
# ---------------------------------------------------------------------------
INDIC_SCRIPTS = [
    "Hindi", "Bengali", "Tamil", "Telugu", "Kannada",
    "Malayalam", "Gujarati", "Punjabi", "Odia", "Urdu",
]

SCRIPT_SAMPLE_WEIGHTS = {
    "Hindi": 0.28, "Bengali": 0.16, "Tamil": 0.12, "Telugu": 0.11,
    "Kannada": 0.07, "Malayalam": 0.06, "Gujarati": 0.06,
    "Punjabi": 0.05, "Odia": 0.04, "Urdu": 0.05,
}

# Latin is used only for the mixed-script secondary language.
LATIN_LABEL = "English"

# ---------------------------------------------------------------------------
# COCO class mapping (matches PROJECT_CONTEXT.md)
# ---------------------------------------------------------------------------
COCO_CLASSES = {
    0: "text_body",
    1: "headline",
    2: "table",
    3: "figure",
    4: "caption",
    5: "advertisement",
    6: "sidebar",
    7: "pull_quote",
    8: "decorative_frame",
}
NUM_CLASSES = len(COCO_CLASSES)
CLASS_NAME_TO_ID = {v: k for k, v in COCO_CLASSES.items()}

# ---------------------------------------------------------------------------
# Pretraining (Step 7 — runs on GPU box)
# ---------------------------------------------------------------------------
BATCH_SIZE = 12                            # fits comfortably at imgsz 1024 on 40GB
LEARNING_RATE = 1e-3
NUM_EPOCHS = 15                            # pretraining converges well before 30
IMG_RES = 1024                             # pretrain at 1024 (3x faster than 1280);
                                           # fine-tune at 1280 in Phase 4 for final detail
DEVICE = "cuda"                            # falls back to "cpu" if unavailable
DETECTION_LOSS_WEIGHT = 1.0
SCRIPT_CLASSIFICATION_LOSS_WEIGHT = 0.1    # auxiliary script head (Angle C)

# Auxiliary script-classification head label space (9-way: 8 Indic groups + Latin).
# Devanagari covers Hindi; scripts are grouped by writing system, not language.
SCRIPT_HEAD_LABELS = [
    "Devanagari", "Bengali", "Tamil", "Telugu", "Kannada",
    "Malayalam", "Gujarati", "Gurmukhi", "Odia", "Arabic", "Latin",
]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 1337
