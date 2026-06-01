# DocLayout-YOLO-Indic Research Project Skill

**Name:** doclayout-yolo-indic-research

**Version:** 1.0

**Author:** Student working on Indic Document Layout Detection

**Updated:** May 2026

---

## Overview

This skill provides a standardized structure and patterns for implementing the DocLayout-YOLO-Indic project — extending real-time layout detection to Indic scripts.

**Current Phase:** Phase 2 (Synthetic Data Generation + Pretraining)

**Project Duration:** May 5 – Aug 2, 2026 (12 weeks)

**Repository:** [Your GitHub repo URL]

---

## Google Colab Adaptation

All training runs on **Google Colab Pro** (500 compute units). Files persist on **Google Drive**.

### Path Mapping (Local → Colab)

| SKILL docs reference | Actual Colab path |
|---|---|
| `doclayout-yolo-indic/` | `/content/drive/MyDrive/doclayout-yolo-indic/` |
| `src/` | `/content/drive/MyDrive/doclayout-yolo-indic/src/` |
| `data/` | `/content/drive/MyDrive/doclayout-yolo-indic/data/` |
| `output/` | `/content/drive/MyDrive/doclayout-yolo-indic/output/` |

### Required: Start of Every Colab Session

```python
from google.colab import drive
drive.mount('/content/drive')

import sys
from pathlib import Path

PROJECT_ROOT = Path('/content/drive/MyDrive/doclayout-yolo-indic')
sys.path.insert(0, str(PROJECT_ROOT))

!pip install ultralytics uharfbuzz pillow numpy opencv-python tqdm matplotlib pycocotools -q
```

### Writing Python Modules to Drive

Use `%%writefile` to create persistent source files:
```python
%%writefile /content/drive/MyDrive/doclayout-yolo-indic/src/config.py
from pathlib import Path
PROJECT_ROOT = Path('/content/drive/MyDrive/doclayout-yolo-indic')
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
```

### Config.py for Colab (use instead of the local version)

```python
import os
from pathlib import Path

if os.path.exists('/content/drive/MyDrive'):
    PROJECT_ROOT = Path('/content/drive/MyDrive/doclayout-yolo-indic')
else:
    PROJECT_ROOT = Path('.')  # Local dev fallback

DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
DEVICE = "cuda"  # Colab provides CUDA GPU
BATCH_SIZE = 16  # Reduced from 32 for Colab GPU memory
NUM_SYNTHETIC_DOCS = 50000  # Reduced from 150K for Colab storage
```

### Storage Management (100 GB Drive limit)

| Phase | Data | Size | Action |
|---|---|---|---|
| Phase 1 | D4LA + DocLayNet | ~40 GB | Delete after Phase 1 ✅ |
| Phase 2 | Synthetic images | ~2.5 GB | Keep until pretraining confirmed |
| Phase 3 | IndicDLP + BaDLAD | ~65 GB | Download when needed, delete after |
| All phases | Checkpoints | ~0.5 GB each | Keep only best |

### Off-Peak Training (IMPORTANT — saves compute units)

- Run training **only 8 PM – 8 AM UTC**
- Off-peak costs ~5 units/hour vs ~10 units/hour peak
- Always run keep-alive in a separate tab during long jobs

---

## When to Use This Skill

Trigger this skill whenever:
- Building code for the DocLayout-YOLO-Indic project
- Setting up new modules or scripts
- Establishing patterns for data handling, training, or evaluation
- Working on Phases 2-5 of the project

---

## Directory Structure

```
doclayout-yolo-indic/
│
├── README.md                          # Project overview
├── PROJECT_CONTEXT.md                 # This full context (copy into Claude Code sessions)
├── requirements.txt                   # Python dependencies
├── setup.py                           # Installation script
├── .gitignore                         # Exclude large files
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── config.py                      # Global configuration
│   ├── logger.py                      # Logging setup
│   │
│   ├── synthetic_data/                # Phase 2: Synthetic data generation
│   │   ├── __init__.py
│   │   ├── font_setup.py              # Download and verify fonts
│   │   ├── templates.py               # Layout templates
│   │   ├── text_renderer.py           # HarfBuzz text shaping
│   │   ├── indic_typography.py        # Script-specific rules
│   │   ├── generator.py               # Main synthetic document generator
│   │   ├── coco_formatter.py          # COCO annotation format
│   │   └── quality_inspector.py       # Visual inspection tools
│   │
│   ├── pretraining/                   # Phase 2: Pretraining on synthetic data
│   │   ├── __init__.py
│   │   ├── train_synthetic.py         # Pretraining script
│   │   ├── config.py                  # Hyperparameters
│   │   └── callbacks.py               # Logging, checkpointing
│   │
│   ├── finetuning/                    # Phase 3: Fine-tuning on IndicDLP
│   │   ├── __init__.py
│   │   ├── train_finetuning.py        # Fine-tuning script
│   │   ├── self_training.py           # Semi-supervised learning
│   │   └── ablation.py                # Ablation study utilities
│   │
│   ├── evaluation/                    # Phase 4: Evaluation
│   │   ├── __init__.py
│   │   ├── eval_metrics.py            # mAP, per-script metrics
│   │   ├── cross_domain.py            # Evaluation on multiple datasets
│   │   └── failure_analysis.py        # Qualitative analysis
│   │
│   └── utils/                         # Shared utilities
│       ├── __init__.py
│       ├── paths.py                   # Path management
│       ├── data_loader.py             # Data loading utilities
│       └── visualize.py               # Visualization tools
│
├── data/                              # Data directory
│   ├── fonts/                         # Noto Sans Indic fonts
│   │   ├── Noto_Sans_Devanagari.ttf
│   │   ├── Noto_Sans_Bengali.ttf
│   │   ├── Noto_Sans_Tamil.ttf
│   │   ├── Noto_Sans_Telugu.ttf
│   │   ├── Noto_Sans_Kannada.ttf
│   │   ├── Noto_Sans_Malayalam.ttf
│   │   ├── Noto_Sans_Gujarati.ttf
│   │   ├── Noto_Sans_Gurmukhi.ttf
│   │   ├── Noto_Sans_Oriya.ttf
│   │   └── Noto_Naskh_Arabic.ttf
│   │
│   ├── text_corpus/                   # Word lists for each language
│   │   ├── hindi_words.txt
│   │   ├── bengali_words.txt
│   │   ├── tamil_words.txt
│   │   ├── telugu_words.txt
│   │   ├── kannada_words.txt
│   │   ├── malayalam_words.txt
│   │   ├── gujarati_words.txt
│   │   ├── punjabi_words.txt
│   │   ├── odia_words.txt
│   │   └── urdu_words.txt
│   │
│   └── raw/                           # Downloaded datasets (NOT in repo)
│       ├── IndicDLP/                  # Downloaded from AIKosh
│       │   ├── images/
│       │   ├── annotations/
│       │   └── metadata.json
│       │
│       └── BaDLAD/                    # Downloaded from GitHub
│           ├── labeled/
│           ├── unlabeled/
│           └── metadata.json
│
├── output/                            # All generated outputs (NOT in repo)
│   ├── synthetic/                     # Phase 2: Generated synthetic data
│   │   ├── images/                    # 150,000 synthetic document images
│   │   ├── annotations/               # COCO format annotations
│   │   ├── train_manifest.json
│   │   └── val_manifest.json
│   │
│   ├── quality_inspection/            # Phase 2: Sample images for inspection
│   │   ├── samples/                   # 500 random samples for manual review
│   │   └── inspection_report.md
│   │
│   ├── checkpoints/                   # Saved model weights
│   │   ├── doclayout_yolo_indic_pretrained.pt     # Phase 2 output
│   │   ├── doclayout_yolo_indic_finetuned.pt      # Phase 3 output
│   │   └── doclayout_yolo_indic_final.pt          # Phase 5 output
│   │
│   ├── logs/                          # Training and evaluation logs
│   │   ├── pretraining_metrics.csv
│   │   ├── finetuning_metrics.csv
│   │   ├── evaluation_results.json
│   │   └── training_curves.png
│   │
│   ├── analysis/                      # Phase 4: Analysis outputs
│   │   ├── ablation_results.csv
│   │   ├── failure_cases.md
│   │   └── visualizations/
│   │
│   └── paper/                         # Phase 4-5: Paper drafts
│       ├── draft_v1.md
│       └── figures/

├── tests/                             # Unit tests
│   ├── __init__.py
│   ├── test_text_renderer.py          # Test HarfBuzz integration
│   ├── test_coco_format.py            # Validate COCO annotations
│   └── test_synthetic_generator.py    # Test document generation
│
└── notebooks/                         # Jupyter notebooks (exploratory)
    ├── 01_explore_fonts.ipynb
    ├── 02_test_harfbuzz.ipynb
    └── 03_visualize_synthetic.ipynb
```

---

## Code Patterns & Conventions

### 1. Module Structure

Every module should follow this pattern:

```python
"""
Module docstring explaining what this module does.

Example:
    >>> from src.synthetic_data import generator
    >>> gen = generator.SyntheticDocumentGenerator(num_docs=100)
    >>> gen.generate()
"""

import logging
from pathlib import Path
from src.config import *
from src.logger import get_logger

logger = get_logger(__name__)

class ModuleName:
    """Class docstring with attributes and methods."""

    def __init__(self, config=None):
        self.config = config or CONFIG
        logger.info(f"Initialized {self.__class__.__name__}")

    def method_name(self):
        """Method docstring."""
        pass

def standalone_function():
    """Function docstring."""
    pass

if __name__ == "__main__":
    logger.info("Running as main script")
    # Example usage here
```

### 2. Configuration Management

Create a `src/config.py` with all hyperparameters:

```python
# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
FONTS_DIR = DATA_DIR / "fonts"
TEXT_CORPUS_DIR = DATA_DIR / "text_corpus"

# Synthetic data generation
NUM_SYNTHETIC_DOCS = 150000
SYNTHETIC_IMG_SIZE = (1280, 1280)
SYNTHETIC_FONT_SIZES = [12, 14, 16, 18, 20, 24]

# Scripts to support
INDIC_SCRIPTS = [
    "Hindi", "Bengali", "Tamil", "Telugu",
    "Kannada", "Malayalam", "Gujarati", "Punjabi", "Odia", "Urdu"
]

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
DEVICE = "cuda"  # or "cpu"

# Pretraining task weights
DETECTION_LOSS_WEIGHT = 1.0
SCRIPT_CLASSIFICATION_LOSS_WEIGHT = 0.1

# COCO classes
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
```

### 3. Logging

Use the centralized logger:

```python
from src.logger import get_logger

logger = get_logger(__name__)

logger.info("Starting synthetic document generation")
logger.warning(f"Font {font_path} not found, using fallback")
logger.error(f"Failed to render script {script}: {error_msg}")
```

### 4. Path Management

Always use `Path` objects from `src.utils.paths`:

```python
from src.utils.paths import get_output_dir, get_checkpoint_path

# Don't do this:
# img_path = "output/synthetic/images/" + str(doc_id) + ".png"

# Do this:
output_dir = get_output_dir("synthetic")
img_path = output_dir / "images" / f"{doc_id}.png"
img_path.parent.mkdir(parents=True, exist_ok=True)
img_path.write_bytes(img_data)
```

### 5. Data Loading

Centralize data loading in `src.utils.data_loader`:

```python
from src.utils.data_loader import load_text_corpus, load_fonts

# Load corpus for all languages
corpus = load_text_corpus()  # Returns dict: {"Hindi": [...], "Tamil": [...]}

# Load all fonts
fonts = load_fonts()  # Returns dict: {"Hindi": font_obj, "Tamil": font_obj}
```

### 6. Saving Checkpoints

Always save with metadata:

```python
import json
from pathlib import Path

checkpoint = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "epoch": epoch,
    "loss": loss,
    "timestamp": datetime.now().isoformat(),
    "config": CONFIG.__dict__,  # Save config for reproducibility
}

checkpoint_path = Path("output/checkpoints") / f"checkpoint_epoch_{epoch}.pt"
torch.save(checkpoint, checkpoint_path)
logger.info(f"Checkpoint saved: {checkpoint_path}")
```

### 7. Progress Tracking

Use tqdm for progress bars:

```python
from tqdm import tqdm
import logging

# Suppress tqdm output when logging is active
for doc_id in tqdm(range(NUM_DOCS), desc="Generating synthetic docs", disable=logger.disabled):
    # Your loop here
    if doc_id % 10000 == 0:
        logger.info(f"Progress: {doc_id}/{NUM_DOCS}")
```

---

## Phase-Specific Code Patterns

### Phase 2: Synthetic Data Generation

**Key files to create:**
- `src/synthetic_data/generator.py` — Main loop (150K documents)
- `src/synthetic_data/text_renderer.py` — HarfBuzz integration
- `src/synthetic_data/coco_formatter.py` — COCO output
- `src/pretraining/train_synthetic.py` — Pretraining loop

**Key patterns:**

```python
# In generator.py
class SyntheticDocumentGenerator:
    def __init__(self, num_docs=150000, config=None):
        self.num_docs = num_docs
        self.templates = load_templates()
        self.fonts = load_fonts()
        self.corpus = load_text_corpus()

    def generate_all(self):
        """Generate all 150K documents."""
        for doc_id in tqdm(range(self.num_docs)):
            img, annotations = self.generate_one(doc_id)
            self.save(img, annotations, doc_id)

            if (doc_id + 1) % 10000 == 0:
                logger.info(f"Generated {doc_id + 1}/{self.num_docs}")

    def generate_one(self, doc_id):
        """Generate a single synthetic document."""
        # Pick random script and template
        # Render text using HarfBuzz
        # Place on page
        # Create COCO annotations
        pass

    def save(self, img, annotations, doc_id):
        """Save image and annotations."""
        pass

# In text_renderer.py
class IndricTextRenderer:
    def shape_and_render(self, text, script, font_path, font_size):
        """
        Shape text using HarfBuzz, then render to image.
        Handles: Devanagari shirorekha, Tamil conjuncts, etc.
        """
        shaped_glyphs = self.shape_with_harfbuzz(text, script, font_path)
        img = self.render_glyphs_to_image(shaped_glyphs, font_size)
        return img
```

### Phase 3: Fine-tuning & Self-Training

**Key files to create:**
- `src/finetuning/train_finetuning.py` — Fine-tune on IndicDLP
- `src/finetuning/self_training.py` — Pseudo-labeling loop
- `src/finetuning/ablation.py` — Run ablation experiments

**Key pattern:**

```python
# In ablation.py
class AblationExperiment:
    """Run systematic ablations to measure component contributions."""

    EXPERIMENTS = [
        {"name": "baseline", "use_synthetic": False, "use_script_head": False, "use_selftraining": False},
        {"name": "+synthetic", "use_synthetic": True, "use_script_head": False, "use_selftraining": False},
        {"name": "+script_head", "use_synthetic": True, "use_script_head": True, "use_selftraining": False},
        {"name": "+selftraining", "use_synthetic": True, "use_script_head": True, "use_selftraining": True},
    ]

    def run_all(self):
        results = {}
        for exp in self.EXPERIMENTS:
            logger.info(f"Running experiment: {exp['name']}")
            metrics = self.run_experiment(exp)
            results[exp['name']] = metrics

        self.save_results(results)
        return results
```

### Phase 4: Evaluation

**Key files to create:**
- `src/evaluation/eval_metrics.py` — Compute mAP, per-script metrics
- `src/evaluation/failure_analysis.py` — Analyze failure cases

**Key pattern:**

```python
# In eval_metrics.py
def evaluate(model, dataloader, dataset_name="IndicDLP"):
    """
    Evaluate model on a dataset.
    Returns: mAP@[0.5:0.95], per-script mAP, per-class mAP
    """
    all_predictions = []
    all_ground_truth = []

    for images, targets, script_ids in dataloader:
        preds = model(images)
        all_predictions.extend(preds)
        all_ground_truth.extend(targets)

    # Compute mAP
    metrics = compute_coco_metrics(all_predictions, all_ground_truth)

    # Compute per-script mAP
    for script in INDIC_SCRIPTS:
        script_mask = [s == script for s in script_ids]
        script_metrics = compute_coco_metrics(
            [p for p, m in zip(all_predictions, script_mask) if m],
            [g for g, m in zip(all_ground_truth, script_mask) if m]
        )
        metrics[f"mAP_{script}"] = script_metrics["mAP"]

    return metrics
```

---

## Important Implementation Notes

### HarfBuzz Integration

Always use HarfBuzz for Indic text rendering. It handles:
- Devanagari shirorekha (horizontal line through text)
- Bengali/Tamil/Telugu conjunct stacking
- Matra (vowel mark) positioning
- Right-to-left flow (Urdu/Arabic)

```python
import uharfbuzz as hb

def shape_text(text, script, font_path):
    with open(font_path, 'rb') as f:
        font_data = f.read()

    face = hb.Face(font_data)
    font = hb.Font(face)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()

    hb.shape(font, buf)

    infos = buf.glyph_infos
    positions = buf.glyph_positions
    return infos, positions
```

### COCO Format Validation

Always validate COCO annotations:

```python
def validate_coco_annotation(annotation):
    """Ensure bbox is valid."""
    bbox = annotation["bbox"]
    assert len(bbox) == 4, "bbox must be [x, y, w, h]"
    assert bbox[2] > 0 and bbox[3] > 0, "width and height must be positive"
    assert 0 <= annotation["category_id"] < len(COCO_CLASSES), "invalid category"
    return True
```

### Handling Large Datasets

For 150K synthetic documents:
- Don't load all into memory at once
- Use generators/data loaders
- Save progress periodically
- Log every 10K documents

```python
def generate_in_batches(batch_size=1000):
    """Generate synthetic data in batches for memory efficiency."""
    for batch_start in range(0, NUM_DOCS, batch_size):
        batch_end = min(batch_start + batch_size, NUM_DOCS)
        batch_docs = []

        for doc_id in range(batch_start, batch_end):
            img, annotations = generate_one(doc_id)
            batch_docs.append((img, annotations))

        yield batch_docs
```

---

## Testing & Validation Checklist

Before moving to next phase, ensure:

- [ ] All fonts download successfully
- [ ] Text corpus loads without errors
- [ ] HarfBuzz renders text correctly (test on small samples)
- [ ] Synthetic documents are generated (inspect first 10)
- [ ] COCO annotations are valid (validate a few)
- [ ] Quality inspection completed (500 samples reviewed)
- [ ] Dataset can be loaded by training code
- [ ] Pretraining runs without errors
- [ ] Loss decreases over epochs
- [ ] Checkpoint saves successfully

---

## Debugging Common Issues

### Issue: HarfBuzz Text Overlaps
**Cause:** Font size too large for glyph positions
**Fix:** Reduce font size or adjust line spacing

### Issue: Synthetic Images Look Unrealistic
**Cause:** Layout templates don't match real documents
**Fix:** Iterate templates based on quality inspection feedback

### Issue: COCO Annotation Mismatches Image
**Cause:** Coordinate system mismatch (PIL vs numpy)
**Fix:** Always save annotations BEFORE final image crop/resize

### Issue: RTL (Urdu) Rendering Breaks
**Cause:** Horizontal flip applied incorrectly
**Fix:** Flip image AND annotations consistently

### Issue: Pretraining Loss Doesn't Decrease
**Cause:** Learning rate too high, or bad data
**Fix:** Reduce LR, verify first batch renders correctly

---

## Reproducibility & Version Control

Every run should be reproducible:

```python
# Save all hyperparameters
import json
from datetime import datetime

run_config = {
    "timestamp": datetime.now().isoformat(),
    "phase": 2,
    "num_synthetic_docs": 150000,
    "config": CONFIG.__dict__,
    "git_commit": get_git_commit_hash(),
}

(Path("output") / "run_config.json").write_text(json.dumps(run_config, indent=2))
```

---

## Dependencies

Create `requirements.txt`:

```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
uharfbuzz>=0.25.0
pillow>=10.0.0
numpy>=1.24.0
opencv-python>=4.8.0
tqdm>=4.66.0
matplotlib>=3.8.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Phase Indicators

Use these to track which phase you're in. Claude Code will know context:

```python
# In src/config.py
CURRENT_PHASE = 2  # 1=Outline, 2=Synthetic, 3=SelfTraining, 4=Eval, 5=Submit

# In any module
if CURRENT_PHASE >= 2:
    # Synthetic data generation code
    pass

if CURRENT_PHASE >= 3:
    # Fine-tuning code
    pass
```

---

## Git Workflow

Commit frequently with clear messages:

```bash
git add src/synthetic_data/generator.py
git commit -m "feat(phase2): implement synthetic document generator with HarfBuzz"

git add output/synthetic/
git commit -m "chore(phase2): add 150K synthetic documents"

git add src/pretraining/train_synthetic.py
git commit -m "feat(phase2): add pretraining loop on synthetic data"
```

---

## Running Claude Code Sessions

When using Claude Code:

1. **Copy PROJECT_CONTEXT.md** into the session
2. **Reference this SKILL file** for structure and patterns
3. **Specify the phase** you're working on
4. **Provide exact file paths** (use paths from this structure)
5. **Request specific outputs** (e.g., "create `src/synthetic_data/generator.py`")

Example prompt to Claude Code:

```
I'm in Phase 2 of the DocLayout-YOLO-Indic project.

Current task: Create src/synthetic_data/text_renderer.py

Requirements:
- Use HarfBuzz to shape Indic text
- Support all 10 scripts (Hindi, Bengali, Tamil, etc.)
- Return glyph infos and positions
- Follow the code patterns in the DOCLAYOUT_YOLO_INDIC_SKILL.md file

Project structure: [paste directory structure from skill]

Create the module with docstrings and example usage.
```

---

## Additional Resources

- **YOLO Docs:** https://docs.ultralytics.com/
- **HarfBuzz Python:** https://github.com/uharfbuzz/uharfbuzz
- **Noto Fonts:** https://www.google.com/get/noto/
- **COCO Dataset Format:** https://cocodataset.org/#format-data
- **Indic Scripts:** https://en.wikipedia.org/wiki/Indic_scripts
- **Base Paper:** Zhao et al., "DocLayout-YOLO," NeurIPS 2024

---

**Last Updated:** May 2026
**Project Owner:** [Your Name]
**Status:** Phase 2 - Synthetic Data Generation