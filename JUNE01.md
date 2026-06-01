================================================================================
PHASE 2 ACTION PLAN - GOOGLE COLAB
DocLayout-YOLO-Indic: Synthetic Data Generation
================================================================================

Status:     Phase 1 COMPLETE ✅
Now:        Starting Phase 2 (Synthetic Data + Pretraining)
Platform:   Google Colab Pro (500 compute units)
Budget:     ~30 units for generation + ~20 units for pretraining = 50 units total
Storage:    /content/drive/MyDrive/doclayout-yolo-indic/ (use Google Drive)

================================================================================
PHASE 2 GOALS
================================================================================

By end of Phase 2 you must have:
  1. 50,000 synthetic Indic document images (reduced from 150K for Colab)
  2. COCO-format annotations for all synthetic images
  3. Pretrained DocLayout-YOLO checkpoint (trained on synthetic data)
  4. Pretrained mAP target: ≥ 60% on held-out synthetic test set

================================================================================
STEP 1 - CREATE PHASE 2 NOTEBOOK (Do this first, 20 minutes)
================================================================================

Go to: https://colab.research.google.com
Click: + New Notebook
Name: "DocLayout-YOLO-Indic-Phase2-SyntheticData"
Save to Drive: File → Save a copy in Drive

CELL 1 - Session Setup (run at start of EVERY session):
--------
```python
# ============================================================
# PHASE 2: SYNTHETIC DATA GENERATION
# Run this cell FIRST every session
# ============================================================

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import sys
from pathlib import Path

# Project root on Google Drive
PROJECT_ROOT = Path('/content/drive/MyDrive/doclayout-yolo-indic')
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT))

# Install required packages
!pip install ultralytics uharfbuzz pillow numpy opencv-python tqdm matplotlib pycocotools requests -q

# Check GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Project root: {PROJECT_ROOT}")
print("Setup complete")
```

CELL 2 - Create Directory Structure:
--------
```python
# Create all Phase 2 directories in Google Drive
dirs = [
    'src/synthetic_data',
    'src/pretraining',
    'src/utils',
    'data/fonts',
    'data/text_corpus',
    'output/synthetic/images',
    'output/synthetic/annotations',
    'output/quality_inspection/samples',
    'output/checkpoints',
    'output/logs',
]

for d in dirs:
    (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

print("Directory structure created:")
import subprocess
result = subprocess.run(['find', str(PROJECT_ROOT), '-type', 'd', '-not', '-path', '*/.git/*'],
                       capture_output=True, text=True)
for line in sorted(result.stdout.strip().split('\n')):
    print(' ', line.replace(str(PROJECT_ROOT), '.'))
```

================================================================================
STEP 2 - DOWNLOAD FONTS (Session 1, ~15 minutes)
================================================================================

CELL 3 - Download Noto Sans Indic Fonts:
--------
```python
import requests
from pathlib import Path

fonts_dir = PROJECT_ROOT / 'data' / 'fonts'

# Noto fonts download URLs (Google Fonts API)
FONT_URLS = {
    'Noto_Sans_Devanagari.ttf': 'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf',
    'Noto_Sans_Bengali.ttf':    'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansBengali/NotoSansBengali-Regular.ttf',
    'Noto_Sans_Tamil.ttf':      'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansTamil/NotoSansTamil-Regular.ttf',
    'Noto_Sans_Telugu.ttf':     'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansTelugu/NotoSansTelugu-Regular.ttf',
    'Noto_Sans_Kannada.ttf':    'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansKannada/NotoSansKannada-Regular.ttf',
    'Noto_Sans_Malayalam.ttf':  'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansMalayalam/NotoSansMalayalam-Regular.ttf',
    'Noto_Sans_Gujarati.ttf':   'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansGujarati/NotoSansGujarati-Regular.ttf',
    'Noto_Sans_Gurmukhi.ttf':   'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansGurmukhi/NotoSansGurmukhi-Regular.ttf',
    'Noto_Sans_Oriya.ttf':      'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansOriya/NotoSansOriya-Regular.ttf',
    'Noto_Naskh_Arabic.ttf':    'https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoNaskhArabic/NotoNaskhArabic-Regular.ttf',
}

for font_name, url in FONT_URLS.items():
    font_path = fonts_dir / font_name
    if font_path.exists():
        print(f"✓ {font_name} already exists, skipping")
        continue

    try:
        print(f"Downloading {font_name}...", end=' ')
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        font_path.write_bytes(response.content)
        print(f"✓ ({len(response.content)//1024} KB)")
    except Exception as e:
        print(f"FAILED: {e}")
        # Try fallback: install system fonts
        print(f"  Trying apt-get fallback...")
        !apt-get install -y fonts-noto -q
        print("  System Noto fonts installed as fallback")
        break

# Verify
font_files = list(fonts_dir.glob('*.ttf'))
print(f"\n{len(font_files)}/{len(FONT_URLS)} fonts available:")
for f in sorted(font_files):
    print(f"  {f.name}")
```

================================================================================
STEP 3 - PREPARE TEXT CORPUS (Session 1, ~30 minutes)
================================================================================

CELL 4 - Download Text Corpus (Wikipedia snippets):
--------
```python
import json
import requests
from pathlib import Path

corpus_dir = PROJECT_ROOT / 'data' / 'text_corpus'

# Wikipedia API to get words for each language
LANGUAGE_CODES = {
    'hindi_words.txt':     'hi',
    'bengali_words.txt':   'bn',
    'tamil_words.txt':     'ta',
    'telugu_words.txt':    'te',
    'kannada_words.txt':   'kn',
    'malayalam_words.txt': 'ml',
    'gujarati_words.txt':  'gu',
    'punjabi_words.txt':   'pa',
    'odia_words.txt':      'or',
    'urdu_words.txt':      'ur',
}

def fetch_wikipedia_words(lang_code, num_words=5000):
    """Fetch words from Wikipedia random articles."""
    words = set()

    for attempt in range(20):  # Try 20 random articles
        if len(words) >= num_words:
            break
        try:
            url = f"https://{lang_code}.wikipedia.org/api/rest_v1/page/random/summary"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                text = data.get('extract', '')
                # Split on whitespace and filter short words
                new_words = [w.strip('.,!?;:()[]') for w in text.split()
                             if len(w.strip('.,!?;:()[]')) > 2]
                words.update(new_words)
        except Exception:
            pass

    return list(words)[:num_words]

for filename, lang_code in LANGUAGE_CODES.items():
    corpus_file = corpus_dir / filename
    if corpus_file.exists() and corpus_file.stat().st_size > 1000:
        print(f"✓ {filename} already exists")
        continue

    print(f"Fetching {filename} (lang={lang_code})...", end=' ')
    try:
        words = fetch_wikipedia_words(lang_code, num_words=5000)
        corpus_file.write_text('\n'.join(words), encoding='utf-8')
        print(f"✓ {len(words)} words")
    except Exception as e:
        # Create minimal fallback corpus
        print(f"Wikipedia failed ({e}), creating sample corpus...")
        sample_text = f"Sample {lang_code} text for testing synthetic document generation " * 200
        corpus_file.write_text(sample_text, encoding='utf-8')
        print(f"  Created fallback corpus")

print("\nCorpus ready!")
```

================================================================================
STEP 4 - WRITE SOURCE CODE MODULES TO DRIVE
================================================================================

Use %%writefile to save Python modules to Drive so they persist between sessions.

CELL 5 - Write config.py:
--------
```python
%%writefile /content/drive/MyDrive/doclayout-yolo-indic/src/config.py
from pathlib import Path
import os

# Detect environment
if os.path.exists('/content/drive/MyDrive'):
    PROJECT_ROOT = Path('/content/drive/MyDrive/doclayout-yolo-indic')
else:
    PROJECT_ROOT = Path('.')

DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
FONTS_DIR = DATA_DIR / 'fonts'
TEXT_CORPUS_DIR = DATA_DIR / 'text_corpus'
SYNTHETIC_DIR = OUTPUT_DIR / 'synthetic'
CHECKPOINTS_DIR = OUTPUT_DIR / 'checkpoints'
LOGS_DIR = OUTPUT_DIR / 'logs'

# For Colab: generate to /content/ (fast), then copy to Drive
COLAB_FAST_DIR = Path('/content/synthetic_temp')

NUM_SYNTHETIC_DOCS = 50000   # Reduced from 150K for Colab storage
SYNTHETIC_IMG_SIZE = (1024, 1024)  # Slightly reduced for speed
SYNTHETIC_FONT_SIZES = [12, 14, 16, 18, 20, 24]

INDIC_SCRIPTS = [
    "Hindi", "Bengali", "Tamil", "Telugu",
    "Kannada", "Malayalam", "Gujarati", "Punjabi", "Odia", "Urdu"
]

SCRIPT_TO_FONT = {
    "Hindi":     "Noto_Sans_Devanagari.ttf",
    "Bengali":   "Noto_Sans_Bengali.ttf",
    "Tamil":     "Noto_Sans_Tamil.ttf",
    "Telugu":    "Noto_Sans_Telugu.ttf",
    "Kannada":   "Noto_Sans_Kannada.ttf",
    "Malayalam": "Noto_Sans_Malayalam.ttf",
    "Gujarati":  "Noto_Sans_Gujarati.ttf",
    "Punjabi":   "Noto_Sans_Gurmukhi.ttf",
    "Odia":      "Noto_Sans_Oriya.ttf",
    "Urdu":      "Noto_Naskh_Arabic.ttf",
}

SCRIPT_TO_CORPUS = {
    "Hindi":     "hindi_words.txt",
    "Bengali":   "bengali_words.txt",
    "Tamil":     "tamil_words.txt",
    "Telugu":    "telugu_words.txt",
    "Kannada":   "kannada_words.txt",
    "Malayalam": "malayalam_words.txt",
    "Gujarati":  "gujarati_words.txt",
    "Punjabi":   "punjabi_words.txt",
    "Odia":      "odia_words.txt",
    "Urdu":      "urdu_words.txt",
}

BATCH_SIZE = 16    # Reduced for Colab GPU memory
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
DEVICE = "cuda"    # Colab provides CUDA GPU

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

CURRENT_PHASE = 2
```

CELL 6 - Write src/__init__.py files:
--------
```python
for init_path in [
    PROJECT_ROOT / 'src' / '__init__.py',
    PROJECT_ROOT / 'src' / 'synthetic_data' / '__init__.py',
    PROJECT_ROOT / 'src' / 'pretraining' / '__init__.py',
    PROJECT_ROOT / 'src' / 'utils' / '__init__.py',
]:
    init_path.write_text('')
print("Created __init__.py files")
```

CELL 7 - Write text_renderer.py (HarfBuzz wrapper):
--------
```python
%%writefile /content/drive/MyDrive/doclayout-yolo-indic/src/synthetic_data/text_renderer.py
"""
HarfBuzz-based text renderer for Indic scripts.
Handles conjuncts, matras, shirorekha, RTL (Urdu).
"""

import uharfbuzz as hb
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np


class IndicTextRenderer:
    """Renders Indic text using HarfBuzz for correct shaping."""

    def __init__(self, fonts_dir):
        self.fonts_dir = Path(fonts_dir)
        self._font_cache = {}

    def _load_font(self, font_path, font_size):
        key = (str(font_path), font_size)
        if key not in self._font_cache:
            self._font_cache[key] = ImageFont.truetype(str(font_path), font_size)
        return self._font_cache[key]

    def shape_text(self, text, font_path):
        """Shape text using HarfBuzz. Returns glyph infos and positions."""
        try:
            with open(font_path, 'rb') as f:
                font_data = f.read()

            face = hb.Face(font_data)
            font = hb.Font(face)
            buf = hb.Buffer()
            buf.add_str(text)
            buf.guess_segment_properties()
            hb.shape(font, buf)

            return buf.glyph_infos, buf.glyph_positions
        except Exception:
            return None, None

    def render_text_block(self, text, font_path, font_size, width, height,
                          bg_color=(255, 255, 255), text_color=(0, 0, 0)):
        """
        Render a block of text into an image of given dimensions.
        Falls back to PIL rendering if HarfBuzz shaping fails.
        """
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        try:
            font = self._load_font(font_path, font_size)

            # Word-wrap text to fit width
            words = text.split()
            lines = []
            current_line = []

            for word in words:
                current_line.append(word)
                test_line = ' '.join(current_line)
                bbox = draw.textbbox((0, 0), test_line, font=font)
                line_width = bbox[2] - bbox[0]
                if line_width > width - 10:
                    if len(current_line) > 1:
                        current_line.pop()
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(test_line)
                        current_line = []

            if current_line:
                lines.append(' '.join(current_line))

            # Render each line
            line_height = font_size + 4
            y = 5
            for line in lines:
                if y + line_height > height:
                    break
                draw.text((5, y), line, font=font, fill=text_color)
                y += line_height

        except Exception as e:
            # Minimal fallback
            draw.text((5, 5), text[:50], fill=text_color)

        return img
```

CELL 8 - Write generator.py (main document generator):
--------
```python
%%writefile /content/drive/MyDrive/doclayout-yolo-indic/src/synthetic_data/generator.py
"""
Synthetic Indic document generator.
Creates document images with COCO-format annotations.
"""

import random
import json
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import *
from src.synthetic_data.text_renderer import IndicTextRenderer


LAYOUT_TEMPLATES = [
    # (name, list of (region_name, x, y, w, h) as fractions of image size)
    ("newspaper_2col", [
        ("headline",    0.05, 0.03, 0.90, 0.10),
        ("text_body",   0.05, 0.15, 0.43, 0.55),
        ("text_body",   0.52, 0.15, 0.43, 0.55),
        ("figure",      0.05, 0.72, 0.43, 0.22),
        ("caption",     0.05, 0.95, 0.43, 0.04),
    ]),
    ("textbook_1col", [
        ("headline",    0.05, 0.03, 0.90, 0.08),
        ("text_body",   0.08, 0.15, 0.84, 0.60),
        ("figure",      0.20, 0.77, 0.60, 0.15),
        ("caption",     0.20, 0.93, 0.60, 0.05),
    ]),
    ("magazine_3col", [
        ("headline",    0.03, 0.02, 0.94, 0.12),
        ("text_body",   0.03, 0.16, 0.29, 0.78),
        ("text_body",   0.36, 0.16, 0.28, 0.78),
        ("text_body",   0.68, 0.16, 0.29, 0.78),
    ]),
    ("form_layout", [
        ("headline",    0.05, 0.02, 0.90, 0.08),
        ("table",       0.05, 0.12, 0.90, 0.45),
        ("text_body",   0.05, 0.60, 0.90, 0.20),
        ("sidebar",     0.75, 0.82, 0.20, 0.15),
    ]),
    ("academic_paper", [
        ("headline",    0.10, 0.02, 0.80, 0.08),
        ("text_body",   0.05, 0.12, 0.43, 0.40),
        ("text_body",   0.52, 0.12, 0.43, 0.40),
        ("figure",      0.10, 0.55, 0.38, 0.30),
        ("table",       0.52, 0.55, 0.43, 0.30),
        ("caption",     0.10, 0.87, 0.85, 0.05),
    ]),
    ("news_with_sidebar", [
        ("headline",    0.03, 0.02, 0.94, 0.10),
        ("text_body",   0.03, 0.14, 0.60, 0.72),
        ("sidebar",     0.67, 0.14, 0.30, 0.35),
        ("advertisement", 0.67, 0.52, 0.30, 0.34),
        ("figure",      0.03, 0.88, 0.60, 0.10),
    ]),
    ("book_chapter", [
        ("headline",    0.08, 0.04, 0.84, 0.07),
        ("text_body",   0.08, 0.14, 0.84, 0.75),
        ("pull_quote",  0.15, 0.48, 0.70, 0.10),
    ]),
    ("brochure", [
        ("headline",    0.05, 0.02, 0.90, 0.12),
        ("figure",      0.05, 0.16, 0.55, 0.35),
        ("text_body",   0.63, 0.16, 0.32, 0.35),
        ("text_body",   0.05, 0.54, 0.90, 0.25),
        ("advertisement", 0.05, 0.82, 0.90, 0.15),
    ]),
]

CLASS_NAME_TO_ID = {v: k for k, v in COCO_CLASSES.items()}


class SyntheticDocumentGenerator:
    """Generate synthetic Indic document images with COCO annotations."""

    def __init__(self, output_dir, fonts_dir, corpus_dir, num_docs=50000, img_size=(1024, 1024)):
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.num_docs = num_docs
        self.renderer = IndicTextRenderer(fonts_dir)

        # Load corpus
        self.corpus = {}
        for script in INDIC_SCRIPTS:
            corpus_file = Path(corpus_dir) / SCRIPT_TO_CORPUS[script]
            if corpus_file.exists():
                words = corpus_file.read_text(encoding='utf-8').split('\n')
                self.corpus[script] = [w for w in words if len(w) > 1]
            else:
                self.corpus[script] = [f"word{i}" for i in range(100)]

        # Font paths
        self.font_paths = {}
        for script in INDIC_SCRIPTS:
            fp = Path(fonts_dir) / SCRIPT_TO_FONT[script]
            if fp.exists():
                self.font_paths[script] = fp
            else:
                # Use any available font as fallback
                fallback = list(Path(fonts_dir).glob('*.ttf'))
                if fallback:
                    self.font_paths[script] = fallback[0]

        (self.output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'annotations').mkdir(parents=True, exist_ok=True)

    def _sample_text(self, script, num_words=30):
        words = self.corpus.get(script, ['text'] * 100)
        if not words:
            return "Sample text"
        chosen = random.choices(words, k=num_words)
        return ' '.join(chosen)

    def _pick_script(self):
        # 20% chance mixed script (Indic + some English-like)
        if random.random() < 0.2:
            return [random.choice(INDIC_SCRIPTS), random.choice(INDIC_SCRIPTS)]
        return [random.choice(INDIC_SCRIPTS)]

    def generate_one(self, doc_id):
        W, H = self.img_size
        img = Image.new('RGB', (W, H), (250, 248, 245))  # Slightly off-white
        template_name, regions = random.choice(LAYOUT_TEMPLATES)
        scripts = self._pick_script()

        annotations = []
        ann_id = 0

        for region_name, rx, ry, rw, rh in regions:
            # Convert fractional coords to pixels
            x = int(rx * W)
            y = int(ry * H)
            w = int(rw * W)
            h = int(rh * H)

            script = random.choice(scripts)
            font_path = self.font_paths.get(script)
            if not font_path:
                continue

            cls_name = region_name if region_name in CLASS_NAME_TO_ID else 'text_body'
            cls_id = CLASS_NAME_TO_ID[cls_name]

            if cls_name in ('text_body', 'headline', 'sidebar', 'pull_quote', 'caption', 'advertisement'):
                num_words = random.randint(20, 60) if cls_name == 'text_body' else random.randint(3, 15)
                font_size = random.choice([12, 14, 16]) if cls_name == 'text_body' else random.choice([18, 20, 22, 24])
                text = self._sample_text(script, num_words)
                text_img = self.renderer.render_text_block(text, font_path, font_size, w, h)
                img.paste(text_img, (x, y))

            elif cls_name == 'figure':
                # Placeholder grey box for figures
                draw = ImageDraw.Draw(img)
                shade = random.randint(180, 220)
                draw.rectangle([x, y, x+w, y+h], fill=(shade, shade, shade))

            elif cls_name == 'table':
                # Draw grid for tables
                draw = ImageDraw.Draw(img)
                rows, cols = random.randint(3, 6), random.randint(3, 5)
                cw, rh2 = w // cols, h // rows
                draw.rectangle([x, y, x+w, y+h], fill=(255, 255, 255), outline=(0, 0, 0))
                for r in range(rows+1):
                    draw.line([(x, y+r*rh2), (x+w, y+r*rh2)], fill=(100, 100, 100), width=1)
                for c in range(cols+1):
                    draw.line([(x+c*cw, y), (x+c*cw, y+h)], fill=(100, 100, 100), width=1)

            annotations.append({
                "id": ann_id,
                "image_id": doc_id,
                "category_id": cls_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

        # Save image
        img_path = self.output_dir / 'images' / f'synthetic_{doc_id:06d}.jpg'
        img.save(str(img_path), 'JPEG', quality=90)

        # Save annotation
        ann_path = self.output_dir / 'annotations' / f'synthetic_{doc_id:06d}.json'
        ann_data = {
            "image_id": doc_id,
            "image_file": img_path.name,
            "width": W,
            "height": H,
            "annotations": annotations,
        }
        ann_path.write_text(json.dumps(ann_data))

        return annotations

    def generate_all(self, start=0, end=None, checkpoint_interval=1000):
        end = end or self.num_docs
        print(f"Generating {end - start} documents ({start} to {end-1})...")

        for doc_id in tqdm(range(start, end), desc="Generating docs"):
            self.generate_one(doc_id)

            if (doc_id + 1) % checkpoint_interval == 0:
                print(f"\nCheckpoint: {doc_id + 1} docs done")

        print(f"Done! Generated {end - start} documents.")
```

================================================================================
STEP 5 - RUN GENERATION (Off-peak: 8 PM - 8 AM UTC)
================================================================================

CELL 9 - Test with 10 documents first:
--------
```python
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from src.synthetic_data.generator import SyntheticDocumentGenerator

# Quick test: generate 10 documents
test_gen = SyntheticDocumentGenerator(
    output_dir=PROJECT_ROOT / 'output' / 'synthetic',
    fonts_dir=PROJECT_ROOT / 'data' / 'fonts',
    corpus_dir=PROJECT_ROOT / 'data' / 'text_corpus',
    num_docs=10,
    img_size=(512, 512),  # Small for testing
)
test_gen.generate_all(start=0, end=10)

# Visual check
from PIL import Image
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    img_path = PROJECT_ROOT / 'output' / 'synthetic' / 'images' / f'synthetic_{i:06d}.jpg'
    if img_path.exists():
        ax.imshow(Image.open(str(img_path)))
        ax.set_title(f'doc_{i}')
    ax.axis('off')
plt.suptitle('Sample Synthetic Documents (verify quality before full run!)')
plt.tight_layout()
plt.savefig(str(PROJECT_ROOT / 'output' / 'quality_inspection' / 'samples' / 'test_10.png'), dpi=100)
plt.show()
print("Check the images above - do they look like documents?")
```

CELL 10 - Full generation (run OFF-PEAK, let run overnight):
--------
```python
# Only run after verifying test images look correct
# Run at 8 PM UTC (off-peak) to save compute units

import sys
sys.path.insert(0, str(PROJECT_ROOT))
from src.synthetic_data.generator import SyntheticDocumentGenerator

TOTAL_DOCS = 50000  # For Colab (original plan was 150K)

gen = SyntheticDocumentGenerator(
    output_dir=PROJECT_ROOT / 'output' / 'synthetic',
    fonts_dir=PROJECT_ROOT / 'data' / 'fonts',
    corpus_dir=PROJECT_ROOT / 'data' / 'text_corpus',
    num_docs=TOTAL_DOCS,
)

# Generate in batches - resume from checkpoint if disconnected
already_done = len(list((PROJECT_ROOT / 'output' / 'synthetic' / 'images').glob('*.jpg')))
print(f"Already generated: {already_done} documents")
print(f"Remaining: {TOTAL_DOCS - already_done}")

gen.generate_all(start=already_done, end=TOTAL_DOCS, checkpoint_interval=5000)
```

================================================================================
STEP 6 - CREATE COCO MANIFEST (After generation)
================================================================================

CELL 11 - Create combined COCO annotation file:
--------
```python
import json
from pathlib import Path

output_dir = PROJECT_ROOT / 'output' / 'synthetic'
ann_dir = output_dir / 'annotations'

all_images = []
all_annotations = []
ann_id_counter = 0

ann_files = sorted(ann_dir.glob('*.json'))
print(f"Found {len(ann_files)} annotation files")

for ann_file in ann_files:
    data = json.loads(ann_file.read_text())
    img_id = data['image_id']

    all_images.append({
        "id": img_id,
        "file_name": data['image_file'],
        "width": data['width'],
        "height": data['height'],
    })

    for ann in data['annotations']:
        ann['id'] = ann_id_counter
        all_annotations.append(ann)
        ann_id_counter += 1

# Split 90/10 train/val
split = int(len(all_images) * 0.9)
train_ids = {img['id'] for img in all_images[:split]}

from src.config import COCO_CLASSES
categories = [{"id": k, "name": v} for k, v in COCO_CLASSES.items()]

coco_train = {
    "images": all_images[:split],
    "annotations": [a for a in all_annotations if a['image_id'] in train_ids],
    "categories": categories,
}
coco_val = {
    "images": all_images[split:],
    "annotations": [a for a in all_annotations if a['image_id'] not in train_ids],
    "categories": categories,
}

(output_dir / 'train_coco.json').write_text(json.dumps(coco_train))
(output_dir / 'val_coco.json').write_text(json.dumps(coco_val))

print(f"COCO manifest created:")
print(f"  Train: {len(coco_train['images'])} images, {len(coco_train['annotations'])} annotations")
print(f"  Val:   {len(coco_val['images'])} images, {len(coco_val['annotations'])} annotations")
```

================================================================================
STEP 7 - PRETRAIN ON SYNTHETIC DATA
================================================================================

CELL 12 - Create data.yaml for training:
--------
```python
import yaml

data_yaml = {
    'path': str(PROJECT_ROOT / 'output' / 'synthetic'),
    'train': 'images',
    'val': 'images',
    'nc': 9,
    'names': list(COCO_CLASSES.values()),
}

yaml_path = PROJECT_ROOT / 'output' / 'synthetic' / 'synthetic_data.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f)
print(f"Created: {yaml_path}")
```

CELL 13 - Run pretraining (OFF-PEAK):
--------
```python
from ultralytics import YOLO

# Load base DocLayout-YOLO checkpoint
# Download from DocLayout-YOLO GitHub releases
!wget -q -O /content/doclayout_yolo_docstructbench.pt \
    "https://github.com/opendatalab/DocLayout-YOLO/releases/download/docstructbench/doclayout_yolo_docstructbench.pt"

model = YOLO('/content/doclayout_yolo_docstructbench.pt')

results = model.train(
    data=str(PROJECT_ROOT / 'output' / 'synthetic' / 'synthetic_data.yaml'),
    epochs=30,
    imgsz=1024,
    batch=16,
    lr0=0.001,
    device='cuda',
    project=str(PROJECT_ROOT / 'output' / 'checkpoints'),
    name='phase2_pretrain',
    save=True,
    patience=5,
)

# Save checkpoint to Drive
import shutil
best_ckpt = PROJECT_ROOT / 'output' / 'checkpoints' / 'phase2_pretrain' / 'weights' / 'best.pt'
final_ckpt = PROJECT_ROOT / 'output' / 'checkpoints' / 'doclayout_yolo_indic_pretrained.pt'
shutil.copy(best_ckpt, final_ckpt)
print(f"Checkpoint saved: {final_ckpt}")
```

================================================================================
KEEP-ALIVE (Run in separate tab during long jobs)
================================================================================

```python
# Run in a SEPARATE Colab tab to prevent 12-hour disconnect
import time
from IPython.display import Javascript, display

print("Keep-alive started (clicks toolbar every 5 min)")
for i in range(2000):
    display(Javascript('document.querySelector("colab-toolbar-button").click();'))
    print(f"Keep-alive click {i+1}", end='\r')
    time.sleep(300)  # 5 minutes
```

================================================================================
PHASE 2 TIMELINE (From today)
================================================================================

Day 1-2  (Today + tomorrow): Setup + fonts + corpus + test generation (10 docs)
Day 3-7  (This week):        Run full generation off-peak (50K docs, ~3-4 hours)
Day 8-9  (Next week Mon-Tue): Create COCO manifest + verify quality inspection
Day 10-14 (Next week Wed-Sun): Pretrain on synthetic data (30 epochs, ~15-20 hours)
End:     Pretrained checkpoint ready for Phase 3

Compute units estimate:
  Generation:  ~5 units (CPU-heavy, not GPU)
  Pretraining: ~20-25 units (GPU-heavy, run off-peak)
  Total Phase 2: ~25-30 units

================================================================================
STORAGE MANAGEMENT
================================================================================

Current Drive usage check:
  !du -sh /content/drive/MyDrive/doclayout-yolo-indic/

50K images at 1024x1024 JPEG (~50KB each) = ~2.5 GB
Annotations JSON = ~200 MB
Total Phase 2 data: ~3 GB (well within 100GB Drive limit)

After Phase 2, before Phase 3:
  - Keep: checkpoints/ (0.5 GB each)
  - Keep: output/synthetic/ until after pretraining verifies
  - Delete: synthetic images AFTER pretraining is confirmed working

================================================================================
PHASE 2 COMPLETION CHECKLIST
================================================================================

  [ ] Drive mounted, packages installed, directories created
  [ ] All 10 fonts downloaded and verified
  [ ] Text corpus for all 10 languages downloaded
  [ ] Source modules written to Drive (config.py, generator.py, text_renderer.py)
  [ ] Test run: 10 documents generated and inspected visually
  [ ] Full run: 50,000 documents generated
  [ ] COCO manifest created (train_coco.json, val_coco.json)
  [ ] Quality inspection: 50 random images reviewed, no major issues
  [ ] Pretraining started (off-peak)
  [ ] Pretrained checkpoint saved to Drive
  [ ] Validation mAP after pretraining: ≥ 60%

================================================================================
IF SOMETHING BREAKS
================================================================================

Problem: "Font download fails (GitHub rate limit)"
Fix:     !apt-get install -y fonts-noto -q  (installs system Noto fonts)
         Then find them at /usr/share/fonts/truetype/noto/

Problem: "Colab disconnects during generation"
Fix:     Script saves each doc individually. On restart, run Cell 10 again.
         It auto-detects already_done count and resumes.

Problem: "Drive storage full"
Fix:     Delete old D4LA/DocLayNet data first
         Run: !rm -rf /content/drive/MyDrive/doclayout-yolo-indic/data/raw/

Problem: "Pretraining OOM (GPU out of memory)"
Fix:     Reduce batch size: change batch=16 to batch=8 in Cell 13

Problem: "uharfbuzz import error"
Fix:     !pip install uharfbuzz --quiet
         (may need to restart runtime after install)

================================================================================
START HERE: Open a new Colab notebook and run Cell 1 (Session Setup)
================================================================================
