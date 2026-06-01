# DocLayout-YOLO-Indic: Project Context Document

## Executive Summary

**Project:** Extend DocLayout-YOLO (real-time layout detector) to work reliably on Indic scripts (Hindi, Tamil, Telugu, Bengali, Urdu, etc.)

**Timeline:** May 5, 2026 – Aug 2, 2026 (12 weeks)

**Current Status:** Phase 1 (Outline & Setup) completed on paper. Starting Phase 2 (Synthetic Data + Pretraining) immediately.

**End Goal:** Open-source, real-time, locally-runnable layout detector for Indic scripts, released with trained checkpoints and code.

---

## Project Phases Overview

| Phase | Duration | Goal | Deliverable |
|---|---|---|---|
| **Phase 1: Outline & Setup** | May 5-10 | Reproduce baseline, download datasets, set up environment | Baseline working, datasets ready |
| **Phase 2: Synthetic Data + Pretraining** | May 11-30 | Build Indic synthetic documents, pretrain model | IndicSynth-150K corpus + pretrained checkpoint |
| **Phase 3: Self-Training + Fine-tuning** | Jun 1-Jul 6 | Learn from unlabeled data, fine-tune on IndicDLP | Fine-tuned checkpoint with ablations |
| **Phase 4: Evaluation & Analysis** | Jul 7-22 | Test on multiple datasets, analyze failures | Results tables, paper draft, failure analysis |
| **Phase 5: Review & Submission** | Jul 23-Aug 2 | Final polish, code release, submission | GitHub repo + trained models + paper |

---

## Phase 2: Synthetic Data Generation (YOUR CURRENT WORK)

**Duration:** May 11 – May 30, 2026 (20 days)

**Goal:** Generate 150,000 realistic synthetic Indic documents with proper script rendering for pretraining.

### Why This Matters

DocLayout-YOLO was trained on 300,000 English/Chinese synthetic documents. The model learned English/Chinese layout patterns. When applied to Indic documents, it fails because:
- Devanagari text has horizontal lines (shirorekha) that confuse the model
- Tamil and Telugu have vertical stacking of conjuncts and matras
- Mixed-script pages (Hindi+English) break expectations
- Urdu flows right-to-left, not left-to-right

**Your solution:** Generate 150,000 *realistic Indic documents* using proper script rendering, so the model learns Indic layout patterns from the start.

---

## Phase 2: Detailed Implementation Steps

### Step 1: Prepare Fonts and Text Corpus (Days 1-2)

**What you do:**
- Download Noto Sans Indic fonts for each script
- Collect text samples in each language

**Outputs:**
```
data/
  fonts/
    Noto_Sans_Devanagari.ttf
    Noto_Sans_Bengali.ttf
    Noto_Sans_Tamil.ttf
    Noto_Sans_Telugu.ttf
    Noto_Sans_Kannada.ttf
    Noto_Sans_Malayalam.ttf
    Noto_Sans_Gujarati.ttf
    Noto_Sans_Gurmukhi.ttf
    Noto_Sans_Oriya.ttf
    Noto_Naskh_Arabic.ttf (for Urdu)

  text_corpus/
    hindi_words.txt (10,000 words)
    bengali_words.txt (10,000 words)
    tamil_words.txt (10,000 words)
    telugu_words.txt (10,000 words)
    kannada_words.txt (10,000 words)
    malayalam_words.txt (10,000 words)
    gujarati_words.txt (10,000 words)
    punjabi_words.txt (10,000 words)
    odia_words.txt (10,000 words)
    urdu_words.txt (10,000 words)
```

**Code location:** `src/data_preparation/font_setup.py`

---

### Step 2: Define Layout Templates (Days 3-4)

**What you do:**
- Define 20-30 realistic document layouts (newspaper, textbook, form, etc.)
- Each template specifies position of headline, body text, tables, sidebars, images

**Template example:**
```python
templates = {
  "newspaper_layout_1": {
    "headline": {"x": 50, "y": 50, "width": 900, "height": 100},
    "body_col1": {"x": 50, "y": 200, "width": 400, "height": 600},
    "body_col2": {"x": 500, "y": 200, "width": 400, "height": 600},
    "sidebar": {"x": 950, "y": 300, "width": 150, "height": 200},
    "image": {"x": 50, "y": 850, "width": 900, "height": 150},
  },
  "textbook_layout_1": {...},
  ...
}
```

**Outputs:** `src/synthetic_data/templates.py`

---

### Step 3: Build HarfBuzz Text Rendering Pipeline (Days 5-8)

**What you do:**
- Use HarfBuzz to properly shape text for each Indic script
- HarfBuzz handles:
  - Devanagari shirorekha (horizontal line)
  - Tamil/Telugu conjuncts and matras
  - Bengali ligatures
  - Urdu right-to-left flow
  - Proper glyph positioning

**Key code:**
```python
import uharfbuzz as hb

def shape_text_indic(text, script, font_path):
    """
    Properly shape Indic text using HarfBuzz.
    Handles conjuncts, matras, RTL, etc.
    """
    with open(font_path, 'rb') as f:
        font_data = f.read()

    face = hb.Face(font_data)
    font = hb.Font(face)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()

    hb.shape(font, buf)

    return buf.glyph_infos, buf.glyph_positions
```

**Outputs:**
- `src/synthetic_data/text_renderer.py` — HarfBuzz wrapper
- `src/synthetic_data/indic_typography.py` — Script-specific rules

---

### Step 4: Generate Synthetic Documents (Days 9-15)

**What you do:**
- For each of 150,000 documents:
  1. Pick random layout template
  2. Pick random Indic script (weighted by language importance)
  3. Sample random text from corpus
  4. Use HarfBuzz to shape text properly
  5. Render shaped text to image
  6. Place rendered text in layout positions
  7. Add images/tables randomly
  8. Save image + COCO annotation

**Pseudocode:**
```python
def generate_synthetic_corpus(num_documents=150000):
    for doc_id in range(num_documents):
        # Pick random script (20% chance of mixed-script)
        if random.random() < 0.2:
            scripts = [random_script(), "English"]
        else:
            scripts = [random_script()]

        # Pick layout template
        template = random.choice(templates)

        # Create blank image
        img = create_blank_page(1280, 1280)
        annotations = []

        # Fill each region in template
        for region_name, region_coords in template.items():
            script = random.choice(scripts)
            text = sample_text(corpus[script], num_words=50)

            # KEY STEP: Shape text for the script
            shaped_text = shape_text_indic(text, script, fonts[script])

            # Render to image
            text_img = render_shaped_text(shaped_text, region_coords)

            # Place on page
            place_on_page(img, text_img, region_coords)

            # Add annotation
            annotations.append({
                "bbox": region_coords,
                "class": classify_region(region_name)
            })

        # Handle RTL for Urdu
        if "Urdu" in scripts:
            img = horizontal_flip(img)
            annotations = flip_annotations(annotations)

        # Save
        save_image(img, f"output/synthetic_{doc_id}.png")
        save_coco_annotation(annotations, f"output/synthetic_{doc_id}.json")

        if doc_id % 10000 == 0:
            print(f"Generated {doc_id}/150000")
```

**Outputs:** `src/synthetic_data/generator.py`

---

### Step 5: COCO Format Annotation (Days 16-17)

**What you do:**
- For each synthetic image, create COCO-format JSON with bounding boxes and class labels
- Ensures compatibility with DocLayout-YOLO training pipeline

**Output format:**
```json
{
  "image_id": 1,
  "image_file": "synthetic_0001.png",
  "width": 1280,
  "height": 1280,
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [50, 50, 900, 100],
      "area": 90000,
      "iscrowd": 0
    },
    {
      "id": 2,
      "image_id": 1,
      "category_id": 2,
      "bbox": [50, 200, 400, 600],
      "area": 240000,
      "iscrowd": 0
    }
  ]
}
```

**Class mapping:**
```
0: text_body
1: headline
2: table
3: figure
4: caption
5: advertisement
6: sidebar
7: pull_quote
8: decorative_frame
```

**Outputs:** `src/synthetic_data/coco_formatter.py`

---

### Step 6: Quality Control & Inspection (Days 18-19)

**What you do:**
- Generate 150,000 documents
- Inspect random sample (500 images) by eye
- Check for:
  - Text is readable and properly positioned
  - Scripts are rendered correctly (no overlapping characters)
  - Layouts are realistic
  - Mixed-script documents look natural
  - Urdu RTL is correct

**Outputs:**
- `output/quality_inspection/` — sample images for manual review
- `output/inspection_report.md` — notes on any issues found

---

### Step 7: Pretraining on Synthetic Data (Days 20+, overlaps with Phase 3)

**What you do:**
- Load pretrained DocLayout-YOLO checkpoint (trained on English/Chinese)
- Load IndicSynth-150K corpus
- Pretrain for 30 epochs with:
  - Detection loss (bounding box prediction)
  - Script classification auxiliary loss (predict Hindi/Tamil/etc.)
  - Cosine learning rate schedule

**Code location:** `src/pretraining/train_synthetic.py`

**Expected outputs:**
- `output/checkpoints/doclayout_yolo_indic_pretrained.pt` — pretrained checkpoint
- `output/logs/pretraining_metrics.csv` — accuracy/loss curves

---

## Key Implementation Details

### Required Libraries
```
torch>=2.0
torchvision
ultralytics  # YOLO
uharfbuzz    # Text shaping for Indic scripts
pillow       # Image handling
numpy
opencv-python
tqdm         # Progress bars
```

### Scripts to Create

```
src/
  synthetic_data/
    __init__.py
    font_setup.py              # Download and verify fonts
    templates.py               # Layout templates
    text_renderer.py           # HarfBuzz wrapper for Indic text
    indic_typography.py        # Script-specific rendering rules
    generator.py               # Main synthetic document generator
    coco_formatter.py          # Convert to COCO format
    quality_inspector.py       # Visual inspection tools

  pretraining/
    __init__.py
    train_synthetic.py         # Pretraining on IndicSynth
    config.py                  # Hyperparameters
    callbacks.py               # Logging, checkpointing

  utils/
    __init__.py
    logger.py                  # Logging utilities
    paths.py                   # Directory management

data/
  fonts/
    (Noto Sans fonts for each script)

  text_corpus/
    (Word lists for each language)

  raw/
    (IndicDLP and BaDLAD when downloaded)

output/
  synthetic/
    (Generated synthetic images and annotations)

  checkpoints/
    (Trained model weights)

  logs/
    (Training metrics, inspection reports)
```

---

## Metrics to Track During Phase 2

By end of Phase 2, you should have:

1. **IndicSynth-150K corpus generated**
   - 150,000 images
   - Proper COCO annotations
   - ~30-50 GB storage

2. **Quality inspection completed**
   - 500 sample images manually reviewed
   - No major rendering issues
   - Mixed-script and RTL documents verified

3. **Pretrained checkpoint saved**
   - `doclayout_yolo_indic_pretrained.pt`
   - Trained for 30 epochs
   - Validation mAP on held-out synthetic test set: target ≥ 60 mAP

4. **Documentation**
   - README explaining synthetic generation process
   - Sample synthetic images in `output/examples/`
   - Pretraining metrics and loss curves

---

## Failure Points to Watch

1. **HarfBuzz rendering bugs** — conjuncts clipping, matras misaligned
   - **Fix:** Unit test the renderer on small text samples first

2. **Unrealistic layouts** — generated documents don't look like real ones
   - **Fix:** Iterate templates based on inspection feedback

3. **COCO annotation errors** — bounding boxes don't match text
   - **Fix:** Visualize a few annotations before full generation

4. **RTL Urdu broken** — right-to-left text still flows left-to-right
   - **Fix:** Test RTL rendering separately before full generation

5. **Pretraining not converging** — loss stays high
   - **Fix:** Check if synthetic data is too different from real data; may need augmentation

---

## Phase 2 Completion Checklist

- [ ] Fonts downloaded and verified (all 10 scripts)
- [ ] Text corpus prepared (10,000 words per language)
- [ ] Layout templates defined (20-30 templates)
- [ ] HarfBuzz text renderer working (tested on small samples)
- [ ] Synthetic document generator running
- [ ] Quality inspection completed (500 samples reviewed, no major issues)
- [ ] IndicSynth-150K corpus generated and saved
- [ ] COCO annotations created and validated
- [ ] Pretraining started and running smoothly
- [ ] First checkpoint saved

---

## Next Phases (Reference)

**Phase 3 (Jun 1-Jul 6):** Self-training on BaDLAD unlabeled data + fine-tuning on IndicDLP

**Phase 4 (Jul 7-22):** Evaluation on multiple Indic datasets + failure analysis + paper draft

**Phase 5 (Jul 23-Aug 2):** Final review, code release, submission

---

## Important Contacts & Resources

- **Supervisor:** [Your supervisor name]
- **Base Paper:** Wang et al., "DocLayout-YOLO" (arXiv:2405.19209, NeurIPS 2024)
- **YOLO Docs:** https://docs.ultralytics.com/
- **HarfBuzz Python:** https://github.com/uharfbuzz/uharfbuzz
- **Noto Fonts:** https://www.google.com/get/noto/
- **COCO Format:** https://cocodataset.org/#format-data