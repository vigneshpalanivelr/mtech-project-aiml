# DocLayout-YOLO-Indic — Phase 2: Synthetic Data Engine

Generates realistic synthetic Indic-script documents with correct typography
(shirorekha, conjuncts, matras, RTL Urdu) and COCO annotations, for pretraining
a layout detector. See `PROJECT_CONTEXT.md` for the full plan.

## Install (data engine — CPU, runs anywhere)
```bash
pip install uharfbuzz fonttools pillow numpy tqdm
```

## Run the pipeline
```bash
# 1. fetch + verify Noto fonts for all 10 scripts (+ Latin)
python -m src.synthetic_data.font_setup

# 2. generate a smoke batch and inspect it
python -m src.synthetic_data.generator --num 50 --tag smoke
python -m src.synthetic_data.quality_inspector --tag smoke --num 50
#   -> output/quality_inspection/contact_sheet_smoke.png

# 3. unit tests (shaping + COCO validity)
python tests/test_phase2.py

# 4. full corpus (run on the 32-core/64 GB box, not a laptop)
python -m src.synthetic_data.generator --num 150000 --tag indicsynth

# 5. prepare YOLO labels (CPU) then pretrain (GPU box)
python -m src.pretraining.train_synthetic --prepare --tag indicsynth
python -m src.pretraining.train_synthetic --train \
    --coco output/synthetic/indicsynth_coco.json \
    --weights <DocLayout-YOLO.pt>
```

## Layout
```
src/synthetic_data/  font_setup, indic_typography, corpus, templates,
                     text_renderer (HarfBuzz+fontTools), generator,
                     coco_formatter, quality_inspector
src/pretraining/     train_synthetic (COCO->YOLO + training launch)
src/{config,logger}  single source of truth + logging
src/utils/paths      directory management
tests/test_phase2    renderer + COCO unit tests
```

## Notes
- Rendering uses HarfBuzz shaping + fontTools outline fill (no libraqm needed).
- Drop real word lists into `data/text_corpus/<lang>_words.txt` to replace the
  synthetic placeholder vocabulary before freezing the released dataset.
- The 9-way auxiliary script head (Angle C) needs a custom Ultralytics trainer
  subclass — scheduled for Week 7.
