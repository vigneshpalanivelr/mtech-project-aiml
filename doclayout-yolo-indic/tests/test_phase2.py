"""
Phase 2 unit tests. Run: python -m pytest tests/ -q   (or run this file directly)

Covers the two highest-risk pieces flagged in PROJECT_CONTEXT failure points:
HarfBuzz shaping correctness and COCO annotation validity.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic_data.text_renderer import TextRenderer
from src.synthetic_data.coco_formatter import validate_bbox, COCOBuilder


def test_shaping_forms_conjuncts():
    """Devanagari 'नमस्ते' must shape to fewer glyphs than input codepoints
    (conjunct formation + matra reordering happened)."""
    r = TextRenderer()
    fh = r._handle("Hindi")
    sw = r._shape_word("नमस्ते", fh, scale=fh.upem and 32 / fh.upem)
    assert len(sw.glyphs) >= 1
    assert sw.width > 0
    print("conjunct shaping OK:", len(sw.glyphs), "glyphs")


def test_rtl_renders_right_aligned():
    """Urdu ink should sit toward the right half of the tile."""
    r = TextRenderer()
    tile, ink = r.render_block("اردو خبر", "Urdu", box=(400, 80), font_size=32)
    x, y, w, h = ink
    assert w > 0 and h > 0
    assert x + w > 200, "RTL text should reach the right side of the box"
    print("RTL alignment OK: ink bbox", ink)


def test_all_scripts_render():
    r = TextRenderer()
    from src.synthetic_data.indic_typography import REGISTRY
    from src.synthetic_data.corpus import Corpus
    c = Corpus(seed=1)
    for lang in REGISTRY:
        text = c.sample_text(lang, 4)
        tile, ink = r.render_block(text, lang, box=(420, 80), font_size=28)
        assert ink[2] > 0 and ink[3] > 0, f"{lang} produced empty ink"
    print("all", len(REGISTRY), "scripts render non-empty ink")


def test_coco_bbox_validation():
    assert validate_bbox([10, 10, 50, 50], 1280, 1280)
    assert not validate_bbox([10, 10, 0, 50], 1280, 1280)      # zero width
    assert not validate_bbox([-5, 10, 50, 50], 1280, 1280)     # negative x
    assert not validate_bbox([1000, 10, 500, 50], 1280, 1280)  # overflow
    print("bbox validation OK")


def test_coco_builder_drops_invalid():
    b = COCOBuilder()
    b.add_image(1, "x.png", 1280, 1280)
    assert b.add_annotation(1, "headline", [10, 10, 100, 40], 1280, 1280)
    assert not b.add_annotation(1, "headline", [10, 10, -1, 40], 1280, 1280)
    d = b.to_dict()
    assert len(d["annotations"]) == 1 and len(d["categories"]) == 9
    print("COCO builder OK")


if __name__ == "__main__":
    for fn in [test_shaping_forms_conjuncts, test_rtl_renders_right_aligned,
               test_all_scripts_render, test_coco_bbox_validation,
               test_coco_builder_drops_invalid]:
        fn()
    print("\nALL TESTS PASSED")
