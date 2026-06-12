"""
Synthetic Indic document generator (Phase 2, Steps 4-5).

Pipeline per page:
  1. pick a layout template
  2. pick a primary script (weighted); 20% of pages add Latin as a secondary
  3. for RTL primary scripts (Urdu) mirror the *column layout* (not the pixels,
     so glyphs stay correct) -> authentic right-to-left reading order
  4. fill each region: shaped text / figure box / table grid / frame
  5. emit the page PNG + COCO annotations + the script label (for the
     auxiliary script-classification head in pretraining)

Run a small smoke batch:
    python -m src.synthetic_data.generator --num 12 --tag smoke
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw

from src.config import (SYNTHETIC_IMG_SIZE, PAGE_BG, MIXED_SCRIPT_PROB,
                        SCRIPT_SAMPLE_WEIGHTS, LATIN_LABEL, RANDOM_SEED)
from src.logger import get_logger
from src.synthetic_data.corpus import Corpus
from src.synthetic_data.templates import pick_template
from src.synthetic_data.text_renderer import TextRenderer
from src.synthetic_data.indic_typography import get_spec, is_rtl
from src.synthetic_data.coco_formatter import COCOBuilder, write_per_image_record
from src.utils.paths import get_synthetic_dirs

logger = get_logger(__name__)

# font-size band (in px) by region class, before area scaling
_FONT_BAND = {
    "headline": (40, 56), "pull_quote": (28, 36), "caption": (16, 20),
    "sidebar": (16, 22), "text_body": (18, 26), "advertisement": (22, 30),
}
_GRID = (190, 190, 190)
_FILL = (225, 225, 225)


def _weighted_script(rng: random.Random) -> str:
    langs = list(SCRIPT_SAMPLE_WEIGHTS)
    weights = [SCRIPT_SAMPLE_WEIGHTS[l] for l in langs]
    return rng.choices(langs, weights=weights, k=1)[0]


def _abs_box(box, W, H) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    return int(x * W), int(y * H), int(w * W), int(h * H)


class SyntheticDocumentGenerator:
    def __init__(self, seed: int = RANDOM_SEED):
        self.rng = random.Random(seed)
        self.renderer = TextRenderer()
        self.corpus = Corpus(seed=seed)
        self.size = SYNTHETIC_IMG_SIZE

    # -- region fillers -----------------------------------------------------

    def _fill_text(self, page, draw, region, abs_box, language):
        x, y, w, h = abs_box
        lo, hi = _FONT_BAND.get(region["cls"], (18, 26))
        font_size = self.rng.randint(lo, hi)
        n_words = max(2, int(region.get("words", 20) *
                             self.rng.uniform(0.7, 1.1)))
        text = self.corpus.sample_text(language, n_words)
        tile, ink = self.renderer.render_block(
            text, language, box=(w, h), font_size=font_size)
        if ink[2] <= 0 or ink[3] <= 0:
            return None
        page.paste(tile, (x, y), tile)
        return [region["cls"], [x + ink[0], y + ink[1], ink[2], ink[3]]]

    def _fill_figure(self, page, draw, region, abs_box, language):
        x, y, w, h = abs_box
        draw.rectangle([x, y, x + w, y + h], fill=_FILL, outline=_GRID, width=2)
        draw.line([x, y, x + w, y + h], fill=_GRID, width=2)
        draw.line([x + w, y, x, y + h], fill=_GRID, width=2)
        return [region["cls"], [x, y, w, h]]

    def _fill_table(self, page, draw, region, abs_box, language):
        x, y, w, h = abs_box
        draw.rectangle([x, y, x + w, y + h], outline=(90, 90, 90), width=2)
        rows = self.rng.randint(3, 7)
        cols = self.rng.randint(2, 5)
        for r in range(1, rows):
            yy = y + h * r / rows
            draw.line([x, yy, x + w, yy], fill=_GRID, width=1)
        for c in range(1, cols):
            xx = x + w * c / cols
            draw.line([xx, y, xx, y + h], fill=_GRID, width=1)
        return [region["cls"], [x, y, w, h]]

    def _fill_frame(self, page, draw, region, abs_box, language):
        x, y, w, h = abs_box
        draw.rectangle([x, y, x + w, y + h], outline=(120, 120, 120), width=3)
        return [region["cls"], [x, y, w, h]]

    # -- one document -------------------------------------------------------

    def generate_one(self, doc_id: int) -> Tuple[Image.Image, List, str]:
        W, H = self.size
        primary = _weighted_script(self.rng)
        mixed = self.rng.random() < MIXED_SCRIPT_PROB
        rtl = is_rtl(primary)

        template = pick_template(self.rng)
        page = Image.new("RGB", (W, H), (PAGE_BG, PAGE_BG, PAGE_BG))
        draw = ImageDraw.Draw(page)
        regions_out: List = []

        for region in template["regions"]:
            box = region["box"]
            if rtl:  # mirror column layout for right-to-left reading order
                x, y, w, h = box
                box = (1.0 - (x + w), y, w, h)
            abs_box = _abs_box(box, W, H)

            # choose language for this region
            lang = primary
            if mixed and region["cls"] in ("caption", "sidebar", "pull_quote"):
                lang = LATIN_LABEL

            kind = region["kind"]
            if kind == "text":
                res = self._fill_text(page, draw, region, abs_box, lang)
            elif kind == "figure":
                res = self._fill_figure(page, draw, region, abs_box, lang)
            elif kind == "table":
                res = self._fill_table(page, draw, region, abs_box, lang)
            else:
                res = self._fill_frame(page, draw, region, abs_box, lang)
            if res:
                regions_out.append(res)

        return page, regions_out, get_spec(primary).script_head

    # -- batch driver -------------------------------------------------------

    def generate(self, num_docs: int, tag: str = "synthetic") -> Path:
        dirs = get_synthetic_dirs()
        builder = COCOBuilder()
        manifest = []
        W, H = self.size

        for i in range(num_docs):
            doc_id = i + 1
            page, regions, script_label = self.generate_one(doc_id)
            fname = f"{tag}_{doc_id:06d}.png"
            page.save(dirs["images"] / fname)

            builder.add_image(doc_id, fname, W, H)
            for cls, bbox in regions:
                builder.add_annotation(doc_id, cls, bbox, W, H)
            write_per_image_record(
                dirs["annotations"] / f"{tag}_{doc_id:06d}.json",
                doc_id, fname, (W, H), regions)
            manifest.append({"image_id": doc_id, "file_name": fname,
                             "script": script_label})

            if doc_id % 10_000 == 0:
                logger.info(f"Generated {doc_id}/{num_docs}")

        builder.write(dirs["root"] / f"{tag}_coco.json")
        (dirs["root"] / f"{tag}_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        logger.info(f"Done: {num_docs} docs under {dirs['root']}")
        return dirs["root"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--tag", type=str, default="smoke")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()
    SyntheticDocumentGenerator(seed=args.seed).generate(args.num, args.tag)


if __name__ == "__main__":
    main()
