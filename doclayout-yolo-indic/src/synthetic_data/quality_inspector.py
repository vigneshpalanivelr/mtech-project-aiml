"""
Quality inspection (Phase 2, Step 6).

Draws COCO bounding boxes + class labels over generated pages so you can
eyeball: text readable, boxes tight, RTL/mixed pages sane. Produces a contact
sheet and a small markdown report stub.

Run:
    python -m src.synthetic_data.quality_inspector --tag smoke --num 12
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.config import COCO_CLASSES
from src.logger import get_logger
from src.utils.paths import get_synthetic_dirs, get_output_dir

logger = get_logger(__name__)

_COLORS = [(220, 50, 50), (50, 140, 220), (40, 170, 90), (200, 130, 30),
           (150, 60, 200), (30, 170, 170), (200, 60, 140), (110, 110, 60),
           (90, 90, 200)]


def overlay(image_path: Path, annotations: list) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(16)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        cid = ann["category_id"]
        color = _COLORS[cid % len(_COLORS)]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        label = COCO_CLASSES.get(cid, str(cid))
        draw.rectangle([x, y - 18, x + 9 * len(label), y], fill=color)
        draw.text((x + 2, y - 18), label, fill=(255, 255, 255), font=font)
    return img


def build_contact_sheet(tag: str, num: int, cols: int = 4) -> Path:
    dirs = get_synthetic_dirs()
    ann_dir = dirs["annotations"]
    out_dir = get_output_dir("quality_inspection", "samples")

    thumbs, used = [], []
    for i in range(1, num + 1):
        rec_path = ann_dir / f"{tag}_{i:06d}.json"
        if not rec_path.exists():
            continue
        rec = json.loads(rec_path.read_text())
        img_path = dirs["images"] / rec["image_file"]
        composed = overlay(img_path, rec["annotations"])
        composed.save(out_dir / f"overlay_{tag}_{i:06d}.png")
        thumbs.append(composed.resize((320, 320)))
        used.append(rec["image_file"])

    if not thumbs:
        logger.warning("No annotated images found to inspect.")
        return out_dir
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 320, rows * 320), (255, 255, 255))
    for idx, t in enumerate(thumbs):
        r, c = divmod(idx, cols)
        sheet.paste(t, (c * 320, r * 320))
    sheet_path = get_output_dir("quality_inspection") / f"contact_sheet_{tag}.png"
    sheet.save(sheet_path)

    report = get_output_dir("quality_inspection") / "inspection_report.md"
    report.write_text(
        f"# Quality Inspection ({tag})\n\n"
        f"Inspected {len(used)} pages. Contact sheet: {sheet_path.name}\n\n"
        "## Checklist\n"
        "- [ ] Text readable, no overlap/clipping\n"
        "- [ ] Boxes tight around content\n"
        "- [ ] Conjuncts/matras correct\n"
        "- [ ] RTL (Urdu) reads right-to-left\n"
        "- [ ] Mixed-script pages look natural\n",
        encoding="utf-8")
    logger.info(f"Contact sheet -> {sheet_path}")
    return sheet_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--num", type=int, default=12)
    args = ap.parse_args()
    build_contact_sheet(args.tag, args.num)


if __name__ == "__main__":
    main()
