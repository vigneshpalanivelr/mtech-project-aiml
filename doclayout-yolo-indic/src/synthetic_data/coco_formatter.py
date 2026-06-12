"""
COCO-format annotation building and validation.

Produces a standard COCO detection JSON (images / annotations / categories)
that pycocotools and the Ultralytics COCO converter both accept. Also exposes
a per-image record writer for streaming large runs without holding everything
in memory.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.config import COCO_CLASSES, CLASS_NAME_TO_ID
from src.logger import get_logger

logger = get_logger(__name__)


def validate_bbox(bbox: List[float], img_w: int, img_h: int) -> bool:
    """COCO bbox is [x, y, w, h]; must be positive and within the image."""
    if len(bbox) != 4:
        return False
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False
    if x < 0 or y < 0 or x + w > img_w + 1 or y + h > img_h + 1:
        return False
    return True


class COCOBuilder:
    """Accumulates images + annotations and writes one COCO JSON."""

    def __init__(self):
        self.images: List[Dict] = []
        self.annotations: List[Dict] = []
        self._ann_id = 1

    @property
    def categories(self) -> List[Dict]:
        return [{"id": cid, "name": name, "supercategory": "layout"}
                for cid, name in COCO_CLASSES.items()]

    def add_image(self, image_id: int, file_name: str,
                  width: int, height: int) -> None:
        self.images.append({
            "id": image_id, "file_name": file_name,
            "width": width, "height": height,
        })

    def add_annotation(self, image_id: int, class_name: str,
                       bbox: List[float], img_w: int, img_h: int) -> bool:
        if not validate_bbox(bbox, img_w, img_h):
            logger.warning(f"img {image_id}: dropped invalid bbox {bbox} "
                           f"for class {class_name}")
            return False
        x, y, w, h = bbox
        self.annotations.append({
            "id": self._ann_id,
            "image_id": image_id,
            "category_id": CLASS_NAME_TO_ID[class_name],
            "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
            "area": round(w * h, 2),
            "iscrowd": 0,
        })
        self._ann_id += 1
        return True

    def to_dict(self) -> Dict:
        return {
            "info": {"description": "IndicSynth synthetic layout corpus"},
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        logger.info(f"Wrote COCO json: {len(self.images)} images, "
                    f"{len(self.annotations)} annotations -> {path.name}")


def write_per_image_record(path: Path, image_id: int, file_name: str,
                           size: Tuple[int, int],
                           regions: List[Tuple[str, List[float]]]) -> None:
    """Write a single-image annotation file (matches PROJECT_CONTEXT format)."""
    w, h = size
    anns = []
    for i, (cls, bbox) in enumerate(regions, start=1):
        if validate_bbox(bbox, w, h):
            anns.append({
                "id": i, "image_id": image_id,
                "category_id": CLASS_NAME_TO_ID[cls],
                "bbox": [round(v, 2) for v in bbox],
                "area": round(bbox[2] * bbox[3], 2), "iscrowd": 0,
            })
    record = {"image_id": image_id, "image_file": file_name,
              "width": w, "height": h, "annotations": anns}
    path.write_text(json.dumps(record), encoding="utf-8")
