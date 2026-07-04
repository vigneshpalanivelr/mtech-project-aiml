"""
Phase 3 — BaDLAD label preparation for the *real* self-training run.

Converts BaDLAD's 4-class COCO annotations into the model's 9-class YOLO space,
carves a held-out validation split from real labels, and partitions the 20,365
images into three DISJOINT sets so the self-training val cannot leak:

    val            real labels, held out, NEVER pseudo-labeled or trained on
    labeled train  real labels, the supervised anchor for self-training
    unlabeled pool  everything else, pseudo-labeled by CBST

Class mapping (BaDLAD id -> 9-class id):
    paragraph(0), text_box(1) -> text_body(0)
    image(2)                  -> figure(3)
    table(3)                  -> table(2)

Outputs (images materialized locally + cached to Drive; small labels on Drive):
    <local_root>/images/{train,val}/<stem>.jpg   (downscaled)
    <local_root>/labels/{train,val}/<stem>.txt   (YOLO: cls cx cy w h, normalised)
    <drive_out>/badlad_9class.yaml               (points at local image dirs)
    <drive_out>/exclude_stems.json               (labeled+val stems to skip in inference)

Usage (from the Phase 3 notebook):
    from src.finetuning.data_prep import prepare_badlad
    yaml_path, exclude_file = prepare_badlad(
        coco_json  = BADLAD_PATH + "/badlad-train-coco.json",
        images_dir = BADLAD_PATH + "/badlad_train",
        drive_out  = PROJECT_ROOT/"data"/"badlad_9class",
        local_root = "/content/badlad_9class",
    )
"""

import json
import random
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

try:
    from src.logger import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    logger = logging.getLogger(__name__)


# 9-class target space — MUST match the detector's head order.
NINE_CLASSES = ["text_body", "headline", "table", "figure", "caption",
                "advertisement", "sidebar", "pull_quote", "decorative_frame"]

# BaDLAD category_id -> 9-class id
CLASS_MAP = {0: 0, 1: 0, 2: 3, 3: 2}  # para,text_box->text_body; image->figure; table->table

# Defaults (tune for compute budget; see notes in the chat).
VAL_SIZE = 2000        # held-out real-label val images
TRAIN_CAP = 8000       # labeled supervised-anchor images (None = all remaining)
SEED = 0
MAX_SIDE = 1536        # downscale cap (>> imgsz=1024); labels are normalised
JPEG_QUALITY = 90


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """COCO [x, y, w, h] absolute -> YOLO [cx, cy, w, h] normalised, clamped."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    return cx, cy, nw, nh


def image_to_yolo_lines(img, anns, class_map=CLASS_MAP):
    """Convert one image's annotations to YOLO label lines (skips unmapped/degenerate)."""
    W, H = img["width"], img["height"]
    lines = []
    for a in anns:
        cls = class_map.get(a["category_id"])
        if cls is None:
            continue
        cx, cy, nw, nh = coco_bbox_to_yolo(a["bbox"], W, H)
        if nw <= 0 or nh <= 0:
            continue
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def _materialize_one(task):
    src, dst, max_side, q = task
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = min(1.0, max_side / max(w, h))
            if s < 1.0:
                im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
            im.save(dst, "JPEG", quality=q)
        return None
    except Exception as e:
        return f"{Path(src).name}: {e}"


def _split_images(images, val_size, train_cap, seed):
    """Return (val, train, rest) disjoint lists of image records."""
    imgs = list(images)
    random.Random(seed).shuffle(imgs)
    val = imgs[:val_size]
    train = imgs[val_size:val_size + train_cap] if train_cap else imgs[val_size:]
    rest = imgs[val_size + (train_cap or (len(imgs) - val_size)):]
    return val, train, rest


def _write_split(split_name, records, ann_by_img, images_dir, local_root,
                 workers, max_side, jpeg_q):
    """Write YOLO labels + materialize downscaled images for one split."""
    img_out = local_root / "images" / split_name
    lbl_out = local_root / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    # Index source images by filename stem, searching ALL sub-folders. kagglehub
    # sometimes extracts BaDLAD as badlad_train/badlad_train/*.png (doubled dir),
    # so a direct images_dir/file_name path is unreliable. This mirrors the
    # recursive lookup self_training.py already uses.
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    index = {}
    for p in Path(images_dir).rglob("*"):
        if p.suffix.lower() in exts:
            index.setdefault(p.stem, p)
    logger.info("%s split: indexed %d source images under %s",
                split_name, len(index), images_dir)

    tasks, n_lbl, n_box, n_nosrc = [], 0, 0, 0
    for img in records:
        stem = Path(img["file_name"]).stem
        lines = image_to_yolo_lines(img, ann_by_img.get(img["id"], []))
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines))
        n_lbl += 1
        n_box += len(lines)
        dst = img_out / f"{stem}.jpg"
        if not dst.exists():
            src = index.get(stem)
            if src is None:
                n_nosrc += 1
                continue
            tasks.append((str(src), str(dst), max_side, jpeg_q))

    miss = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, err in enumerate(ex.map(_materialize_one, tasks, chunksize=16), 1):
            if err:
                miss += 1
                logger.warning("  %s", err)
            if i % 2000 == 0:
                logger.info("  %s: materialised %d/%d images", split_name, i, len(tasks))
    logger.info("%s split: %d labels (%d boxes), %d images, %d failed, %d source-not-found",
                split_name, n_lbl, n_box, len(tasks), miss, n_nosrc)
    return img_out


def prepare_badlad(coco_json, images_dir, drive_out, local_root,
                   val_size=VAL_SIZE, train_cap=TRAIN_CAP, seed=SEED,
                   max_side=MAX_SIDE, jpeg_q=JPEG_QUALITY, workers=None):
    """Build the 9-class BaDLAD dataset. Returns (yaml_path, exclude_stems_file).

    Reuses a Drive cache if present so BaDLAD is only processed once.
    """
    import os
    workers = workers or os.cpu_count()
    images_dir = Path(images_dir)
    drive_out = Path(drive_out)
    local_root = Path(local_root)
    drive_out.mkdir(parents=True, exist_ok=True)
    yaml_path = drive_out / "badlad_9class.yaml"
    exclude_file = drive_out / "exclude_stems.json"
    cache_root = drive_out / "cache"

    # Fast path: reuse Drive cache ONLY if it is actually complete. The failure
    # mode we hit: a disk-full run wrote all the labels but few/no images, so a
    # naive "does train/ exist?" check kept reusing a broken cache forever. Here
    # we require each split's image count to match its label count; otherwise we
    # delete the bad cache and rebuild from scratch.
    def _n(d, ext):
        return len(list(d.glob(ext))) if d.exists() else 0

    if cache_root.exists() and exclude_file.exists():
        v_img = _n(cache_root / "images" / "val", "*.jpg")
        v_lbl = _n(cache_root / "labels" / "val", "*.txt")
        t_img = _n(cache_root / "images" / "train", "*.jpg")
        t_lbl = _n(cache_root / "labels" / "train", "*.txt")
        complete = (v_lbl > 0 and t_lbl > 0
                    and v_img >= 0.98 * v_lbl and t_img >= 0.98 * t_lbl)
        if complete:
            logger.info("Reusing Drive cache (train %d imgs / val %d imgs) -> local", t_img, v_img)
            for sub in ("images/train", "images/val", "labels/train", "labels/val"):
                (local_root / sub).mkdir(parents=True, exist_ok=True)
                for f in (cache_root / sub).glob("*"):
                    d = local_root / sub / f.name
                    if not d.exists():
                        shutil.copy(f, d)
            _write_yaml(yaml_path, local_root)
            return yaml_path, exclude_file
        logger.warning("Drive cache INCOMPLETE (train img/lbl=%d/%d, val img/lbl=%d/%d) "
                       "-> deleting and rebuilding", t_img, t_lbl, v_img, v_lbl)
        shutil.rmtree(cache_root, ignore_errors=True)
        shutil.rmtree(local_root, ignore_errors=True)

    logger.info("Loading BaDLAD COCO: %s", coco_json)
    coco = json.loads(Path(coco_json).read_text())
    ann_by_img = defaultdict(list)
    for a in coco["annotations"]:
        ann_by_img[a["image_id"]].append(a)

    val, train, rest = _split_images(coco["images"], val_size, train_cap, seed)
    logger.info("Partition: %d val | %d labeled-train | %d unlabeled-pool (disjoint)",
                len(val), len(train), len(rest))

    _write_split("val", val, ann_by_img, images_dir, local_root, workers, max_side, jpeg_q)
    _write_split("train", train, ann_by_img, images_dir, local_root, workers, max_side, jpeg_q)

    # Stems that must NOT enter the unlabeled/pseudo pool (val + labeled train).
    exclude = sorted({Path(im["file_name"]).stem for im in val}
                     | {Path(im["file_name"]).stem for im in train})
    exclude_file.write_text(json.dumps(exclude))
    logger.info("Wrote %d exclude stems -> %s", len(exclude), exclude_file)

    _write_yaml(yaml_path, local_root)

    # Cache to Drive so future sessions skip the rebuild.
    logger.info("Caching 9-class dataset to Drive -> %s", cache_root)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (cache_root / sub).mkdir(parents=True, exist_ok=True)
        for f in (local_root / sub).glob("*"):
            d = cache_root / sub / f.name
            if not d.exists():
                shutil.copy(f, d)

    return yaml_path, exclude_file


def _write_yaml(yaml_path, local_root):
    import yaml
    doc = {
        "train": str((local_root / "images" / "train").resolve()),
        "val": str((local_root / "images" / "val").resolve()),
        "nc": len(NINE_CLASSES),
        "names": NINE_CLASSES,
    }
    Path(yaml_path).write_text(yaml.safe_dump(doc, sort_keys=False))
    logger.info("Wrote %s", yaml_path)
    return yaml_path