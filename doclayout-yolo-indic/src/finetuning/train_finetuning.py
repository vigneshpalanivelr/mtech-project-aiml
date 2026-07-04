"""
Phase 3 — Fine-tune the self-trained checkpoint on IndicDLP (9 -> 42 classes).

The final training stage. Takes the CBST self-trained checkpoint (9-class) and
fine-tunes it on IndicDLP's real labeled data in its NATIVE ontology. The
detection head reinitialises for IndicDLP's class count; backbone/neck transfer.

IndicDLP is stored on HF (VigneshPR/IndicDLP) as tars. Because we only fine-tune
on a SUBSET (train_cap of ~95K), we download just enough `train2017.tar.part-*`
chunks to cover that subset and partial-extract the complete images from them —
a tar is sequential, so the first K parts yield the first ~K*imgs complete
pages. This turns a ~42 GB pull into ~10-15 GB, all from Colab, then caches the
downscaled result to Drive so it never repeats.
"""

import itertools
import json
import os
import shutil
import string
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
# Not installed here anyway, but be explicit: plain, resumable HF downloads.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from src.finetuning import self_training as _st  # noqa: F401  (side-effect: patches)
from src.finetuning.data_prep import coco_bbox_to_yolo, _materialize_one

try:
    from src.logger import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    logger = logging.getLogger(__name__)


DEFAULT_TRAIN_CAP = 20000
DEFAULT_VAL_CAP = 3000
MAX_SIDE = 1536
JPEG_QUALITY = 90

FT_EPOCHS = 30
FT_IMGSZ = 1024
FT_BATCH = 16
FT_LR0 = 2e-4
FT_LRF = 0.01
FT_PATIENCE = 8

_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Staging: download only what we need, partial-extract from truncated tars
# ---------------------------------------------------------------------------
def _extract_complete(tar_path, out_dir):
    """Stream-extract every COMPLETE file from a (possibly truncated) tar.
    Stops gracefully at the truncation point; returns #files extracted."""
    n = 0
    try:
        with tarfile.open(tar_path, "r|") as t:
            for m in t:
                if m.isfile():
                    t.extract(m, out_dir)
                    n += 1
    except (tarfile.ReadError, EOFError, OSError):
        pass  # hit the truncation; everything before it is on disk
    return n


def _hf_get(hf_repo, filename, local_dir, token):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(hf_repo, filename, repo_type="dataset",
                                local_dir=str(local_dir), token=token))


def stage_indicdlp(hf_repo, local_dir, need_train=None, token=None):
    """Download + extract IndicDLP. val2017 fully; train2017 only enough parts to
    reach `need_train` complete images. Idempotent and resumable."""
    from huggingface_hub.utils import EntryNotFoundError
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train2017", "val2017"):
        try:
            _hf_get(hf_repo, f"annotations/instances_{split}.json", local_dir, token)
        except EntryNotFoundError:
            logger.warning("annotations for %s not found on repo", split)

    # ---- val2017: full ----
    val_dir = local_dir / "val2017"
    if not (val_dir.exists() and any(val_dir.glob("*.jpg"))):
        tar = local_dir / "val2017.tar"
        if not tar.exists():
            logger.info("Downloading val2017.tar ...")
            tar = _hf_get(hf_repo, "val2017.tar", local_dir, token)
        logger.info("Extracting val2017 ...")
        _extract_complete(tar, local_dir)
    logger.info("val2017: %d images", len(list(val_dir.glob('*.jpg'))))

    # ---- train2017: only enough parts ----
    train_dir = local_dir / "train2017"
    have = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0
    if not need_train:                                   # 0/None -> skip train entirely
        logger.info("need_train=%s -> skipping train2017 download", need_train)
        return local_dir
    if have >= need_train:
        logger.info("train2017 already has %d images (>= %d needed) -> skip", have, need_train)
        return local_dir

    # Append each downloaded part onto ONE growing tar, deleting the part right
    # away, so peak disk = (one 5 GB part) + (growing tar) + (extracted images) —
    # never all parts at once. Re-extract the growing prefix after each part.
    combined = local_dir / "train2017_partial.tar"
    combined.unlink(missing_ok=True)
    got = 0
    for i, suf in enumerate(("".join(p) for p in itertools.product(string.ascii_lowercase, repeat=2))):
        try:
            part = _hf_get(hf_repo, f"train2017.tar.part-{suf}", local_dir, token)
        except EntryNotFoundError:
            logger.info("no more train parts after %d", i)
            break
        with open(combined, "ab") as out, open(part, "rb") as f:   # append, don't rebuild
            shutil.copyfileobj(f, out, 16 * 1024 * 1024)
        part.unlink(missing_ok=True)                     # delete the 5 GB part immediately
        # clear HF's download cache too (it keeps a copy under .cache)
        shutil.rmtree(local_dir / ".cache", ignore_errors=True)
        shutil.rmtree(train_dir, ignore_errors=True)
        got = _extract_complete(combined, local_dir)
        logger.info("train2017: %d parts -> %d complete images (need %d)", i + 1, got, need_train)
        if got >= need_train:
            break
    combined.unlink(missing_ok=True)
    return local_dir


def read_classes(coco_json):
    """Return (names_in_id_order, id_to_index). Handles non-contiguous COCO ids."""
    coco = json.loads(Path(coco_json).read_text())
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    names = [c["name"] for c in cats]
    id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    logger.info("IndicDLP has %d classes: %s%s", len(names),
                ", ".join(names[:8]), " ..." if len(names) > 8 else "")
    return names, id_to_idx


# ---------------------------------------------------------------------------
# COCO -> YOLO for one split (only images actually on disk), Drive-cached
# ---------------------------------------------------------------------------
def _write_split(split_name, coco_json, images_dir, id_to_idx, local_root,
                 cap, workers, max_side, jpeg_q):
    img_out = local_root / "images" / split_name
    lbl_out = local_root / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    coco = json.loads(Path(coco_json).read_text())
    ann_by_img = {}
    for a in coco["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    # recursive index of images actually present on disk
    index = {}
    for p in Path(images_dir).rglob("*"):
        if p.suffix.lower() in _EXTS:
            index.setdefault(p.stem, p)
    logger.info("%s: %d source images present on disk", split_name, len(index))

    # use only images we HAVE (train2017 is a partial extraction), then cap
    images = [im for im in coco["images"] if Path(im["file_name"]).stem in index]
    if cap and len(images) > cap:
        import random
        random.Random(0).shuffle(images)
        images = images[:cap]

    tasks, n_lbl, n_box = [], 0, 0
    for img in images:
        stem = Path(img["file_name"]).stem
        W, H = img["width"], img["height"]
        lines = []
        for a in ann_by_img.get(img["id"], []):
            cls = id_to_idx.get(a["category_id"])
            if cls is None:
                continue
            cx, cy, nw, nh = coco_bbox_to_yolo(a["bbox"], W, H)
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines))
        n_lbl += 1
        n_box += len(lines)
        dst = img_out / f"{stem}.jpg"
        if not dst.exists():
            tasks.append((str(index[stem]), str(dst), max_side, jpeg_q))

    miss = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, err in enumerate(ex.map(_materialize_one, tasks, chunksize=16), 1):
            if err:
                miss += 1
            if i % 2000 == 0:
                logger.info("  %s: materialised %d/%d", split_name, i, len(tasks))
    logger.info("%s: %d labels (%d boxes), %d images, %d failed",
                split_name, n_lbl, n_box, len(tasks), miss)
    return img_out


def prepare_indicdlp(hf_repo, drive_out, local_root, staging="/content/IndicDLP_raw",
                     train_cap=DEFAULT_TRAIN_CAP, val_cap=DEFAULT_VAL_CAP,
                     max_side=MAX_SIDE, jpeg_q=JPEG_QUALITY, workers=None, token=None):
    """Produce a local 42-class YOLO dataset + data.yaml, Drive-cached (integrity
    checked). Returns (yaml_path, class_names)."""
    workers = workers or os.cpu_count()
    drive_out = Path(drive_out)
    local_root = Path(local_root)
    drive_out.mkdir(parents=True, exist_ok=True)
    yaml_path = drive_out / "indicdlp.yaml"
    cache_root = drive_out / "cache"
    names_file = drive_out / "classes.json"

    def _n(d, ext):
        return len(list(Path(d).glob(ext))) if Path(d).exists() else 0

    if cache_root.exists() and names_file.exists():
        v_img = _n(cache_root / "images" / "val2017", "*.jpg")
        v_lbl = _n(cache_root / "labels" / "val2017", "*.txt")
        t_img = _n(cache_root / "images" / "train2017", "*.jpg")
        t_lbl = _n(cache_root / "labels" / "train2017", "*.txt")
        if v_lbl > 0 and t_lbl > 0 and v_img >= 0.98 * v_lbl and t_img >= 0.98 * t_lbl:
            logger.info("Reusing Drive cache (train %d / val %d imgs) -> local", t_img, v_img)
            for sub in ("images/train2017", "images/val2017", "labels/train2017", "labels/val2017"):
                (local_root / sub).mkdir(parents=True, exist_ok=True)
                for f in (cache_root / sub).glob("*"):
                    d = local_root / sub / f.name
                    if not d.exists():
                        shutil.copy(f, d)
            names = json.loads(names_file.read_text())
            _write_yaml(yaml_path, local_root, names)
            return yaml_path, names
        logger.warning("Drive cache incomplete -> rebuilding")
        shutil.rmtree(cache_root, ignore_errors=True)
        shutil.rmtree(local_root, ignore_errors=True)

    raw = stage_indicdlp(hf_repo, staging, need_train=int(train_cap * 1.1) if train_cap else None,
                         token=token)
    train_json = raw / "annotations" / "instances_train2017.json"
    val_json = raw / "annotations" / "instances_val2017.json"
    names, id_to_idx = read_classes(train_json)
    names_file.write_text(json.dumps(names))

    _write_split("val2017", val_json, raw / "val2017", id_to_idx, local_root,
                 val_cap, workers, max_side, jpeg_q)
    _write_split("train2017", train_json, raw / "train2017", id_to_idx, local_root,
                 train_cap, workers, max_side, jpeg_q)

    _write_yaml(yaml_path, local_root, names)

    logger.info("Caching 42-class dataset to Drive -> %s", cache_root)
    for sub in ("images/train2017", "images/val2017", "labels/train2017", "labels/val2017"):
        (cache_root / sub).mkdir(parents=True, exist_ok=True)
        for f in (local_root / sub).glob("*"):
            d = cache_root / sub / f.name
            if not d.exists():
                shutil.copy(f, d)
    return yaml_path, names


def _write_yaml(yaml_path, local_root, names):
    import yaml
    doc = {
        "train": str((Path(local_root) / "images" / "train2017").resolve()),
        "val": str((Path(local_root) / "images" / "val2017").resolve()),
        "nc": len(names),
        "names": list(names),
    }
    Path(yaml_path).write_text(yaml.safe_dump(doc, sort_keys=False))
    logger.info("Wrote %s (nc=%d)", yaml_path, len(names))
    return yaml_path


# ---------------------------------------------------------------------------
# Fine-tune: re-head 9 -> 42 and train
# ---------------------------------------------------------------------------
def run_finetuning(self_trained, hf_repo, drive_out, local_root, work_dir,
                   train_cap=DEFAULT_TRAIN_CAP, val_cap=DEFAULT_VAL_CAP,
                   epochs=FT_EPOCHS, device=0, token=None):
    """Prepare IndicDLP then fine-tune. Head auto-reinitialises to IndicDLP's
    class count; backbone transfers. Returns best.pt (the final Phase 3 model)."""
    from doclayout_yolo import YOLOv10
    work_dir = Path(work_dir)
    best = work_dir / "finetune" / "weights" / "best.pt"
    if best.exists():
        logger.info("Fine-tuning already complete -> %s (skipping)", best)
        return best

    yaml_path, names = prepare_indicdlp(hf_repo, drive_out, local_root,
                                        train_cap=train_cap, val_cap=val_cap, token=token)
    logger.info("Fine-tuning %s on IndicDLP (%d classes)", Path(self_trained).name, len(names))

    last = work_dir / "finetune" / "weights" / "last.pt"
    resume = last.exists()
    init = str(last) if resume else str(self_trained)
    model = YOLOv10(init)
    model.train(
        data=str(yaml_path), epochs=epochs, imgsz=FT_IMGSZ, batch=FT_BATCH,
        lr0=FT_LR0, lrf=FT_LRF, patience=FT_PATIENCE, optimizer="AdamW", device=device,
        project=str(work_dir), name="finetune", save_period=1, resume=resume,
        exist_ok=True, plots=True,
    )
    logger.info("Fine-tuning done. Final Phase 3 model -> %s", best)
    return best