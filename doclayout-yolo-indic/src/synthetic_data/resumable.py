"""
Resumable, crash-safe synthetic generation for Colab.

The plain parallel driver writes everything to local /content and only emits the
combined COCO at the very end -- so a disconnect loses the whole run. This driver
instead generates in SHARDS and saves each finished shard as one tar on Drive:

    drive_dir/<tag>_shard_0000.tar
    drive_dir/<tag>_shard_0001.tar
    ...

On restart it checks Drive, skips shards already present, and regenerates only
what is missing. Each shard's pages are deterministic (seeded by absolute id),
so a regenerated shard is byte-for-byte the same. A shard tar appears on Drive
only after it is fully written (temp-then-rename), so a mid-copy crash never
leaves a shard that looks done but is corrupt.

Cell 4 usage:
    from pathlib import Path
    from src.synthetic_data.resumable import generate_resumable
    generate_resumable(
        num_docs=150000, tag="indicsynth",
        work_dir=Path("/content/work"),
        drive_dir=Path("/content/drive/MyDrive/doclayout-yolo-indic/shards"),
        shard_size=5000, workers=0)   # workers=0 -> all vCPUs

Cell 5 usage (assemble for training, also resumable/idempotent):
    from src.synthetic_data.resumable import assemble_from_shards
    coco, images = assemble_from_shards(
        tag="indicsynth",
        drive_dir=Path("/content/drive/MyDrive/doclayout-yolo-indic/shards"),
        out_root=Path("/content/IndicSynth"))
"""

import json
import math
import os
import shutil
import tarfile
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

from src.config import SYNTHETIC_IMG_SIZE, RANDOM_SEED
from src.logger import get_logger
from src.synthetic_data.parallel import _worker  # reuse the page-generating worker
from src.synthetic_data.coco_formatter import COCOBuilder

logger = get_logger(__name__)


def _shard_tar(drive_dir: Path, tag: str, k: int) -> Path:
    return drive_dir / f"{tag}_shard_{k:04d}.tar"


def _is_valid_tar(path: Path) -> bool:
    try:
        with tarfile.open(path, "r") as tf:
            tf.getmembers()
        return True
    except Exception:  # noqa: BLE001
        return False


def generate_resumable(num_docs: int, tag: str, work_dir: Path, drive_dir: Path,
                       shard_size: int = 5000, workers: int = 0,
                       chunk: int = 1000, cleanup_local: bool = True) -> Path:
    workers = workers or (os.cpu_count() or 1)
    work_dir = Path(work_dir)
    drive_dir = Path(drive_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    drive_dir.mkdir(parents=True, exist_ok=True)
    W, H = SYNTHETIC_IMG_SIZE

    # clean any half-copied shard tars from a previous crash
    for p in drive_dir.glob(f".{tag}_shard_*.partial"):
        p.unlink()

    n_shards = math.ceil(num_docs / shard_size)
    logger.info(f"{num_docs} docs -> {n_shards} shards of {shard_size}, "
                f"{workers} workers, shards on {drive_dir}")

    done = 0
    for k in range(n_shards):
        start = k * shard_size + 1
        end = min((k + 1) * shard_size, num_docs) + 1   # exclusive
        tar_path = _shard_tar(drive_dir, tag, k)

        if tar_path.exists() and _is_valid_tar(tar_path):
            done += 1
            logger.info(f"shard {k:04d} [{start}-{end-1}] already on Drive — skip")
            continue
        if tar_path.exists():  # exists but corrupt -> redo
            logger.warning(f"shard {k:04d} corrupt — regenerating")
            tar_path.unlink()

        shard_dir = work_dir / f"{tag}_shard_{k:04d}"
        img_dir = shard_dir / "images"
        ann_dir = shard_dir / "annotations"
        shutil.rmtree(shard_dir, ignore_errors=True)
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)

        # chunk tasks; seed by ABSOLUTE chunk index so reruns are identical
        tasks = []
        for cs in range(start, end, chunk):
            ce = min(cs + chunk, end)
            seed = RANDOM_SEED * 1000 + (cs - 1) // chunk
            tasks.append((cs, ce, seed, tag, str(img_dir), str(ann_dir)))

        builder = COCOBuilder()
        manifest = []
        with Pool(workers) as pool:
            for result in pool.imap_unordered(_worker, tasks):
                for doc_id, fname, regions, script in result:
                    builder.add_image(doc_id, fname, W, H)
                    for cls, bbox in regions:
                        builder.add_annotation(doc_id, cls, bbox, W, H)
                    manifest.append({"image_id": doc_id, "file_name": fname,
                                     "script": script})
        builder.write(shard_dir / f"{tag}_shard_{k:04d}_coco.json")
        (shard_dir / f"{tag}_shard_{k:04d}_manifest.json").write_text(
            json.dumps(manifest))

        # tar locally, then copy to Drive under a .partial name and rename
        local_tar = work_dir / f"{tag}_shard_{k:04d}.tar"
        with tarfile.open(local_tar, "w") as tf:
            tf.add(shard_dir, arcname=shard_dir.name)
        partial = drive_dir / f".{tag}_shard_{k:04d}.partial"
        shutil.copy(local_tar, partial)
        partial.rename(tar_path)        # appears "done" only when fully written
        local_tar.unlink()
        if cleanup_local:
            shutil.rmtree(shard_dir, ignore_errors=True)

        done += 1
        logger.info(f"shard {k:04d} [{start}-{end-1}] -> Drive "
                    f"({tar_path.stat().st_size/1e6:.0f} MB)  [{done}/{n_shards}]")

    logger.info(f"All {n_shards} shards present on Drive ✓")
    return drive_dir


def assemble_from_shards(tag: str, drive_dir: Path, out_root: Path) -> Tuple[Path, Path]:
    """Extract all shard tars and merge into one COCO json + flat images dir.

    Idempotent: safe to re-run. Returns (coco_json_path, images_dir).
    """
    drive_dir = Path(drive_dir)
    out_root = Path(out_root)
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_root / "_extract"

    merged = {"images": [], "annotations": [], "categories": None}
    manifest = []
    ann_id = 1

    shard_tars = sorted(drive_dir.glob(f"{tag}_shard_*.tar"))
    if not shard_tars:
        raise FileNotFoundError(f"No shard tars in {drive_dir}")
    logger.info(f"Assembling {len(shard_tars)} shards -> {out_root}")

    for tar_path in shard_tars:
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(tmp)
        shard_root = next(tmp.iterdir())          # the <tag>_shard_xxxx dir
        # move images
        for img in (shard_root / "images").glob("*.png"):
            dst = images_dir / img.name
            if not dst.exists():
                shutil.move(str(img), str(dst))
        # merge coco
        coco = json.loads(next(shard_root.glob("*_coco.json")).read_text())
        if merged["categories"] is None:
            merged["categories"] = coco["categories"]
        merged["images"].extend(coco["images"])
        for a in coco["annotations"]:
            a["id"] = ann_id; ann_id += 1
            merged["annotations"].append(a)
        mf = next(shard_root.glob("*_manifest.json"), None)
        if mf:
            manifest.extend(json.loads(mf.read_text()))

    shutil.rmtree(tmp, ignore_errors=True)
    coco_path = out_root / f"{tag}_coco.json"
    coco_path.write_text(json.dumps(merged))
    (out_root / f"{tag}_manifest.json").write_text(json.dumps(manifest))
    logger.info(f"Merged: {len(merged['images'])} images, "
                f"{len(merged['annotations'])} annotations -> {coco_path.name}")
    return coco_path, images_dir
