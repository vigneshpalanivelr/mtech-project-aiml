"""
Parallel synthetic generation (multiprocessing) + tar packaging.

Single-core generation of 150K pages is ~20h; Colab A100 high-RAM runtimes
expose ~12 vCPUs, so this driver fans the work across workers and merges the
per-worker COCO records at the end. Output is written to LOCAL disk (pass an
images/annotations dir on /content, never Drive -- writing 150K small files to
Drive is slow and hits file-count limits). Use `pack_tar` to roll the result
into one archive for Drive or a HuggingFace dataset.

Run:
    python -m src.synthetic_data.parallel --num 150000 --tag indicsynth \
        --workers 8 --out /content/IndicSynth
"""

import argparse
import json
import os
import tarfile
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

from src.config import SYNTHETIC_IMG_SIZE, RANDOM_SEED
from src.logger import get_logger
from src.synthetic_data.generator import SyntheticDocumentGenerator
from src.synthetic_data.coco_formatter import COCOBuilder, write_per_image_record

logger = get_logger(__name__)


def _worker(task: Tuple[int, int, int, str, str, str]) -> List:
    start_id, end_id, seed, tag, img_dir, ann_dir = task
    gen = SyntheticDocumentGenerator(seed=seed)
    W, H = SYNTHETIC_IMG_SIZE
    img_dir, ann_dir = Path(img_dir), Path(ann_dir)
    out = []
    for doc_id in range(start_id, end_id):
        page, regions, script_label = gen.generate_one(doc_id)
        fname = f"{tag}_{doc_id:06d}.png"
        page.save(img_dir / fname)
        write_per_image_record(ann_dir / f"{tag}_{doc_id:06d}.json",
                               doc_id, fname, (W, H), regions)
        out.append((doc_id, fname, regions, script_label))
    return out


def generate_parallel(num_docs: int, tag: str, out_root: Path,
                      workers: int = 0, chunk: int = 2000) -> Path:
    workers = workers or (os.cpu_count() or 1)
    out_root = Path(out_root)
    img_dir = out_root / "images"
    ann_dir = out_root / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # build chunked tasks; each chunk gets an independent, reproducible seed
    tasks = []
    cid = 0
    for start in range(1, num_docs + 1, chunk):
        end = min(start + chunk, num_docs + 1)
        seed = RANDOM_SEED * 1000 + cid
        tasks.append((start, end, seed, tag, str(img_dir), str(ann_dir)))
        cid += 1
    logger.info(f"Generating {num_docs} docs with {workers} workers "
                f"in {len(tasks)} chunks -> {out_root}")

    builder = COCOBuilder()
    manifest = []
    W, H = SYNTHETIC_IMG_SIZE
    done = 0
    with Pool(workers) as pool:
        for result in pool.imap_unordered(_worker, tasks):
            for doc_id, fname, regions, script_label in result:
                builder.add_image(doc_id, fname, W, H)
                for cls, bbox in regions:
                    builder.add_annotation(doc_id, cls, bbox, W, H)
                manifest.append({"image_id": doc_id, "file_name": fname,
                                 "script": script_label})
            done += len(result)
            logger.info(f"  merged {done}/{num_docs}")

    builder.write(out_root / f"{tag}_coco.json")
    (out_root / f"{tag}_manifest.json").write_text(json.dumps(manifest))
    logger.info(f"Corpus complete: {num_docs} docs, "
                f"{len(builder.annotations)} annotations")
    return out_root


def pack_tar(out_root: Path, tar_path: Path) -> Path:
    """Roll the whole corpus into one tar (for Drive / HF upload)."""
    out_root, tar_path = Path(out_root), Path(tar_path)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tf:
        tf.add(out_root, arcname=out_root.name)
    logger.info(f"Packed -> {tar_path} ({tar_path.stat().st_size/1e9:.2f} GB)")
    return tar_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=150000)
    ap.add_argument("--tag", default="indicsynth")
    ap.add_argument("--out", default="/content/IndicSynth")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--tar", default="")
    args = ap.parse_args()
    root = generate_parallel(args.num, args.tag, Path(args.out),
                             args.workers, args.chunk)
    if args.tar:
        pack_tar(root, Path(args.tar))


if __name__ == "__main__":
    main()
