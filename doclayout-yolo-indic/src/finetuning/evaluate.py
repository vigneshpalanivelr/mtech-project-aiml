"""
Phase 4 — Evaluate the final model on IndicDLP's TEST split.

Produces three things:
  1. Overall 42-class mAP on the held-out test split (the officially reported number).
  2. A per-SCRIPT breakdown (mAP for each language) — the core "cross-script"
     evidence for the thesis. IndicDLP filenames encode language as the 2nd
     underscore field, e.g. nv_ml_000221_0 -> Malayalam.
  3. English as one of those rows — this replaces the DocLayNet no-regression
     check, which is unrunnable here (final model uses IndicDLP's 42-class
     ontology, not DocLayNet's 11, and DocLayNet is ~80 GB). Measuring English
     on IndicDLP's own English pages is ontology-consistent and needs no download.

Usage (Phase 4 notebook):
    from src.finetuning.evaluate import run_evaluation
    run_evaluation(
        weights   = PROJECT_ROOT/'output'/'ablation_no_selftrain'/'finetune'/'weights'/'best.pt',
        hf_repo   = 'VigneshPR/IndicDLP',
        drive_out = PROJECT_ROOT/'data'/'indicdlp_test',
        local_root= '/content/indicdlp_test',
        out_dir   = PROJECT_ROOT/'output'/'phase4_eval',
    )
"""

import json
import os
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from src.finetuning import self_training as _st  # noqa: F401  (patches)
from src.finetuning.data_prep import coco_bbox_to_yolo, _materialize_one
from src.finetuning.train_finetuning import _extract_complete, _hf_get, read_classes

try:
    from src.logger import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    logger = logging.getLogger(__name__)

LANG_NAMES = {
    "as": "Assamese", "bn": "Bengali", "en": "English", "gu": "Gujarati",
    "hi": "Hindi", "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi",
    "or": "Odia", "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
}
_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MAX_SIDE = 1536
JPEG_QUALITY = 90


def _lang_of(stem):
    parts = stem.split("_")
    return parts[1] if len(parts) >= 2 else "unknown"


def stage_test(hf_repo, staging, token=None):
    """Download + extract IndicDLP test2017 (single tar). Idempotent."""
    staging = Path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    _hf_get(hf_repo, "annotations/instances_test2017.json", staging, token)
    tdir = staging / "test2017"
    if not (tdir.exists() and any(tdir.glob("*.jpg"))):
        tar = staging / "test2017.tar"
        if not tar.exists():
            logger.info("Downloading test2017.tar ...")
            tar = _hf_get(hf_repo, "test2017.tar", staging, token)
        logger.info("Extracting test2017 ...")
        _extract_complete(tar, staging)
        tar.unlink(missing_ok=True)
        shutil.rmtree(staging / ".cache", ignore_errors=True)
    logger.info("test2017: %d images", len(list(tdir.glob("*.jpg"))))
    return staging


def prepare_test(hf_repo, drive_out, local_root, cap=None, token=None, workers=None):
    """Build a local YOLO test set (images+labels), a data.yaml, and a
    stem->language index. Drive-caches the labels + yaml (images are light)."""
    workers = workers or os.cpu_count()
    drive_out = Path(drive_out)
    local_root = Path(local_root)
    drive_out.mkdir(parents=True, exist_ok=True)
    yaml_path = drive_out / "indicdlp_test.yaml"

    img_out = local_root / "images" / "test2017"
    lbl_out = local_root / "labels" / "test2017"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    staging = stage_test(hf_repo, "/content/IndicDLP_test_raw", token=token)
    coco = json.loads((staging / "annotations" / "instances_test2017.json").read_text())
    names, id_to_idx = read_classes(staging / "annotations" / "instances_test2017.json")

    ann_by_img = defaultdict(list)
    for a in coco["annotations"]:
        ann_by_img[a["image_id"]].append(a)

    index = {}
    for p in (staging / "test2017").rglob("*"):
        if p.suffix.lower() in _EXTS:
            index.setdefault(p.stem, p)

    images = [im for im in coco["images"] if Path(im["file_name"]).stem in index]
    if cap and len(images) > cap:
        import random
        random.Random(0).shuffle(images)
        images = images[:cap]

    tasks, lang_of = [], {}
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
        lang_of[stem] = _lang_of(stem)
        dst = img_out / f"{stem}.jpg"
        if not dst.exists():
            tasks.append((str(index[stem]), str(dst), MAX_SIDE, JPEG_QUALITY))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, _ in enumerate(ex.map(_materialize_one, tasks, chunksize=16), 1):
            if i % 2000 == 0:
                logger.info("  test: materialised %d/%d", i, len(tasks))
    logger.info("test set ready: %d images", len(lang_of))

    import yaml as _yaml
    yaml_path.write_text(_yaml.safe_dump({
        "train": str(img_out.resolve()), "val": str(img_out.resolve()),
        "nc": len(names), "names": list(names)}, sort_keys=False))
    (drive_out / "lang_of.json").write_text(json.dumps(lang_of))
    return yaml_path, names, lang_of, local_root


def _val(weights, data_yaml, device=0):
    from doclayout_yolo import YOLOv10
    m = YOLOv10(str(weights))
    r = m.val(data=str(data_yaml), split="val", device=device, verbose=False)
    return r


def run_evaluation(weights, hf_repo, drive_out, local_root, out_dir,
                   cap=None, device=0, token=None):
    """Overall test eval + per-script breakdown. Writes a JSON report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path, names, lang_of, local_root = prepare_test(
        hf_repo, drive_out, local_root, cap=cap, token=token)

    # ---- overall ----
    logger.info("=== Overall test-set evaluation (42 classes) ===")
    r = _val(weights, yaml_path, device)
    overall = {"mAP50": float(r.box.map50), "mAP50_95": float(r.box.map),
               "precision": float(r.box.mp), "recall": float(r.box.mr),
               "images": len(lang_of)}
    logger.info("OVERALL: mAP50=%.3f mAP50-95=%.3f P=%.3f R=%.3f",
                overall["mAP50"], overall["mAP50_95"], overall["precision"], overall["recall"])

    # ---- per-language: one val .txt per language, labels resolve via images->labels ----
    img_dir = local_root / "images" / "test2017"
    by_lang = defaultdict(list)
    for stem, lg in lang_of.items():
        by_lang[lg].append(str((img_dir / f"{stem}.jpg").resolve()))

    import yaml as _yaml
    per_lang = {}
    for lg, paths in sorted(by_lang.items()):
        if len(paths) < 20:            # too few to be a meaningful mAP
            logger.info("skip %s (%d imgs)", lg, len(paths))
            continue
        list_txt = local_root / f"test_{lg}.txt"
        list_txt.write_text("\n".join(paths))
        lang_yaml = local_root / f"test_{lg}.yaml"
        lang_yaml.write_text(_yaml.safe_dump({
            "train": str(list_txt), "val": str(list_txt),
            "nc": len(names), "names": list(names)}, sort_keys=False))
        rl = _val(weights, lang_yaml, device)
        per_lang[lg] = {"name": LANG_NAMES.get(lg, lg), "images": len(paths),
                        "mAP50": float(rl.box.map50), "mAP50_95": float(rl.box.map)}
        logger.info("  %-10s (%5d imgs): mAP50=%.3f mAP50-95=%.3f",
                    LANG_NAMES.get(lg, lg), len(paths), per_lang[lg]["mAP50"],
                    per_lang[lg]["mAP50_95"])

    report = {"weights": str(weights), "overall": overall, "per_language": per_lang}
    (out_dir / "phase4_report.json").write_text(json.dumps(report, indent=2))

    # pretty table
    print(f"\n{'script':12s} {'images':>7s} {'mAP50':>8s} {'mAP50-95':>9s}")
    print(f"{'OVERALL':12s} {overall['images']:>7d} {overall['mAP50']:>8.3f} {overall['mAP50_95']:>9.3f}")
    for lg, d in sorted(per_lang.items(), key=lambda kv: -kv[1]["mAP50_95"]):
        print(f"{d['name']:12s} {d['images']:>7d} {d['mAP50']:>8.3f} {d['mAP50_95']:>9.3f}")
    print(f"\nReport -> {out_dir/'phase4_report.json'}")
    return report
