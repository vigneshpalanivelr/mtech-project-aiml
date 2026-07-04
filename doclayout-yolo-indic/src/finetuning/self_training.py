"""
Phase 3 — Self-Training with Class-Balanced Self-Training (CBST).

This module closes the synthetic-to-real distribution gap. It takes the Phase 2
pretrained DocLayout-YOLO-Indic checkpoint, runs it over BaDLAD-unlabeled real
documents, keeps confident detections as *pseudo-labels*, and re-trains on a
mix of {real labeled + pseudo-labeled} data. The pseudo-labeling threshold is
chosen PER CLASS (CBST, Zou et al., ECCV 2018) so that rare classes are not
swamped by the dominant text class.

It operates in whatever class space the loaded model has (9 synthetic classes
for the Phase 2 checkpoint). BaDLAD-unlabeled is treated as genuinely unlabeled.
Re-alignment to the IndicDLP ontology happens later in `train_finetuning.py`.

Design goals (match the rest of the project):
  - Resumable: every round and every epoch checkpoints to Drive; skip-if-exists.
  - A100-only training; CPU/T4 fine for the inference pass.
  - No notebook-cell patches: all logic lives here, the notebook just calls it.

Example:
    >>> from src.finetuning.self_training import run_self_training
    >>> run_self_training(
    ...     pretrained="output/checkpoints/doclayout_yolo_indic_pretrained.pt",
    ...     unlabeled_dir="data/raw/BaDLAD/unlabeled/images",
    ...     labeled_yaml="data/raw/BaDLAD/badlad_labeled.yaml",
    ...     work_dir="output/self_training",
    ...     num_rounds=2,
    ... )
"""

import gc
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# BaDLAD contains some very large scans (90+ megapixels). These are legitimate
# documents, not attacks, so lift PIL's decompression-bomb guard. We bound the
# decode cost ourselves by pre-shrinking oversized pages in collect_predictions.
Image.MAX_IMAGE_PIXELS = None

# Project logger if available, else a stdlib fallback so the file runs standalone.
try:
    from src.logger import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    logger = logging.getLogger(__name__)


# The doclayout_yolo (Ultralytics fork) ships a Weights & Biases callback that
# auto-calls wandb.init(project=<ultralytics output path>). That path contains
# '/', which W&B rejects as a project name, crashing training before epoch 1.
# Disable the W&B integration up front, before any trainer registers callbacks.
import os
os.environ.setdefault("WANDB_MODE", "disabled")
try:
    from doclayout_yolo.utils import SETTINGS as _YOLO_SETTINGS
    if _YOLO_SETTINGS.get("wandb") is not False:
        _YOLO_SETTINGS.update({"wandb": False})
except Exception:  # package not importable in a bare inspection environment
    pass


# PyTorch 2.6+ changed torch.load's default to weights_only=True. The old
# doclayout_yolo fork calls torch.load without that argument in strip_optimizer(),
# so finalising a trained checkpoint crashes on torch >= 2.6. Every checkpoint we
# load is our own (trusted), so restore the pre-2.6 default of weights_only=False.
try:
    import torch as _torch
    if not getattr(_torch.load, "_wo_patched", False):
        _orig_torch_load = _torch.load

        def _torch_load(*a, **k):
            k.setdefault("weights_only", False)
            return _orig_torch_load(*a, **k)

        _torch_load._wo_patched = True
        _torch.load = _torch_load
except Exception:
    pass


# The G2L_CRM (GL-CRM) block in this doclayout_yolo fork reads self.dcv.bn inside
# its forward pass. Model fusion folds Conv+BN together and removes .bn, so the
# fused inference path crashes ("'Conv' object has no attribute 'bn'") during the
# post-training validation. Make fuse() a no-op: validation then runs on the
# unfused model, which is numerically identical, just marginally slower.
try:
    from doclayout_yolo.nn.tasks import BaseModel as _BaseModel
    if not getattr(_BaseModel.fuse, "_noop_patched", False):
        def _fuse_noop(self, *a, **k):
            return self

        _fuse_noop._noop_patched = True
        _BaseModel.fuse = _fuse_noop
except Exception:
    pass


# ---------------------------------------------------------------------------
# CBST hyper-parameters
# ---------------------------------------------------------------------------
# Portion p_c of each class's detections to keep, by round (self-paced schedule).
# Round 1 keeps the top 20% most-confident detections per class; round 2 relaxes
# to 30% because the model is now better and we trust more of its predictions.
PORTION_SCHEDULE = [0.20, 0.30]

# Hard confidence floor. Even if the portion schedule would admit a detection,
# never accept anything below this — it is almost certainly noise.
CONF_FLOOR = 0.40

# Inference settings for the pseudo-labeling pass.
INFER_CONF = 0.25             # collect everything above this, then CBST-filter on top
INFER_IMGSZ = 1024
INFER_BATCH = 16              # chunk size; peak RAM bounded because pages are pre-shrunk
INFER_PIXEL_CAP = 50_000_000  # pages above ~50 MP get pre-shrunk before inference
INFER_PRESHRINK_SIDE = 2048   # cap long side when pre-shrinking (still >> imgsz=1024)

# Self-training fit settings (A100-40GB).
TRAIN_EPOCHS = 20
TRAIN_IMGSZ = 1024
TRAIN_BATCH = 16
TRAIN_LR0 = 5e-4      # lower than pretraining: we are adapting, not learning fresh
TRAIN_LRF = 0.01      # cosine final-LR fraction
TRAIN_PATIENCE = 5    # early-stop if val mAP stalls


# ---------------------------------------------------------------------------
# 1. Inference: collect raw detections over the unlabeled set
# ---------------------------------------------------------------------------
def load_detector(weights_path):
    """Load a DocLayout-YOLO (YOLOv10) detector. Imported lazily so this file
    can be inspected without the heavy package installed."""
    from doclayout_yolo import YOLOv10
    weights_path = str(weights_path)
    model = YOLOv10(weights_path)
    names = model.names  # {idx: name}
    logger.info("Loaded detector %s | %d classes -> %s",
                Path(weights_path).name, len(names), list(names.values()))
    return model, names


def _safe_source_path(path, tmp_dir):
    """Return a path safe to feed the detector.

    Reads the image size lazily (header only, no full pixel decode); if the page
    exceeds INFER_PIXEL_CAP it is pre-shrunk to a temp JPEG so a single giant scan
    can't blow up memory or the native decoder. 2048 px is still well above the
    1024 inference size, so detection quality is unaffected. Returns None for
    unreadable/corrupt images so the caller skips them instead of crashing.
    """
    try:
        with Image.open(path) as im:
            w, h = im.size                       # lazy: header only
            if w * h <= INFER_PIXEL_CAP:
                return path
            im = im.convert("RGB")
            s = INFER_PRESHRINK_SIDE / max(w, h)
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
            out = Path(tmp_dir) / (Path(path).stem + ".jpg")
            im.save(out, "JPEG", quality=90)
            return str(out)
    except Exception as e:
        logger.warning("skipping unreadable image %s (%s)", Path(path).name, e)
        return None


def _predict_paths(model, src_paths, conf, imgsz, device):
    return model.predict(source=src_paths, conf=conf, imgsz=imgsz,
                         device=device, stream=False, verbose=False)


def collect_predictions(model, image_dir, conf=INFER_CONF, imgsz=INFER_IMGSZ,
                        batch=INFER_BATCH, device=0, limit=None, log_every=1000,
                        exclude_stems=None):
    """Run inference over `image_dir` and return per-image detections.

    Hardened for large, messy, real-world datasets:
      - pre-shrinks oversized pages so decode memory stays bounded,
      - skips corrupt/unreadable images instead of crashing,
      - on a batch failure, retries one image at a time so a single bad page
        costs one skip rather than the whole batch,
      - logs progress so any failure point is visible,
      - frees GPU cache periodically to avoid creep over a long sweep,
      - `exclude_stems`: image stems to skip (the labeled + val images), so the
        held-out val never enters the pseudo-labeled training pool (no leakage).

    Boxes are stored normalised (xywhn), so they are valid for the ORIGINAL image
    regardless of any pre-shrink. Returns:
        [{"image": <original abs path>, "dets": [{"cls","conf","xywhn"}, ...]}, ...]
    """
    import torch
    image_dir = Path(image_dir)
    images = sorted(p for p in image_dir.rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"})
    if exclude_stems:
        before = len(images)
        images = [p for p in images if p.stem not in exclude_stems]
        logger.info("Excluded %d labeled/val images from the unlabeled pool",
                    before - len(images))
    if limit:
        images = images[:limit]
    logger.info("Pseudo-label inference over %d images from %s", len(images), image_dir)

    records, n_dets, n_skip, done = [], 0, 0, 0
    tmp_dir = tempfile.mkdtemp(prefix="preshrink_")

    def _consume(results, origs):
        nonlocal n_dets
        for r, orig in zip(results, origs):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xywhn = b.xywhn.cpu().numpy()
            cf = b.conf.cpu().numpy()
            cl = b.cls.cpu().numpy().astype(int)
            dets = [{"cls": int(c), "conf": float(s), "xywhn": [float(v) for v in bb]}
                    for bb, s, c in zip(xywhn, cf, cl)]
            records.append({"image": str(orig), "dets": dets})   # keep ORIGINAL path
            n_dets += len(dets)

    try:
        for start in range(0, len(images), batch):
            chunk = images[start:start + batch]
            src, origs = [], []
            for p in chunk:
                sp = _safe_source_path(str(p), tmp_dir)
                if sp is None:
                    n_skip += 1
                else:
                    src.append(sp)
                    origs.append(p)
            if src:
                try:
                    _consume(_predict_paths(model, src, conf, imgsz, device), origs)
                except Exception as e:                       # isolate the bad page
                    logger.warning("batch @%d failed (%s); retrying per-image", start, e)
                    torch.cuda.empty_cache()
                    for sp, orig in zip(src, origs):
                        try:
                            _consume(_predict_paths(model, [sp], conf, imgsz, device), [orig])
                        except Exception as e2:
                            n_skip += 1
                            logger.warning("  dropped %s (%s)", Path(orig).name, e2)
            done += len(chunk)
            if done % log_every < batch:
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("  ...%d/%d images | %d dets | %d skipped",
                            done, len(images), n_dets, n_skip)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("Collected %d detections over %d images (%d skipped)",
                n_dets, len(records), n_skip)
    return records


# ---------------------------------------------------------------------------
# 2. CBST: class-balanced thresholds
# ---------------------------------------------------------------------------
def compute_cbst_thresholds(records, num_classes, portion, conf_floor=CONF_FLOOR):
    """Class-Balanced Self-Training thresholds (Zou et al., ECCV 2018).

    For each class c, keep the top `portion` fraction of that class's detections
    by confidence. Concretely, sort class c's confidences descending and set the
    threshold at the value sitting at the `portion`-quantile. Because the SAME
    fraction is kept for every class, selection rate is equalised across classes:
    the dominant text class does not drown out rare classes like sidebar or
    pull_quote. A hard `conf_floor` guards against accepting noise when a class
    is uniformly low-confidence.

    Returns: {class_id: threshold_float}
    """
    by_class = defaultdict(list)
    for rec in records:
        for d in rec["dets"]:
            by_class[d["cls"]].append(d["conf"])

    thresholds = {}
    for c in range(num_classes):
        scores = sorted(by_class.get(c, []), reverse=True)
        if not scores:
            # Class never predicted: set an unreachable threshold (admit nothing).
            thresholds[c] = 1.01
            continue
        # Index of the cut-off: keep the top `portion` of detections.
        k = max(1, int(len(scores) * portion))
        tau = scores[k - 1]               # confidence at the portion boundary
        thresholds[c] = max(tau, conf_floor)
    return thresholds


def filter_with_thresholds(records, thresholds):
    """Keep only detections whose confidence clears their class threshold.
    Returns filtered records and a per-class kept/total stat dict."""
    kept_records, total, kept = [], defaultdict(int), defaultdict(int)
    for rec in records:
        keep = []
        for d in rec["dets"]:
            total[d["cls"]] += 1
            if d["conf"] >= thresholds.get(d["cls"], 1.01):
                keep.append(d)
                kept[d["cls"]] += 1
        if keep:
            kept_records.append({"image": rec["image"], "dets": keep})
    stats = {int(c): {"kept": kept[c], "total": total[c]} for c in sorted(total)}
    return kept_records, stats


# ---------------------------------------------------------------------------
# 3. Write pseudo-labels in YOLO format + build the mixed dataset yaml
# ---------------------------------------------------------------------------
def write_pseudo_labels(kept_records, pseudo_root, names):
    """Write pseudo-labels (YOLO .txt) to `pseudo_root/labels/` on persistent
    storage (Drive). Images are NOT linked here: Drive's FUSE mount does not
    follow symlinks, and the source image cache is wiped between sessions, so any
    link would dangle. The self-contained image+label dataset is built on local
    disk at train time by `materialize_local_dataset`.
    """
    pseudo_root = Path(pseudo_root)
    lbl_dir = pseudo_root / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for rec in kept_records:
        stem = Path(rec["image"]).stem
        lines = [f"{d['cls']} " + " ".join(f"{v:.6f}" for v in d["xywhn"])
                 for d in rec["dets"]]
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
        n += 1
    logger.info("Wrote %d pseudo-label files -> %s", n, lbl_dir)
    return pseudo_root


# Cap for the local training copies. Well above imgsz=1024, so the model loses
# nothing; labels are normalised, so downscaling never touches them. Smaller
# images also make every training epoch decode faster.
MATERIALIZE_MAX_SIDE = 1536


def _sync_dataset(src_root, dst_root):
    """Copy {images,labels} between two dataset roots, skipping files that already
    exist. Used to move the small downscaled dataset between Drive (persistent
    cache) and local disk (fast training)."""
    src_root, dst_root = Path(src_root), Path(dst_root)
    n = 0
    for sub in ("images", "labels"):
        (dst_root / sub).mkdir(parents=True, exist_ok=True)
        for f in (src_root / sub).glob("*"):
            d = dst_root / sub / f.name
            if not d.exists():
                shutil.copy(f, d)
                n += 1
    return n


def materialize_local_dataset(labels_dir, source_image_dir, local_root,
                              persist_root=None, max_side=MATERIALIZE_MAX_SIDE):
    """Produce a self-contained {images, labels} dataset on LOCAL disk for training.

    To avoid re-downloading the 25 GB BaDLAD set every session, the small
    downscaled dataset (~0.2 MB/image) is cached on `persist_root` (Drive):
      - if that cache already looks complete, it is just copied to local disk and
        the heavy BaDLAD source is never touched (no Kaggle download needed);
      - otherwise it is built from `source_image_dir` once, then pushed up to
        `persist_root` so every later session skips the download.

    YOLO needs images/ and labels/ as siblings with the files physically present;
    we pair each persistent pseudo-label with its source image by filename stem.
    Oversized scans are downscaled to `max_side` — lossless for the normalised
    boxes. Returns the local images dir to point the dataset yaml at.
    """
    labels_dir = Path(labels_dir)
    local_root = Path(local_root)
    n_labels = len(list(labels_dir.glob("*.txt")))

    # Fast path: a complete Drive cache exists -> copy to local, skip BaDLAD.
    if persist_root is not None:
        persist_root = Path(persist_root)
        cached = list((persist_root / "images").glob("*.jpg")) \
            if (persist_root / "images").exists() else []
        if cached and len(cached) >= 0.95 * max(n_labels, 1):
            logger.info("Reusing Drive-cached dataset (%d imgs) -> copying to local",
                        len(cached))
            _sync_dataset(persist_root, local_root)
            return local_root / "images"

    # Build from the BaDLAD source (needed only the first time per round).
    img_out = local_root / "images"
    lbl_out = local_root / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    index = {}
    for p in Path(source_image_dir).rglob("*"):
        if p.suffix.lower() in exts:
            index.setdefault(p.stem, p)
    logger.info("Indexed %d source images for materialisation", len(index))

    n, miss = 0, 0
    for lbl in labels_dir.glob("*.txt"):
        src = index.get(lbl.stem)
        if src is None:
            miss += 1
            continue
        dst_img = img_out / (lbl.stem + ".jpg")
        if not dst_img.exists():
            try:
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    s = min(1.0, max_side / max(w, h))
                    if s < 1.0:
                        im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
                    im.save(dst_img, "JPEG", quality=90)
            except Exception as e:
                logger.warning("skip materialising %s (%s)", src.name, e)
                miss += 1
                continue
        dst_lbl = lbl_out / lbl.name
        if not dst_lbl.exists():
            shutil.copy(lbl, dst_lbl)
        n += 1
    logger.info("Materialised %d image/label pairs -> %s (%d source images missing)",
                n, local_root, miss)

    # Cache to Drive so future sessions skip the BaDLAD download entirely.
    if persist_root is not None:
        copied = _sync_dataset(local_root, persist_root)
        logger.info("Cached %d files to Drive for future sessions -> %s",
                    copied, persist_root)

    return img_out


def build_selftrain_yaml(out_yaml, labeled_yaml, pseudo_images_dir, names):
    """Write a YOLO data.yaml pointing at the (local, materialized) pseudo-labeled
    images and, optionally, the real labeled split. Ultralytics accepts a list of
    train dirs, so the two sources are sampled together.

    `labeled_yaml` is the existing BaDLAD-labeled data.yaml (real ground truth in
    the SAME class space as the model). Pass None to self-train on pseudo-labels
    only (pure unsupervised adaptation).
    """
    out_yaml = Path(out_yaml)
    pseudo_train = str(Path(pseudo_images_dir).resolve())

    train_dirs = [pseudo_train]
    val_path = None
    if labeled_yaml and Path(labeled_yaml).exists():
        import yaml
        base = yaml.safe_load(Path(labeled_yaml).read_text())
        base_root = Path(labeled_yaml).parent
        def _abs(p):
            p = Path(p)
            return str(p if p.is_absolute() else (base_root / p).resolve())
        if base.get("train"):
            train_dirs.insert(0, _abs(base["train"]))
        val_path = _abs(base["val"]) if base.get("val") else None

    names_list = [names[i] for i in range(len(names))]
    doc = {
        "train": train_dirs,
        "val": val_path or pseudo_train,   # fall back to pseudo if no real val
        "nc": len(names_list),
        "names": names_list,
    }
    import yaml
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(doc, sort_keys=False))
    logger.info("Self-training data.yaml -> %s\n%s", out_yaml, out_yaml.read_text())
    return out_yaml


# ---------------------------------------------------------------------------
# 4. One self-training round (resumable)
# ---------------------------------------------------------------------------
def self_train_round(weights, data_yaml, project, name, epochs=TRAIN_EPOCHS,
                     device=0):
    """Train one round from `weights` on `data_yaml`. Resumes from last.pt if a
    previous attempt was interrupted. Returns the path to this round's best.pt."""
    from doclayout_yolo import YOLOv10
    run_dir = Path(project) / name
    last = run_dir / "weights" / "last.pt"
    best = run_dir / "weights" / "best.pt"

    if best.exists():
        logger.info("Round '%s' already complete -> %s (skipping)", name, best)
        return best

    resume = last.exists()
    init = str(last) if resume else str(weights)
    logger.info("Round '%s': %s from %s",
                name, "RESUMING" if resume else "starting", Path(init).name)

    model = YOLOv10(init)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=TRAIN_IMGSZ,
        batch=TRAIN_BATCH,
        lr0=TRAIN_LR0,
        lrf=TRAIN_LRF,
        patience=TRAIN_PATIENCE,
        optimizer="AdamW",
        device=device,
        project=str(project),
        name=name,
        save_period=1,        # per-epoch checkpoints (Colab resilience)
        resume=resume,
        exist_ok=True,
        plots=True,
    )
    logger.info("Round '%s' done -> %s", name, best)
    return best


# ---------------------------------------------------------------------------
# 5. Orchestrator: pseudo-label -> train, repeated for N rounds
# ---------------------------------------------------------------------------
def run_self_training(pretrained, unlabeled_dir, labeled_yaml, work_dir,
                      num_rounds=2, device=0, infer_limit=None,
                      exclude_stems_file=None):
    """Full CBST loop. Each round: pseudo-label the unlabeled set with the
    current model, write a mixed dataset, train. The next round re-pseudo-labels
    with the improved model and a relaxed portion. Everything is skip-if-exists
    so a disconnected Colab session resumes cleanly.

    `exclude_stems_file`: JSON list of image stems (labeled + val) to skip during
    inference, so the held-out val never leaks into the pseudo-labeled pool.

    Returns the best.pt of the final round (input to fine-tuning).
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    current_weights = Path(pretrained)
    final_best = None

    exclude_stems = set()
    if exclude_stems_file and Path(exclude_stems_file).exists():
        exclude_stems = set(json.loads(Path(exclude_stems_file).read_text()))
        logger.info("Loaded %d exclude stems (labeled+val) from %s",
                    len(exclude_stems), exclude_stems_file)

    for rnd in range(1, num_rounds + 1):
        portion = PORTION_SCHEDULE[min(rnd - 1, len(PORTION_SCHEDULE) - 1)]
        round_dir = work_dir / f"round_{rnd}"
        round_dir.mkdir(parents=True, exist_ok=True)
        pseudo_root = round_dir / "pseudo"
        data_yaml = round_dir / "selftrain.yaml"
        stats_file = round_dir / "cbst_stats.json"
        best = round_dir / "train" / "weights" / "best.pt"

        logger.info("=== Self-training round %d/%d (portion=%.0f%%) ===",
                    rnd, num_rounds, portion * 100)

        if best.exists():
            logger.info("Round %d already trained -> %s (skipping)", rnd, best)
            current_weights, final_best = best, best
            continue

        # (a) ensure pseudo-labels exist on Drive (skip the 25-min sweep if so)
        labels_dir = pseudo_root / "labels"
        if not labels_dir.exists() or not any(labels_dir.glob("*.txt")):
            model, names = load_detector(current_weights)
            records = collect_predictions(model, unlabeled_dir, device=device,
                                          limit=infer_limit,
                                          exclude_stems=exclude_stems)
            thresholds = compute_cbst_thresholds(records, num_classes=len(names),
                                                 portion=portion)
            kept, stats = filter_with_thresholds(records, thresholds)
            write_pseudo_labels(kept, pseudo_root, names)
            stats_file.write_text(json.dumps(
                {"round": rnd, "portion": portion,
                 "thresholds": {int(k): round(v, 4) for k, v in thresholds.items()},
                 "per_class": stats,
                 "images_kept": len(kept)}, indent=2))
            logger.info("CBST round %d: kept %d images. Stats -> %s",
                        rnd, len(kept), stats_file)
            del model
        else:
            logger.info("Pseudo-labels for round %d already present -> reuse", rnd)
            _, names = load_detector(current_weights)

        # (b) materialize a self-contained dataset on LOCAL disk (Drive FUSE can't
        #     hold symlinks, and reads off the local SSD are far faster to train on)
        local_base = Path("/content/selftrain_local") if Path("/content").exists() \
            else work_dir / ".local_cache"
        pseudo_images = materialize_local_dataset(
            labels_dir, unlabeled_dir, local_base / f"round_{rnd}",
            persist_root=round_dir / "dataset")
        build_selftrain_yaml(data_yaml, labeled_yaml, pseudo_images, names)

        # (c) train this round
        best = self_train_round(current_weights, data_yaml,
                                project=str(round_dir), name="train", device=device)
        current_weights, final_best = best, best

    logger.info("Self-training complete. Final checkpoint: %s", final_best)
    return final_best


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Phase 3 CBST self-training")
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--unlabeled-dir", required=True)
    ap.add_argument("--labeled-yaml", default=None,
                    help="BaDLAD-labeled data.yaml in the model's class space; "
                         "omit for pseudo-only adaptation")
    ap.add_argument("--work-dir", default="output/self_training")
    ap.add_argument("--num-rounds", type=int, default=2)
    ap.add_argument("--device", default=0)
    ap.add_argument("--infer-limit", type=int, default=None,
                    help="cap #unlabeled images (smoke test)")
    args = ap.parse_args()
    run_self_training(args.pretrained, args.unlabeled_dir, args.labeled_yaml,
                      args.work_dir, args.num_rounds, args.device, args.infer_limit)
