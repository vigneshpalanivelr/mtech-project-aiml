"""
Pretraining config + COCO->YOLO dataset preparation (Phase 2, Step 7).

The conversion and data.yaml generation run on CPU (useful to prepare here).
The actual training launch requires torch + ultralytics + a GPU and the public
DocLayout-YOLO checkpoint, so it is guarded and will report what is missing.

Prepare YOLO labels from the synthetic COCO json (CPU):
    python -m src.pretraining.train_synthetic --prepare --tag smoke

Launch pretraining (GPU box, after `pip install -r requirements.txt`):
    python -m src.pretraining.train_synthetic --train \
        --coco output/synthetic/indicsynth_coco.json \
        --weights <path-to-DocLayout-YOLO.pt>
"""

import argparse
import json
import os
from pathlib import Path

# reduce CUDA fragmentation (set before torch initializes CUDA). Helps avoid the
# "reserved but unallocated" OOM pattern at large image sizes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.config import (COCO_CLASSES, NUM_EPOCHS, BATCH_SIZE, IMG_RES,
                        LEARNING_RATE)
from src.logger import get_logger
from src.utils.paths import get_output_dir, get_synthetic_dirs, get_checkpoint_path

logger = get_logger(__name__)


def coco_to_yolo(coco_json: Path, images_dir: Path, out_root: Path,
                 val_frac: float = 0.05) -> Path:
    """Convert a COCO detection json to Ultralytics YOLO layout.

    Writes out_root/{images,labels}/{train,val} and a data.yaml.
    YOLO label line: <class> <xc> <yc> <w> <h>  (all normalized 0-1).
    """
    data = json.loads(Path(coco_json).read_text())
    images = {im["id"]: im for im in data["images"]}
    anns_by_img = {}
    for a in data["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    ids = sorted(images)
    n_val = max(1, int(len(ids) * val_frac))
    val_ids = set(ids[:n_val])

    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    import shutil
    total_dropped = 0
    for img_id in ids:
        im = images[img_id]
        split = "val" if img_id in val_ids else "train"
        W, H = im["width"], im["height"]
        src_img = images_dir / im["file_name"]
        dst_img = out_root / "images" / split / im["file_name"]
        if src_img.exists() and not dst_img.exists():
            shutil.copy(src_img, dst_img)
        lines = []
        for a in anns_by_img.get(img_id, []):
            x, y, w, h = a["bbox"]
            # clamp to image bounds, then drop degenerate (near-zero-area) boxes.
            # A zero-area box makes the CIoU box loss evaluate to inf during
            # training (AMP then skips the step), so it must not be written.
            x0 = max(0.0, min(float(x), W)); y0 = max(0.0, min(float(y), H))
            x1 = max(0.0, min(float(x) + float(w), W))
            y1 = max(0.0, min(float(y) + float(h), H))
            bw, bh = x1 - x0, y1 - y0
            if bw < 1.0 or bh < 1.0:        # below 1px on a side -> degenerate
                total_dropped += 1
                continue
            xc, yc = (x0 + bw / 2) / W, (y0 + bh / 2) / H
            lines.append(f"{a['category_id']} {xc:.6f} {yc:.6f} "
                         f"{bw / W:.6f} {bh / H:.6f}")
        label_path = (out_root / "labels" / split /
                      im["file_name"].replace(".png", ".txt"))
        label_path.write_text("\n".join(lines), encoding="utf-8")
    if total_dropped:
        logger.info(f"coco_to_yolo: dropped {total_dropped} degenerate (zero-area) "
                    f"boxes to prevent inf box loss")

    yaml_path = out_root / "data.yaml"
    names = "\n".join(f"  {cid}: {name}" for cid, name in COCO_CLASSES.items())
    yaml_path.write_text(
        f"path: {out_root.resolve()}\n"
        f"train: images/train\nval: images/val\n\n"
        f"names:\n{names}\n", encoding="utf-8")
    logger.info(f"YOLO dataset ready: {len(ids)} imgs "
                f"({len(val_ids)} val) -> {out_root}")
    logger.info(f"data.yaml -> {yaml_path}")
    return yaml_path


# The correct Phase 2 initializer per the methodology (Sec 5.3): the pure
# DocSynth300K-pretrained backbone -- NOT a D4LA/DocLayNet-finetuned checkpoint.
DOCSYNTH_REPO = "juliozhao/DocLayout-YOLO-DocSynth300K-pretrain"
DOCSYNTH_FILE = "doclayout_yolo_docsynth300k_imgsz1600.pt"


def download_base_checkpoint(dest_dir: str = "/content") -> str:
    """Fetch the DocSynth300K-pretrained checkpoint from HuggingFace."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=DOCSYNTH_REPO, filename=DOCSYNTH_FILE,
                           local_dir=dest_dir)
    logger.info(f"Base checkpoint -> {path}")
    return path


def _disable_wandb() -> None:
    """Disable Weights & Biases logging.

    The doclayout_yolo fork passes the filesystem save path as the W&B project
    name (W&B forbids '/'), which crashes the run before training. We log via
    TensorBoard instead, so W&B is turned off at the env + settings level.
    """
    import os
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    try:
        from doclayout_yolo.utils import SETTINGS
        SETTINGS.update({"wandb": False})
    except Exception:  # noqa: BLE001
        pass


def _ensure_amp_asset() -> bool:
    """Provide yolov8n.pt for the fork's AMP check.

    The fork builds a broken download URL for the AMP self-test asset
    (github.com/doclayout_yolo/assets/... which 404s). We fetch the real
    Ultralytics asset into the CWD, where the check looks before downloading,
    so AMP stays enabled. Returns False (caller disables AMP) if unreachable.
    """
    import urllib.request
    dst = Path.cwd() / "yolov8n.pt"
    if dst.exists():
        return True
    url = ("https://github.com/ultralytics/assets/releases/download/"
           "v8.1.0/yolov8n.pt")
    try:
        urllib.request.urlretrieve(url, dst)
        logger.info(f"Fetched AMP-check asset -> {dst}")
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not fetch yolov8n.pt ({e}); training with amp=False.")
        return False


def _attach_clean_logging(model) -> None:
    """Attach callbacks that print a tidy, human-readable per-epoch summary
    through the project logger, on top of the framework's own table."""
    import time
    state = {"t0": None}

    def _hm(secs: float) -> str:
        secs = int(secs)
        return f"{secs // 3600}h{(secs % 3600) // 60:02d}m"

    def on_start(trainer):
        state["t0"] = time.time()
        logger.info("=" * 64)
        logger.info(f"PRETRAINING START | {trainer.args.epochs} epochs | "
                    f"batch {trainer.args.batch} | imgsz {trainer.args.imgsz} | "
                    f"device {trainer.device}")
        logger.info(f"save_dir: {trainer.save_dir}")
        logger.info("loss legend: box/cls/dfl = box, classification, "
                    "distribution-focal; *_om = one-to-many head, "
                    "*_oo = one-to-one head (YOLOv10 dual heads)")
        logger.info("=" * 64)

    def on_epoch(trainer):
        e, E = trainer.epoch + 1, trainer.epochs
        m = trainer.metrics or {}
        def g(k):
            return m.get(k, float("nan"))
        elapsed = time.time() - (state["t0"] or time.time())
        eta = (elapsed / e) * (E - e)
        logger.info(
            f"epoch {e:>3}/{E} | "
            f"mAP50 {g('metrics/mAP50(B)'):.3f} | "
            f"mAP50-95 {g('metrics/mAP50-95(B)'):.3f} | "
            f"P {g('metrics/precision(B)'):.3f} R {g('metrics/recall(B)'):.3f} | "
            f"elapsed {_hm(elapsed)} | ETA {_hm(eta)}")

    def on_end(trainer):
        logger.info("=" * 64)
        logger.info(f"PRETRAINING DONE in {_hm(time.time() - (state['t0'] or 0))} | "
                    f"best weights: {trainer.save_dir}/weights/best.pt")
        logger.info("=" * 64)

    model.add_callback("on_pretrain_routine_end", on_start)
    model.add_callback("on_fit_epoch_end", on_epoch)
    model.add_callback("on_train_end", on_end)


def train(data_yaml: Path, weights: str, resume: bool = True):
    """Launch (or resume) pretraining. Requires GPU + ultralytics + checkpoint.

    If a previous run's last.pt exists in the save dir (on Drive), training
    resumes from it automatically — so a Colab disconnect mid-run costs only the
    current epoch, not the whole job. Pass resume=False to force a fresh run.
    """
    import shutil
    try:
        import torch
        from doclayout_yolo import YOLOv10
    except ImportError as e:  # noqa: BLE001
        logger.error(f"Training deps missing ({e}). On the GPU box run: "
                     "pip install -r requirements.txt && pip install "
                     "git+https://github.com/opendatalab/DocLayout-YOLO.git")
        return

    _disable_wandb()                      # avoid the W&B project-name crash

    # PyTorch >=2.6 defaults torch.load to weights_only=True, which rejects the
    # fork's full-model checkpoints in the end-of-training validation reload
    # (harmless UnpicklingError printed AFTER 'epochs completed'). Our
    # checkpoints are trusted, so restore the old load behavior.
    if not getattr(torch.load, "_wo_patched", False):
        _orig_load = torch.load
        def _compat_load(*a, **k):       # noqa: ANN001
            k.setdefault("weights_only", False)
            return _orig_load(*a, **k)
        _compat_load._wo_patched = True
        torch.load = _compat_load

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("No CUDA detected -- pretraining 150K on CPU is "
                       "impractical. Use a Colab A100 runtime.")
    if not Path(weights).exists():
        logger.error(f"Base checkpoint not found: {weights}. Run with "
                     "--download-ckpt to fetch DocSynth300K-pretrain.")
        return

    amp_ok = _ensure_amp_asset()          # keep AMP if asset reachable, else off

    # quiet the framework's noisy (harmless) deprecation/augmentation warnings
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*quality_lower.*")
    warnings.filterwarnings("ignore", message=".*no transform to process.*")

    project = get_output_dir("logs")
    save_dir = project / "indicsynth_pretrain"
    last_pt = save_dir / "weights" / "last.pt"

    resuming = resume and last_pt.exists()
    model = YOLOv10(str(last_pt) if resuming else weights)  # nc auto-adapts to 9
    _attach_clean_logging(model)
    try:
        if resuming:
            logger.info(f"Resuming from {last_pt} (disconnect-safe).")
            model.train(resume=True)      # reuses the saved args
        else:
            model.train(
                data=str(data_yaml), epochs=NUM_EPOCHS, batch=BATCH_SIZE,
                imgsz=IMG_RES, lr0=LEARNING_RATE, cos_lr=True, device=device,
                amp=amp_ok, save_period=5, exist_ok=True,
                project=str(project), name="indicsynth_pretrain",
            )
    except AssertionError as e:           # e.g. "nothing to resume" if complete
        logger.info(f"Resume reported nothing to do ({e}); treating as complete.")

    # save the best checkpoint to a stable path (copy, don't rely on model.save)
    best = save_dir / "weights" / "best.pt"
    out = get_checkpoint_path("doclayout_yolo_indic_pretrained.pt")
    if best.exists():
        shutil.copy(best, out)
        logger.info(f"Saved best checkpoint -> {out}")
    else:
        logger.warning(f"best.pt not found at {best}; check the training logs. "
                       f"last.pt (if any) is at {last_pt}.")
    # NOTE: the 9-way auxiliary script-classification head (config.SCRIPT_HEAD_
    # LABELS, Angle C) attaches to the GL-CRM bottleneck. Ultralytics' stock
    # trainer has no second head, so wiring L = L_det + 0.1*L_script requires a
    # custom trainer subclass -- planned for Week 7 per the 12-week plan.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--coco", default=None)
    ap.add_argument("--weights", default="/content/" + DOCSYNTH_FILE)
    ap.add_argument("--download-ckpt", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any last.pt and start training from scratch")
    args = ap.parse_args()

    if args.download_ckpt:
        args.weights = download_base_checkpoint()

    dirs = get_synthetic_dirs()
    coco = Path(args.coco) if args.coco else dirs["root"] / f"{args.tag}_coco.json"
    out_root = get_output_dir("yolo_dataset")

    if args.prepare or args.train:
        yaml_path = coco_to_yolo(coco, dirs["images"], out_root)
    if args.train:
        train(yaml_path, args.weights, resume=not args.fresh)
    if not (args.prepare or args.train):
        ap.print_help()


if __name__ == "__main__":
    main()
