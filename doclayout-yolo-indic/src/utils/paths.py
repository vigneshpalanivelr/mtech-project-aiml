"""
Path management. Every module gets its directories from here so the layout
in DOCLAYOUT_YOLO_INDIC_SKILL.md is enforced in one place.

Usage:
    >>> from src.utils.paths import get_output_dir, get_checkpoint_path
    >>> img_dir = get_output_dir("synthetic") / "images"
"""

from pathlib import Path

from src.config import OUTPUT_DIR, FONTS_DIR, TEXT_CORPUS_DIR


def ensure(path: Path) -> Path:
    """Create a directory (and parents) if missing; return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_dir(*parts: str) -> Path:
    """output/<parts...>, created on demand."""
    return ensure(OUTPUT_DIR.joinpath(*parts))


def get_fonts_dir() -> Path:
    return ensure(FONTS_DIR)


def get_corpus_dir() -> Path:
    return ensure(TEXT_CORPUS_DIR)


def get_checkpoint_path(name: str) -> Path:
    return get_output_dir("checkpoints") / name


def get_synthetic_dirs() -> dict:
    """Return the canonical synthetic-data subdirectories."""
    root = get_output_dir("synthetic")
    return {
        "root": root,
        "images": ensure(root / "images"),
        "annotations": ensure(root / "annotations"),
    }
