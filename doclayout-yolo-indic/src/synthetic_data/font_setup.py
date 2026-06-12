"""
Download and verify Noto fonts for every supported script.

Fonts are pulled from the google/fonts repo (OFL-licensed, redistributable),
which is reachable from restricted networks where fonts.google.com is not.
Variable fonts are fine: HarfBuzz shapes them and the renderer reads the
default instance.

Run standalone:
    python -m src.synthetic_data.font_setup
"""

import urllib.request
from pathlib import Path

from fontTools.ttLib import TTFont

from src.logger import get_logger
from src.synthetic_data.indic_typography import REGISTRY
from src.utils.paths import get_fonts_dir

logger = get_logger(__name__)

_BASE = "https://github.com/google/fonts/raw/main/ofl"

# folder + url-encoded variable-axis filename in the google/fonts repo
_FONT_SOURCES = {
    "NotoSansDevanagari.ttf": f"{_BASE}/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf",
    "NotoSansBengali.ttf":    f"{_BASE}/notosansbengali/NotoSansBengali%5Bwdth%2Cwght%5D.ttf",
    "NotoSansTamil.ttf":      f"{_BASE}/notosanstamil/NotoSansTamil%5Bwdth%2Cwght%5D.ttf",
    "NotoSansTelugu.ttf":     f"{_BASE}/notosanstelugu/NotoSansTelugu%5Bwdth%2Cwght%5D.ttf",
    "NotoSansKannada.ttf":    f"{_BASE}/notosanskannada/NotoSansKannada%5Bwdth%2Cwght%5D.ttf",
    "NotoSansMalayalam.ttf":  f"{_BASE}/notosansmalayalam/NotoSansMalayalam%5Bwdth%2Cwght%5D.ttf",
    "NotoSansGujarati.ttf":   f"{_BASE}/notosansgujarati/NotoSansGujarati%5Bwdth%2Cwght%5D.ttf",
    "NotoSansGurmukhi.ttf":   f"{_BASE}/notosansgurmukhi/NotoSansGurmukhi%5Bwdth%2Cwght%5D.ttf",
    "NotoSansOriya.ttf":      f"{_BASE}/notosansoriya/NotoSansOriya%5Bwdth%2Cwght%5D.ttf",
    "NotoNaskhArabic.ttf":    f"{_BASE}/notonaskharabic/NotoNaskhArabic%5Bwght%5D.ttf",
    "NotoSans.ttf":           f"{_BASE}/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf",
}


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    dest.write_bytes(data)


def verify_font(path: Path) -> bool:
    """True if the file is a parseable TTF/OTF with a usable cmap."""
    try:
        f = TTFont(str(path), lazy=True)
        n_glyphs = len(f.getGlyphOrder())
        has_cmap = "cmap" in f and f["cmap"].getBestCmap() is not None
        f.close()
        return n_glyphs > 50 and has_cmap
    except Exception as e:  # noqa: BLE001
        logger.error(f"{path.name} failed verification: {e}")
        return False


def setup_fonts(force: bool = False) -> dict:
    """Ensure all required fonts exist and are valid. Returns {lang: Path}."""
    fonts_dir = get_fonts_dir()
    needed = {spec.font_file for spec in REGISTRY.values()}

    for fname in sorted(needed):
        dest = fonts_dir / fname
        if dest.exists() and not force and verify_font(dest):
            logger.info(f"OK (cached)  {fname}")
            continue
        url = _FONT_SOURCES.get(fname)
        if url is None:
            logger.error(f"No source URL configured for {fname}")
            continue
        logger.info(f"Downloading  {fname}")
        try:
            _download(url, dest)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Download failed for {fname}: {e}")
            continue
        if verify_font(dest):
            logger.info(f"OK           {fname} ({dest.stat().st_size // 1024} KB)")
        else:
            logger.error(f"INVALID      {fname} (re-check URL)")

    # Build language -> path map (only for valid fonts)
    resolved = {}
    for lang, spec in REGISTRY.items():
        p = fonts_dir / spec.font_file
        if p.exists() and verify_font(p):
            resolved[lang] = p
    return resolved


if __name__ == "__main__":
    paths = setup_fonts()
    logger.info(f"Fonts ready for {len(paths)}/{len(REGISTRY)} languages: "
                f"{sorted(paths)}")
