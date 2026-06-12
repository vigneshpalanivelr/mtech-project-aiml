"""
Indic text renderer: HarfBuzz shaping + fontTools outline rasterization.

Why not Pillow's draw.text? Pillow only shapes complex scripts when libraqm
is present. On many machines it is not, and Pillow then renders glyphs in
logical order with no conjunct formation, matra reordering, or RTL handling --
exactly the failures this project exists to avoid. Shaping with HarfBuzz and
filling the resulting glyph outlines ourselves is portable and correct.

The renderer shapes word-by-word (Indic scripts do not shape across spaces),
which lets it wrap text inside a box. It returns the rendered RGBA tile plus
the tight ink bounding box, so the generator can emit accurate COCO boxes.

Usage:
    >>> r = TextRenderer()
    >>> tile, ink = r.render_block("नमस्ते भारत", "Hindi", box=(400, 120), font_size=28)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import uharfbuzz as hb
import numpy as np
from PIL import Image, ImageDraw
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen

from src.config import PAGE_FG
from src.logger import get_logger
from src.synthetic_data.indic_typography import get_spec

logger = get_logger(__name__)

_CURVE_STEPS = 8  # bezier flattening resolution


class _PolygonPen(BasePen):
    """Collects flattened contours (lists of (x, y) in font units)."""

    def __init__(self, glyph_set):
        super().__init__(glyph_set)
        self.contours: List[List[Tuple[float, float]]] = []
        self._cur: List[Tuple[float, float]] = []

    def _moveTo(self, p):
        if self._cur:
            self.contours.append(self._cur)
        self._cur = [p]

    def _lineTo(self, p):
        self._cur.append(p)

    def _curveToOne(self, c1, c2, p):
        p0 = self._cur[-1]
        for t in np.linspace(0, 1, _CURVE_STEPS)[1:]:
            mt = 1 - t
            x = mt**3*p0[0] + 3*mt**2*t*c1[0] + 3*mt*t**2*c2[0] + t**3*p[0]
            y = mt**3*p0[1] + 3*mt**2*t*c1[1] + 3*mt*t**2*c2[1] + t**3*p[1]
            self._cur.append((x, y))

    def _qCurveToOne(self, c, p):
        p0 = self._cur[-1]
        for t in np.linspace(0, 1, _CURVE_STEPS)[1:]:
            mt = 1 - t
            x = mt**2*p0[0] + 2*mt*t*c[0] + t**2*p[0]
            y = mt**2*p0[1] + 2*mt*t*c[1] + t**2*p[1]
            self._cur.append((x, y))

    def _closePath(self):
        if self._cur:
            self.contours.append(self._cur)
            self._cur = []


@dataclass
class _ShapedWord:
    glyphs: List[Tuple[int, float, float, float]]  # (gid, x_off, y_off, x_adv) px
    width: float


class _FontHandle:
    """Caches the HarfBuzz font + fontTools glyph set for one font file."""

    def __init__(self, path: Path):
        data = path.read_bytes()
        self.hb_face = hb.Face(data)
        self.hb_font = hb.Font(self.hb_face)
        self.upem = self.hb_face.upem
        self.hb_font.scale = (self.upem, self.upem)
        self.tt = TTFont(str(path), fontNumber=0)
        self.glyph_order = self.tt.getGlyphOrder()
        self.glyph_set = self.tt.getGlyphSet()
        hhea = self.tt["hhea"]
        self.ascender = hhea.ascent
        self.descender = hhea.descent  # negative


class TextRenderer:
    """Shapes and rasterizes Indic/Latin text into transparent RGBA tiles."""

    def __init__(self, fonts: Optional[dict] = None):
        # fonts: {language: Path}. If None, resolve lazily from data/fonts.
        from src.synthetic_data.font_setup import setup_fonts
        self.font_paths = fonts or setup_fonts()
        self._cache: dict = {}

    def _handle(self, language: str) -> _FontHandle:
        path = self.font_paths[language]
        if path not in self._cache:
            self._cache[path] = _FontHandle(path)
        return self._cache[path]

    def _shape_word(self, word: str, fh: _FontHandle, scale: float) -> _ShapedWord:
        buf = hb.Buffer()
        buf.add_str(word)
        buf.guess_segment_properties()
        hb.shape(fh.hb_font, buf)
        glyphs, pen_x = [], 0.0
        for gi, gp in zip(buf.glyph_infos, buf.glyph_positions):
            glyphs.append((
                gi.codepoint,
                (pen_x + gp.x_offset) * scale,
                gp.y_offset * scale,
                gp.x_advance * scale,
            ))
            pen_x += gp.x_advance
        return _ShapedWord(glyphs, pen_x * scale)

    def _draw_glyph(self, draw: ImageDraw.ImageDraw, fh: _FontHandle,
                    gid: int, ox: float, oy: float, scale: float):
        pen = _PolygonPen(fh.glyph_set)
        fh.glyph_set[fh.glyph_order[gid]].draw(pen)
        for contour in pen.contours:
            if len(contour) >= 3:
                pts = [(ox + px * scale, oy - py * scale) for px, py in contour]
                draw.polygon(pts, fill=(PAGE_FG, PAGE_FG, PAGE_FG, 255))

    def render_block(self, text: str, language: str,
                     box: Tuple[int, int], font_size: int,
                     line_spacing: float = 1.55,
                     ) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """
        Render wrapped text into a tile of size `box` (w, h).

        Returns (RGBA tile, ink_bbox) where ink_bbox is [x, y, w, h] of the
        actual rendered pixels (clamped to the tile), suitable for COCO.
        """
        spec = get_spec(language)
        fh = self._handle(language)
        scale = font_size / fh.upem
        box_w, box_h = box
        line_h = font_size * line_spacing

        tile = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(tile)

        words = text.split()
        if not words:
            return tile, (0, 0, 0, 0)
        space_w = font_size * 0.32
        rtl = spec.direction == "rtl"

        # group words into lines that fit box_w
        lines: List[List[_ShapedWord]] = [[]]
        cur_w = 0.0
        for w in words:
            sw = self._shape_word(w, fh, scale)
            add = sw.width + (space_w if lines[-1] else 0)
            if cur_w + add > box_w and lines[-1]:
                lines.append([sw])
                cur_w = sw.width
            else:
                lines[-1].append(sw)
                cur_w += add

        baseline = font_size
        min_x, min_y, max_x, max_y = box_w, box_h, 0, 0
        for line in lines:
            if baseline - font_size > box_h:
                break
            # x cursor: left for LTR, right for RTL
            x = box_w if rtl else 0.0
            for i, sw in enumerate(line):
                if rtl:
                    x -= sw.width
                    word_x = x
                    x -= space_w
                else:
                    word_x = x
                    x += sw.width + space_w
                for gid, gx, gy, _ in sw.glyphs:
                    ox = word_x + gx
                    oy = baseline - gy
                    if -font_size < ox < box_w + font_size:
                        self._draw_glyph(draw, fh, gid, ox, oy, scale)
                min_x = min(min_x, word_x)
                max_x = max(max_x, word_x + sw.width)
            min_y = min(min_y, baseline - font_size)
            max_y = max(max_y, baseline + (-fh.descender) * scale)
            baseline += line_h

        # tight ink bbox via alpha channel (more accurate than glyph metrics)
        alpha = np.array(tile.split()[-1])
        ys, xs = np.where(alpha > 0)
        if len(xs) == 0:
            return tile, (0, 0, 0, 0)
        bx, by = int(xs.min()), int(ys.min())
        bw, bh = int(xs.max() - bx + 1), int(ys.max() - by + 1)
        return tile, (bx, by, bw, bh)


if __name__ == "__main__":
    r = TextRenderer()
    for lang, sample in [("Hindi", "नमस्ते भारत समाचार"),
                         ("Tamil", "தமிழ் செய்தி இன்று"),
                         ("Urdu", "اردو خبر آج")]:
        tile, ink = r.render_block(sample, lang, box=(420, 80), font_size=34)
        logger.info(f"{lang:8s} ink bbox = {ink}")
