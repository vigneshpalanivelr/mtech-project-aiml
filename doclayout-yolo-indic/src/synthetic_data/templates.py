"""
Layout templates for synthetic Indic documents.

Coordinates are normalized (fractions of page width/height) so a template
renders at any resolution. Each region declares:
    cls   : COCO class name (see config.COCO_CLASSES)
    box   : (x, y, w, h) as fractions in [0, 1]
    kind  : how the generator fills it -> "text" | "figure" | "table" | "frame"
    words : approximate word count for text regions

This library spans the document families that dominate Indic DLA corpora
(newspapers, books, textbooks, forms/gazettes, magazines, reference works,
exam papers, commerce). Newspaper templates use deliberately narrow columns,
which is characteristic of Indic newspapers. pick_template() samples by
family weights so common families appear more often than rare ones.

Add more dicts to TEMPLATES freely; the generator picks up new entries with no
other code changes.
"""

import random
from typing import Dict, List

# ---------------------------------------------------------------------------
# Newspapers (narrow columns, dense)
# ---------------------------------------------------------------------------
_NEWSPAPER_2COL = {
    "name": "newspaper_2col",
    "regions": [
        {"cls": "headline",     "box": (0.05, 0.04, 0.90, 0.09), "kind": "text", "words": 6},
        {"cls": "text_body",    "box": (0.05, 0.16, 0.43, 0.55), "kind": "text", "words": 120},
        {"cls": "text_body",    "box": (0.52, 0.16, 0.43, 0.40), "kind": "text", "words": 90},
        {"cls": "figure",       "box": (0.52, 0.58, 0.43, 0.20), "kind": "figure"},
        {"cls": "caption",      "box": (0.52, 0.79, 0.43, 0.04), "kind": "text", "words": 8},
        {"cls": "advertisement","box": (0.05, 0.74, 0.43, 0.21), "kind": "frame"},
    ],
}

_NEWSPAPER_3COL = {
    "name": "newspaper_3col",
    "regions": [
        {"cls": "headline",  "box": (0.04, 0.03, 0.92, 0.08), "kind": "text", "words": 7},
        {"cls": "text_body", "box": (0.04, 0.13, 0.29, 0.60), "kind": "text", "words": 110},
        {"cls": "text_body", "box": (0.355,0.13, 0.29, 0.60), "kind": "text", "words": 110},
        {"cls": "text_body", "box": (0.67, 0.13, 0.29, 0.40), "kind": "text", "words": 70},
        {"cls": "figure",    "box": (0.67, 0.55, 0.29, 0.18), "kind": "figure"},
        {"cls": "pull_quote","box": (0.04, 0.76, 0.92, 0.07), "kind": "text", "words": 14},
        {"cls": "sidebar",   "box": (0.04, 0.86, 0.45, 0.10), "kind": "text", "words": 30},
    ],
}

_NEWSPAPER_4COL = {
    "name": "newspaper_4col_dense",
    "regions": [
        {"cls": "headline",  "box": (0.03, 0.03, 0.94, 0.07), "kind": "text", "words": 8},
        {"cls": "text_body", "box": (0.03, 0.12, 0.225, 0.62), "kind": "text", "words": 95},
        {"cls": "text_body", "box": (0.275,0.12, 0.225, 0.62), "kind": "text", "words": 95},
        {"cls": "text_body", "box": (0.52, 0.12, 0.225, 0.45), "kind": "text", "words": 70},
        {"cls": "text_body", "box": (0.765,0.12, 0.205, 0.62), "kind": "text", "words": 90},
        {"cls": "figure",    "box": (0.52, 0.59, 0.225, 0.15), "kind": "figure"},
        {"cls": "advertisement","box": (0.03, 0.77, 0.55, 0.20),"kind": "frame"},
        {"cls": "sidebar",   "box": (0.61, 0.77, 0.36, 0.20), "kind": "text", "words": 40},
    ],
}

_NEWSPAPER_FRONTPAGE = {
    "name": "newspaper_frontpage",
    "regions": [
        {"cls": "decorative_frame","box": (0.04, 0.02, 0.92, 0.06),"kind": "frame"},
        {"cls": "headline",  "box": (0.04, 0.10, 0.62, 0.08), "kind": "text", "words": 8},
        {"cls": "figure",    "box": (0.68, 0.10, 0.28, 0.22), "kind": "figure"},
        {"cls": "caption",   "box": (0.68, 0.33, 0.28, 0.03), "kind": "text", "words": 7},
        {"cls": "text_body", "box": (0.04, 0.20, 0.30, 0.50), "kind": "text", "words": 90},
        {"cls": "text_body", "box": (0.36, 0.20, 0.30, 0.50), "kind": "text", "words": 90},
        {"cls": "headline",  "box": (0.68, 0.38, 0.28, 0.04), "kind": "text", "words": 5},
        {"cls": "text_body", "box": (0.68, 0.43, 0.28, 0.27), "kind": "text", "words": 65},
        {"cls": "advertisement","box": (0.04, 0.73, 0.92, 0.23),"kind": "frame"},
    ],
}

_MULTI_STORY_NEWS = {
    "name": "newspaper_multistory",
    "regions": [
        {"cls": "headline",  "box": (0.05, 0.04, 0.90, 0.06), "kind": "text", "words": 6},
        {"cls": "text_body", "box": (0.05, 0.12, 0.43, 0.24), "kind": "text", "words": 60},
        {"cls": "headline",  "box": (0.52, 0.04, 0.43, 0.05), "kind": "text", "words": 4},
        {"cls": "text_body", "box": (0.52, 0.11, 0.43, 0.25), "kind": "text", "words": 60},
        {"cls": "figure",    "box": (0.05, 0.39, 0.43, 0.18), "kind": "figure"},
        {"cls": "headline",  "box": (0.52, 0.39, 0.43, 0.05), "kind": "text", "words": 5},
        {"cls": "text_body", "box": (0.52, 0.46, 0.43, 0.28), "kind": "text", "words": 70},
        {"cls": "text_body", "box": (0.05, 0.60, 0.43, 0.34), "kind": "text", "words": 85},
        {"cls": "caption",   "box": (0.05, 0.58, 0.43, 0.02), "kind": "text", "words": 6},
    ],
}

# ---------------------------------------------------------------------------
# Books & textbooks
# ---------------------------------------------------------------------------
_TEXTBOOK = {
    "name": "textbook_single",
    "regions": [
        {"cls": "headline",  "box": (0.10, 0.05, 0.80, 0.06), "kind": "text", "words": 5},
        {"cls": "text_body", "box": (0.10, 0.14, 0.80, 0.30), "kind": "text", "words": 140},
        {"cls": "figure",    "box": (0.25, 0.46, 0.50, 0.22), "kind": "figure"},
        {"cls": "caption",   "box": (0.25, 0.69, 0.50, 0.04), "kind": "text", "words": 10},
        {"cls": "text_body", "box": (0.10, 0.75, 0.80, 0.20), "kind": "text", "words": 90},
    ],
}

_TEXTBOOK_2COL = {
    "name": "textbook_2col",
    "regions": [
        {"cls": "headline",  "box": (0.08, 0.05, 0.84, 0.06), "kind": "text", "words": 6},
        {"cls": "text_body", "box": (0.08, 0.14, 0.40, 0.50), "kind": "text", "words": 120},
        {"cls": "text_body", "box": (0.52, 0.14, 0.40, 0.32), "kind": "text", "words": 80},
        {"cls": "figure",    "box": (0.52, 0.48, 0.40, 0.16), "kind": "figure"},
        {"cls": "caption",   "box": (0.52, 0.65, 0.40, 0.03), "kind": "text", "words": 8},
        {"cls": "text_body", "box": (0.08, 0.68, 0.84, 0.27), "kind": "text", "words": 110},
    ],
}

_TEXTBOOK_TABLE = {
    "name": "textbook_with_table",
    "regions": [
        {"cls": "headline",  "box": (0.10, 0.05, 0.80, 0.06), "kind": "text", "words": 5},
        {"cls": "text_body", "box": (0.10, 0.14, 0.80, 0.18), "kind": "text", "words": 80},
        {"cls": "table",     "box": (0.12, 0.35, 0.76, 0.32), "kind": "table"},
        {"cls": "caption",   "box": (0.12, 0.68, 0.76, 0.03), "kind": "text", "words": 8},
        {"cls": "text_body", "box": (0.10, 0.73, 0.80, 0.22), "kind": "text", "words": 95},
    ],
}

_BOOK_PROSE = {
    "name": "book_prose_single",
    "regions": [
        {"cls": "headline",  "box": (0.18, 0.07, 0.64, 0.05), "kind": "text", "words": 4},
        {"cls": "text_body", "box": (0.16, 0.15, 0.68, 0.72), "kind": "text", "words": 230},
        {"cls": "caption",   "box": (0.45, 0.92, 0.10, 0.03), "kind": "text", "words": 2},
    ],
}

_BOOK_FOOTNOTES = {
    "name": "book_with_footnotes",
    "regions": [
        {"cls": "text_body", "box": (0.16, 0.08, 0.68, 0.62), "kind": "text", "words": 200},
        {"cls": "decorative_frame","box": (0.16, 0.72, 0.68, 0.006),"kind": "frame"},
        {"cls": "sidebar",   "box": (0.16, 0.74, 0.68, 0.18), "kind": "text", "words": 55},
        {"cls": "caption",   "box": (0.45, 0.94, 0.10, 0.03), "kind": "text", "words": 2},
    ],
}

_RELIGIOUS_TEXT = {
    "name": "religious_text_commentary",
    "regions": [
        {"cls": "decorative_frame","box": (0.06, 0.05, 0.88, 0.90),"kind": "frame"},
        {"cls": "headline",  "box": (0.25, 0.08, 0.50, 0.05), "kind": "text", "words": 4},
        {"cls": "pull_quote","box": (0.22, 0.16, 0.56, 0.18), "kind": "text", "words": 30},
        {"cls": "text_body", "box": (0.10, 0.37, 0.37, 0.52), "kind": "text", "words": 110},
        {"cls": "text_body", "box": (0.53, 0.37, 0.37, 0.52), "kind": "text", "words": 110},
    ],
}

# ---------------------------------------------------------------------------
# Reference works
# ---------------------------------------------------------------------------
_DICTIONARY = {
    "name": "dictionary_glossary",
    "regions": [
        {"cls": "headline",  "box": (0.06, 0.04, 0.88, 0.05), "kind": "text", "words": 3},
        {"cls": "text_body", "box": (0.06, 0.11, 0.43, 0.20), "kind": "text", "words": 50},
        {"cls": "text_body", "box": (0.06, 0.33, 0.43, 0.20), "kind": "text", "words": 50},
        {"cls": "text_body", "box": (0.06, 0.55, 0.43, 0.20), "kind": "text", "words": 50},
        {"cls": "text_body", "box": (0.06, 0.77, 0.43, 0.18), "kind": "text", "words": 45},
        {"cls": "text_body", "box": (0.52, 0.11, 0.43, 0.20), "kind": "text", "words": 50},
        {"cls": "text_body", "box": (0.52, 0.33, 0.43, 0.20), "kind": "text", "words": 50},
        {"cls": "text_body", "box": (0.52, 0.55, 0.43, 0.40), "kind": "text", "words": 95},
    ],
}

# ---------------------------------------------------------------------------
# Forms, gazettes, official
# ---------------------------------------------------------------------------
_FORM = {
    "name": "govt_form",
    "regions": [
        {"cls": "headline",        "box": (0.08, 0.05, 0.84, 0.07), "kind": "text", "words": 6},
        {"cls": "text_body",       "box": (0.08, 0.15, 0.84, 0.10), "kind": "text", "words": 40},
        {"cls": "table",           "box": (0.08, 0.28, 0.84, 0.45), "kind": "table"},
        {"cls": "text_body",       "box": (0.08, 0.76, 0.55, 0.10), "kind": "text", "words": 30},
        {"cls": "decorative_frame","box": (0.70, 0.76, 0.22, 0.18), "kind": "frame"},
    ],
}

_FORM_SIGNATURE = {
    "name": "form_fields_signature",
    "regions": [
        {"cls": "decorative_frame","box": (0.08, 0.04, 0.84, 0.06),"kind": "frame"},
        {"cls": "headline",  "box": (0.12, 0.05, 0.50, 0.04), "kind": "text", "words": 5},
        {"cls": "table",     "box": (0.08, 0.13, 0.84, 0.30), "kind": "table"},
        {"cls": "text_body", "box": (0.08, 0.46, 0.84, 0.22), "kind": "text", "words": 70},
        {"cls": "table",     "box": (0.08, 0.70, 0.84, 0.14), "kind": "table"},
        {"cls": "decorative_frame","box": (0.62, 0.86, 0.30, 0.09),"kind": "frame"},
    ],
}

_GAZETTE = {
    "name": "govt_gazette",
    "regions": [
        {"cls": "decorative_frame","box": (0.10, 0.04, 0.80, 0.05),"kind": "frame"},
        {"cls": "headline",  "box": (0.15, 0.10, 0.70, 0.05), "kind": "text", "words": 6},
        {"cls": "text_body", "box": (0.12, 0.18, 0.76, 0.70), "kind": "text", "words": 280},
        {"cls": "decorative_frame","box": (0.70, 0.89, 0.18, 0.07),"kind": "frame"},
    ],
}

_LETTER = {
    "name": "official_letter",
    "regions": [
        {"cls": "decorative_frame","box": (0.10, 0.05, 0.80, 0.08),"kind": "frame"},
        {"cls": "text_body", "box": (0.62, 0.15, 0.28, 0.05), "kind": "text", "words": 8},
        {"cls": "text_body", "box": (0.10, 0.22, 0.50, 0.06), "kind": "text", "words": 12},
        {"cls": "text_body", "box": (0.10, 0.31, 0.80, 0.45), "kind": "text", "words": 170},
        {"cls": "decorative_frame","box": (0.10, 0.80, 0.30, 0.10),"kind": "frame"},
    ],
}

# ---------------------------------------------------------------------------
# Magazines
# ---------------------------------------------------------------------------
_MAGAZINE = {
    "name": "magazine_feature",
    "regions": [
        {"cls": "figure",     "box": (0.0, 0.0, 1.0, 0.38), "kind": "figure"},
        {"cls": "headline",   "box": (0.06, 0.40, 0.88, 0.08), "kind": "text", "words": 5},
        {"cls": "pull_quote", "box": (0.06, 0.50, 0.40, 0.10), "kind": "text", "words": 16},
        {"cls": "text_body",  "box": (0.50, 0.50, 0.44, 0.44), "kind": "text", "words": 130},
        {"cls": "text_body",  "box": (0.06, 0.62, 0.40, 0.32), "kind": "text", "words": 95},
    ],
}

_MAGAZINE_2COL = {
    "name": "magazine_2col_pullquote",
    "regions": [
        {"cls": "headline",  "box": (0.06, 0.05, 0.88, 0.07), "kind": "text", "words": 6},
        {"cls": "text_body", "box": (0.06, 0.15, 0.42, 0.55), "kind": "text", "words": 130},
        {"cls": "pull_quote","box": (0.52, 0.15, 0.42, 0.14), "kind": "text", "words": 22},
        {"cls": "text_body", "box": (0.52, 0.32, 0.42, 0.38), "kind": "text", "words": 90},
        {"cls": "figure",    "box": (0.06, 0.73, 0.42, 0.22), "kind": "figure"},
        {"cls": "text_body", "box": (0.52, 0.73, 0.42, 0.22), "kind": "text", "words": 55},
    ],
}

_MAGAZINE_GRID = {
    "name": "magazine_photo_grid",
    "regions": [
        {"cls": "headline", "box": (0.06, 0.05, 0.88, 0.07), "kind": "text", "words": 5},
        {"cls": "figure",   "box": (0.06, 0.15, 0.42, 0.28), "kind": "figure"},
        {"cls": "caption",  "box": (0.06, 0.44, 0.42, 0.03), "kind": "text", "words": 7},
        {"cls": "figure",   "box": (0.52, 0.15, 0.42, 0.28), "kind": "figure"},
        {"cls": "caption",  "box": (0.52, 0.44, 0.42, 0.03), "kind": "text", "words": 7},
        {"cls": "figure",   "box": (0.06, 0.50, 0.42, 0.28), "kind": "figure"},
        {"cls": "caption",  "box": (0.06, 0.79, 0.42, 0.03), "kind": "text", "words": 7},
        {"cls": "figure",   "box": (0.52, 0.50, 0.42, 0.28), "kind": "figure"},
        {"cls": "caption",  "box": (0.52, 0.79, 0.42, 0.03), "kind": "text", "words": 7},
        {"cls": "text_body","box": (0.06, 0.84, 0.88, 0.11), "kind": "text", "words": 45},
    ],
}

# ---------------------------------------------------------------------------
# Exams & commerce
# ---------------------------------------------------------------------------
_EXAM_PAPER = {
    "name": "exam_question_paper",
    "regions": [
        {"cls": "decorative_frame","box": (0.08, 0.04, 0.84, 0.07),"kind": "frame"},
        {"cls": "headline",  "box": (0.20, 0.05, 0.60, 0.04), "kind": "text", "words": 6},
        {"cls": "text_body", "box": (0.08, 0.13, 0.84, 0.08), "kind": "text", "words": 30},
        {"cls": "text_body", "box": (0.08, 0.23, 0.84, 0.14), "kind": "text", "words": 45},
        {"cls": "text_body", "box": (0.08, 0.39, 0.84, 0.14), "kind": "text", "words": 45},
        {"cls": "text_body", "box": (0.08, 0.55, 0.84, 0.14), "kind": "text", "words": 45},
        {"cls": "text_body", "box": (0.08, 0.71, 0.84, 0.22), "kind": "text", "words": 70},
    ],
}

_INVOICE = {
    "name": "invoice_receipt",
    "regions": [
        {"cls": "decorative_frame","box": (0.08, 0.04, 0.84, 0.09),"kind": "frame"},
        {"cls": "headline",  "box": (0.12, 0.05, 0.45, 0.05), "kind": "text", "words": 4},
        {"cls": "text_body", "box": (0.08, 0.16, 0.50, 0.10), "kind": "text", "words": 25},
        {"cls": "text_body", "box": (0.62, 0.16, 0.30, 0.10), "kind": "text", "words": 18},
        {"cls": "table",     "box": (0.08, 0.30, 0.84, 0.40), "kind": "table"},
        {"cls": "table",     "box": (0.58, 0.72, 0.34, 0.14), "kind": "table"},
        {"cls": "text_body", "box": (0.08, 0.88, 0.45, 0.06), "kind": "text", "words": 18},
    ],
}

_CATALOG = {
    "name": "product_catalog",
    "regions": [
        {"cls": "headline", "box": (0.06, 0.04, 0.88, 0.06), "kind": "text", "words": 4},
        {"cls": "figure",   "box": (0.07, 0.13, 0.26, 0.22), "kind": "figure"},
        {"cls": "caption",  "box": (0.07, 0.36, 0.26, 0.05), "kind": "text", "words": 10},
        {"cls": "figure",   "box": (0.37, 0.13, 0.26, 0.22), "kind": "figure"},
        {"cls": "caption",  "box": (0.37, 0.36, 0.26, 0.05), "kind": "text", "words": 10},
        {"cls": "figure",   "box": (0.67, 0.13, 0.26, 0.22), "kind": "figure"},
        {"cls": "caption",  "box": (0.67, 0.36, 0.26, 0.05), "kind": "text", "words": 10},
        {"cls": "figure",   "box": (0.07, 0.45, 0.26, 0.22), "kind": "figure"},
        {"cls": "caption",  "box": (0.07, 0.68, 0.26, 0.05), "kind": "text", "words": 10},
        {"cls": "figure",   "box": (0.37, 0.45, 0.26, 0.22), "kind": "figure"},
        {"cls": "caption",  "box": (0.37, 0.68, 0.26, 0.05), "kind": "text", "words": 10},
        {"cls": "figure",   "box": (0.67, 0.45, 0.26, 0.22), "kind": "figure"},
        {"cls": "caption",  "box": (0.67, 0.68, 0.26, 0.05), "kind": "text", "words": 10},
        {"cls": "advertisement","box": (0.07, 0.76, 0.86, 0.18),"kind": "frame"},
    ],
}

_POSTER = {
    "name": "poster_flyer",
    "regions": [
        {"cls": "headline",  "box": (0.08, 0.06, 0.84, 0.14), "kind": "text", "words": 4},
        {"cls": "figure",    "box": (0.15, 0.24, 0.70, 0.38), "kind": "figure"},
        {"cls": "pull_quote","box": (0.12, 0.65, 0.76, 0.10), "kind": "text", "words": 14},
        {"cls": "text_body", "box": (0.15, 0.78, 0.70, 0.10), "kind": "text", "words": 30},
        {"cls": "advertisement","box": (0.30, 0.90, 0.40, 0.06),"kind": "frame"},
    ],
}

# ---------------------------------------------------------------------------
# Registry + weighted sampling
# ---------------------------------------------------------------------------
TEMPLATES: List[Dict] = [
    _NEWSPAPER_2COL, _NEWSPAPER_3COL, _NEWSPAPER_4COL, _NEWSPAPER_FRONTPAGE,
    _MULTI_STORY_NEWS,
    _TEXTBOOK, _TEXTBOOK_2COL, _TEXTBOOK_TABLE, _BOOK_PROSE, _BOOK_FOOTNOTES,
    _RELIGIOUS_TEXT, _DICTIONARY,
    _FORM, _FORM_SIGNATURE, _GAZETTE, _LETTER,
    _MAGAZINE, _MAGAZINE_2COL, _MAGAZINE_GRID,
    _EXAM_PAPER, _INVOICE, _CATALOG, _POSTER,
]

# Gentle weighting: families common in real Indic DLA corpora appear more
# often. Tune once IndicDLP domain frequencies are known. Unlisted -> 1.0.
_FAMILY_WEIGHT = {
    "newspaper": 1.6, "textbook": 1.4, "book": 1.3, "govt": 1.2, "form": 1.2,
    "magazine": 1.0, "dictionary": 0.7, "religious": 0.8,
    "exam": 0.8, "invoice": 0.7, "product": 0.6, "poster": 0.6, "official": 0.9,
}


def _weight_of(name: str) -> float:
    for key, w in _FAMILY_WEIGHT.items():
        if name.startswith(key):
            return w
    return 1.0


_WEIGHTS = [_weight_of(t["name"]) for t in TEMPLATES]


def pick_template(rng: random.Random) -> Dict:
    return rng.choices(TEMPLATES, weights=_WEIGHTS, k=1)[0]
