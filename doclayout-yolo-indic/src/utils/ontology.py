"""
Ontology bridge: synthetic (Phase 2) <-> IndicDLP (Phase 3/4).

WHY THIS IS NOT A BLOCKER FOR PHASE 2
-------------------------------------
Our synthetic corpus uses a coarse 9-class layout ontology (config.COCO_CLASSES).
IndicDLP uses 42 fine-grained, M6Doc-aligned region labels. These do NOT need to
match: DocLayout-YOLO's recipe pretrains on a coarse synthetic ontology
(DocSynth300K) and then *replaces the detection head* when fine-tuning on a
dataset with a different class count. We do the same -- pretrain a 9-class head
on IndicSynth (Phase 2), then re-init a 42-class head for IndicDLP (Phase 4).
Pretraining transfers the backbone's sense of Indic layout structure; the head
is discarded. So no relabeling of the synthetic data is required.

WHAT THIS MODULE GIVES YOU FOR PHASE 3
--------------------------------------
1. inspect_indicdlp_categories(): dump the real 42 category names + instance
   counts straight from the IndicDLP COCO annotations once you've extracted the
   tar -- run this first thing in Phase 3 to finalize any mapping you want.
2. SYNTHETIC_TO_INDICDLP_COARSE: a *starting* coarse map (our 9 classes -> the
   IndicDLP region names they most likely correspond to). Refine it against the
   real category list before using it for cross-dataset analysis. The IndicDLP
   names below are best-guess M6Doc-style labels and MUST be verified against
   inspect_indicdlp_categories() output -- treat unverified names as TODO.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from src.logger import get_logger

logger = get_logger(__name__)

# Best-guess coarse correspondence (VERIFY against the real category list).
SYNTHETIC_TO_INDICDLP_COARSE: Dict[str, List[str]] = {
    "text_body":        ["paragraph", "text", "plain-text"],
    "headline":         ["title", "headline", "section-title", "header"],
    "table":            ["table"],
    "figure":           ["figure", "image", "photo"],
    "caption":          ["caption"],
    "advertisement":    ["advertisement"],
    "sidebar":          ["sidebar"],
    "pull_quote":       ["quote", "pull-quote"],
    "decorative_frame": ["flag", "folio", "header", "footer", "ornament"],
}


def inspect_indicdlp_categories(coco_json_path: str) -> List[dict]:
    """Print and return IndicDLP categories with instance counts.

    Run in Phase 3 after extracting indicdlp.tar, pointing at its COCO json
    (e.g. .../annotations/train.json). Use the output to confirm the real 42
    labels and finalize any mapping.
    """
    data = json.loads(Path(coco_json_path).read_text())
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    counts = Counter(a["category_id"] for a in data.get("annotations", []))
    rows = sorted(cats.items(), key=lambda kv: -counts.get(kv[0], 0))
    logger.info(f"IndicDLP: {len(cats)} categories, "
                f"{len(data.get('annotations', []))} instances")
    out = []
    for cid, name in rows:
        n = counts.get(cid, 0)
        out.append({"id": cid, "name": name, "count": n})
        print(f"  {cid:3d}  {name:28s}  {n:>8d}")
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        inspect_indicdlp_categories(sys.argv[1])
    else:
        print("Phase 2 uses a 9-class synthetic ontology; the head is re-init'd "
              "for IndicDLP's 42 classes at fine-tuning (no remap needed).")
        print("Phase 3: python -m src.utils.ontology <indicdlp_coco.json>")
