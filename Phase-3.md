# PHASE 3 DETAILED COMPLIANCE REVIEW
## Deep Analysis Against Phase 2 Standards

**Review Date:** May 2026
**Standard Reference:** PROJECT_CONTEXT.md (Phase 2)
**Document Under Review:** PHASE_3_CONTEXT.md

---

## 1. DOCUMENT STRUCTURE COMPARISON

### Phase 2 Structure (Gold Standard):
1. Executive Summary
2. Phase 2 Overview (motivation + why it matters)
3. Phase 2 Detailed Implementation Steps (7 clear steps)
4. Key Implementation Details (libraries, patterns)
5. Metrics to Track
6. Failure Points to Watch
7. Phase 2 Completion Checklist
8. Next Phases

### Phase 3 Structure (Current):
1. Phase 3 Overview
2. What Happens in Phase 3 + Why It Matters
3. Phase 3 Detailed Steps (organized as Weeks 1-5)
   - Week 1-2: Self-Training (3 days + rounds)
   - Week 3-4: Fine-tuning
   - Week 5: Ablations
4. Phase 3 Completion Checklist
5. Key Files to Create
6. Key Metrics to Track
7. Failure Points to Watch
8. What Comes Next

**Verdict:** ✅ **Structure matches 95%** — Actually well-organized by weeks, which is better than Phase 2.

---

## 2. EXECUTIVE SUMMARY ANALYSIS

### Phase 2 Has:
```
**Duration:** May 11 – May 30, 2026 (20 days)
**Goal:** Generate 150,000 realistic synthetic Indic documents...
**Deliverables:** IndicSynth-150K corpus + pretrained checkpoint
```

### Phase 3 Has:
```
**Duration:** Jun 1 - Jul 6, 2026 (36 days / 5 weeks)
**Dependencies:** Phase 2 must be complete...
**Goal:** Apply semi-supervised learning on unlabeled Indic documents...
**Deliverables:** Fine-tuned checkpoint, Ablation experiment results...
```

✅ **GOOD:** Phase 3 actually adds "Dependencies" which is better than Phase 2!

---

## 3. DETAILED IMPLEMENTATION STEPS ANALYSIS

### 3.1 Structure & Granularity

**Phase 2 Example (Step 1):**
```
### Step 1: Prepare Fonts and Text Corpus (Days 1-2)

**What you do:**
- Download Noto Sans Indic fonts for each script
- Collect text samples in each language

**Outputs:**
[Explicit directory structure with files]

**Code location:** `src/data_preparation/font_setup.py`
```

**Phase 3 Example (Step 1):**
```
#### Step 1: Pseudo-Label Generation (Days 1-3)

**What you do:**
- Load Phase 2 pretrained checkpoint
- Run inference on BaDLAD unlabeled set (200K images subset)
- Generate confidence-thresholded pseudo-labels

**Key insight:** [Explanation provided]

**Python script:**
[Code block]

**Expected output:**
- `output/logs/badlad_pseudo_labels.json` containing 150K-200K pseudo-labeled images
- Statistics: average 4-6 detections per image
```

✅ **COMPARABLE or BETTER:** Phase 3 adds "Key insight" and explicit expected outputs.

---

### 3.2 Code Examples Analysis

**Phase 2 Code Count:** 11 code blocks (full Python examples)

**Phase 3 Code Count:** 8 code blocks (full Python examples)

**Phase 3 Gaps - Missing Code Examples:**

**Gap 1: Class-Balanced Threshold Computation (Step 2)**

Current Phase 3:
```python
# Code shown for thresholding
class_thresholds = {}
for cls_id in range(9):
    ...
```

⚠️ **Issue:** The LOGIC of class-balanced selection is unclear.

**Should ADD detailed explanation:**
```python
# Class-Balanced Threshold Selection Logic

"""
Why class-balanced thresholds?

Problem: Uniform threshold (e.g., 0.5) leads to:
  - Common classes (text_body) heavily pseudo-labeled
  - Rare classes (sidebar) barely represented
  - Self-training learns common class patterns, forgets rare ones

Solution: Adjust threshold per class to ensure balanced representation

Algorithm:
  1. Count detections per class at different confidence levels
  2. Find threshold that gives each class ~target_count detections
  3. Rare classes get lower thresholds → more pseudo-labels
  4. Common classes get higher thresholds → less pseudo-labels
  5. Result: Balanced pseudo-label distribution
"""

# Implementation:

from collections import defaultdict
import numpy as np

# Step 1: Collect all predictions with scores per class
class_predictions = defaultdict(list)

for item in raw_inference_results:
    for pred in item['predictions']:
        cls_id = pred['class']
        score = pred['confidence']
        class_predictions[cls_id].append(score)

# Step 2: Sort scores descending for each class
for cls_id in class_predictions:
    class_predictions[cls_id].sort(reverse=True)

# Step 3: Determine target count
# Most common class count
max_class_count = max(len(preds) for preds in class_predictions.values())
target_count = int(max_class_count * 0.5)  # 50% of max

print(f"Max predictions on any class: {max_class_count}")
print(f"Target for balanced: {target_count}")

# Step 4: Find threshold for each class
class_thresholds = {}

for cls_id in range(9):
    scores = class_predictions[cls_id]

    # Find the score at position target_count
    if len(scores) > target_count:
        threshold = scores[target_count]
    else:
        # Rare class: use lower threshold
        threshold = 0.3 + (cls_id * 0.02)  # Between 0.3-0.5

    class_thresholds[cls_id] = threshold

    num_kept = sum(1 for s in scores if s >= threshold)
    print(f"Class {cls_id}: threshold={threshold:.2f}, "
          f"predictions kept={num_kept}/{len(scores)}")

# Expected output:
# Class 0: threshold=0.70, predictions kept=500/1500
# Class 1: threshold=0.68, predictions kept=520/1480
# Class 2: threshold=0.55, predictions kept=495/900  ← lower threshold for rare class
# Class 3: threshold=0.52, predictions kept=480/850  ← even lower
# ...
```

**Gap 2: YAML Dataset File Format Not Shown**

Phase 3 references:
```python
data='data/badlad_self_training.yaml',
```

But doesn't show what this file contains!

**Should ADD:**
```yaml
# data/badlad_self_training.yaml
# Configuration file for self-training dataset
# Mixes labeled real data + pseudo-labeled real data

path: /path/to/data/raw/badlad
train: |
  labeled/images:labeled/annotations.json
  unlabeled_pseudo/images:output/logs/badlad_pseudo_labels_balanced.json
val: test/images:test/annotations.json
test: test/images:test/annotations.json

nc: 9
names: ['text_body', 'headline', 'table', 'figure', 'caption',
        'advertisement', 'sidebar', 'pull_quote', 'decorative_frame']

# Data mixing strategy:
# - 50% from labeled set (high confidence, official annotations)
# - 50% from pseudo-labeled set (model-generated, class-balanced)
# - This prevents overfitting to labeled subset while improving on unlabeled data
```

**Gap 3: Complete Self-Training Round JSON Output Not Shown**

Current Phase 3 has code that returns:
```python
results = model.train(...)
```

But doesn't show expected structure!

**Should ADD:**
```python
# Expected output from model.train() for Round 1:
# saved as: output/logs/selftraining_round_1_results.json

{
  "round": 1,
  "checkpoint": "output/checkpoints/doclayout_yolo_indic_selftraining_round1.pt",
  "training": {
    "epochs": 20,
    "total_images": 200000,  # Labeled + pseudo-labeled
    "batch_size": 32,
    "learning_rate_0": 0.001,
    "final_learning_rate": 0.00001,
    "total_batches": 6250,
    "total_iterations": 125000
  },
  "metrics": {
    "loss_per_epoch": [1.234, 1.198, 1.156, ..., 0.845],
    "val_mAP_per_epoch": [0.621, 0.638, 0.652, ..., 0.659],
    "training_time_hours": 8.5,
    "gpu_memory_peak_gb": 38.2
  },
  "best": {
    "epoch": 18,
    "mAP50-95": 0.6589,
    "mAP50": 0.7234,
    "mAP75": 0.6845
  }
}
```

**Gap 4: Ablation Results JSON Not Shown**

Phase 3 shows code that produces:
```python
ablation_results[exp['name']] = {
    "indicdlp_mAP": float(...),
    "badlad_mAP": float(...),
}
```

But doesn't show complete structure!

**Should ADD:**
```json
// output/logs/ablation_results.json - Complete structure

{
  "baseline": {
    "description": "English-trained DocLayout-YOLO, zero-shot on Indic",
    "checkpoint": "output/checkpoints/doclayout_yolo_english_baseline.pt",
    "indicdlp_mAP": 0.602,
    "indicdlp_mAP50": 0.685,
    "indicdlp_mAP75": 0.612,
    "badlad_mAP": 0.585,
    "per_script": {
      "Hindi": 0.625,
      "Tamil": 0.521,
      "Telugu": 0.559,
      "Bengali": 0.618,
      "Kannada": 0.548,
      "Malayalam": 0.541,
      "Gujarati": 0.562,
      "Punjabi": 0.555,
      "Odia": 0.539
    },
    "training_time_hours": 0,  // No training, zero-shot
    "gpu_memory_gb": 12.4
  },

  "+synthetic": {
    "description": "Phase 2 pretrained on IndicSynth-150K, no script-head, no self-training",
    "checkpoint": "output/checkpoints/doclayout_yolo_indic_pretrained.pt",
    "indicdlp_mAP": 0.661,
    "indicdlp_mAP50": 0.742,
    "indicdlp_mAP75": 0.678,
    "badlad_mAP": 0.643,
    "improvement_over_baseline": {
      "indicdlp_mAP": 0.059,  // +5.9%
      "badlad_mAP": 0.058
    },
    "training_time_hours": 4.2,
    "gpu_memory_gb": 38.1
  },

  "+selftraining": {
    "description": "Round 2 self-training on BaDLAD-unlabeled (200K images)",
    "checkpoint": "output/checkpoints/doclayout_yolo_indic_selftraining_round2.pt",
    "indicdlp_mAP": 0.728,
    "indicdlp_mAP50": 0.815,
    "indicdlp_mAP75": 0.752,
    "badlad_mAP": 0.702,
    "improvement_over_synthetic": {
      "indicdlp_mAP": 0.067,  // +6.7%
      "badlad_mAP": 0.059
    },
    "total_training_time_hours": 8.4,
    "gpu_memory_gb": 38.2
  },

  "full_finetuned": {
    "description": "Final system: synthetic + script-head + self-training + fine-tuned on IndicDLP",
    "checkpoint": "output/checkpoints/doclayout_yolo_indic_finetuned.pt",
    "indicdlp_mAP": 0.755,
    "indicdlp_mAP50": 0.823,
    "indicdlp_mAP75": 0.768,
    "badlad_mAP": 0.724,
    "improvement_over_baseline": {
      "indicdlp_mAP": 0.153,  // +15.3% total!
      "badlad_mAP": 0.139
    },
    "total_training_time_hours": 12.6,
    "gpu_memory_gb": 38.2
  }
}
```

---

## 4. EXPECTED OUTPUTS & FILE STRUCTURES

### Phase 2 Standard:
Each step explicitly shows output files and locations.

### Phase 3 Current:
Shows "Outputs:" but sometimes vague about file formats.

**Example - After Pseudo-Label Generation:**

Current:
```
**Outputs:**
- `output/logs/badlad_pseudo_labels.json` containing 150K-200K pseudo-labeled images
- Statistics: average 4-6 detections per image
```

Should ADD (like Phase 2):
```
**Expected Outputs:**

output/logs/
  ├── badlad_pseudo_labels.json              (file size: ~450 MB)
  │   Structure: List of 180000 objects
  │   [
  │     {
  │       "image": "badlad_00001.jpg",
  │       "annotations": [
  │         {"bbox": [...], "score": 0.87, "class": 0},
  │         ...
  │       ]
  │     },
  │     ...
  │   ]
  │
  ├── class_balanced_thresholds.json         (file size: ~2 KB)
  │   {
  │     "0": 0.70,
  │     "1": 0.70,
  │     "2": 0.55,
  │     ...
  │   }
  │
  └── selftraining_log.json                  (file size: ~5 KB)
      [
        {"round": 1, "epoch": 20, "metrics": {...}},
        {"round": 2, "epoch": 40, "metrics": {...}}
      ]

output/checkpoints/
  ├── doclayout_yolo_indic_selftraining_round1.pt    (180 MB)
  ├── doclayout_yolo_indic_selftraining_round2.pt    (180 MB)
  └── doclayout_yolo_indic_finetuned.pt              (180 MB)
```

---

## 5. FAILURE POINTS ANALYSIS

### Phase 3 Failure Points (Current):
```
1. **Self-training makes accuracy worse** → pseudo-labels are too noisy
   - Fix: Lower confidence threshold further, use only Round 1

2. **Fine-tuning overfits** → mAP on IndicDLP rises, but zero-shot drops
   - Fix: Use early stopping, regularization

3. **Rare classes still fail** → class balancing not strong enough
   - Fix: Use even more aggressive lower bounds

4. **Ablations don't show clear improvements** → components don't work together
   - Fix: Check if models were trained with same hyperparameters
```

⚠️ **Issue:** Diagnostic steps are missing.

**Should ADD (Phase 2 style):**

```
### 1. Self-Training Makes Accuracy Worse

**Symptom:**
- Round 1 mAP on BaDLAD test: 62.1% (expected: 64%+)
- Or: mAP decreases from epoch 10 to epoch 20
- Or: Training loss plateaus or increases

**Root Causes:**
1. Pseudo-labels are too noisy
   - Indicator: Visualize a few pseudo-labeled images
   - Look for: Incorrect boxes, wrong classes, spurious detections

2. Confidence threshold too low
   - Indicator: Check average confidence in pseudo-labels
   - Look for: Too many low-confidence (0.3-0.4) labels

3. Pseudo-label:Ground-truth ratio imbalanced
   - Indicator: Compare distribution in training data
   - Look for: 80% pseudo, 20% real (should be ~50/50)

4. Learning rate too high for fine-tuning
   - Indicator: Loss oscillates instead of decreasing
   - Look for: Unstable loss curve (spikes)

**Diagnostic Steps:**

Step 1: Visualize pseudo-labels
```python
import cv2
import json
import random

with open('output/logs/badlad_pseudo_labels_balanced.json') as f:
    pseudo_labels = json.load(f)

# Sample 10 random images
sample = random.sample(pseudo_labels, 10)

for item in sample:
    img_path = f'data/raw/badlad/unlabeled/images/{item["image"]}'
    img = cv2.imread(img_path)

    # Draw pseudo-label boxes
    for ann in item['annotations']:
        x1, y1, x2, y2 = map(int, ann['bbox'])
        score = ann['score']
        cls_id = ann['class']

        # Color by confidence
        if score > 0.8:
            color = (0, 255, 0)  # Green - high confidence
        elif score > 0.6:
            color = (255, 255, 0)  # Yellow - medium confidence
        else:
            color = (0, 0, 255)  # Red - low confidence

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f'{score:.2f}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

    cv2.imwrite(f'output/analysis/pseudo_label_sample_{item["image"]}', img)

print("Saved 10 samples to output/analysis/")
print("Inspect them: Do boxes look reasonable?")
```

Expected good pseudo-labels:
- Boxes align with text regions
- Scores mostly 0.7-0.95 (avoid too many <0.5)
- Classes seem correct (text_body for large text, etc.)

Step 2: Check confidence distribution
```python
from collections import defaultdict
import numpy as np

scores_by_class = defaultdict(list)

for item in pseudo_labels:
    for ann in item['annotations']:
        cls_id = ann['class']
        score = ann['score']
        scores_by_class[cls_id].append(score)

print("Confidence distribution per class:")
print(f"{'Class':<15} {'Count':<8} {'Mean Score':<12} {'Min':<8} {'Max':<8}")

for cls_id in sorted(scores_by_class.keys()):
    scores = scores_by_class[cls_id]
    print(f"{cls_id:<15} {len(scores):<8} "
          f"{np.mean(scores):.3f}        {min(scores):.2f}   {max(scores):.2f}")

# Expected output:
# Class          Count    Mean Score   Min      Max
# 0              95000    0.745        0.700    0.950
# 1              82000    0.742        0.700    0.940
# 2              45000    0.658        0.550    0.920  ← Lower for rare class
# ...

print("\nAnalysis: If any class has mean <0.60, lower the threshold for that class")
```

**Fixes (in order of likelihood):**

Fix 1: Increase confidence threshold globally
```python
# In step 2 of Phase 3, change:
class_thresholds = {
    0: 0.75,  # Was: 0.70
    1: 0.75,  # Was: 0.70
    2: 0.65,  # Was: 0.55
    ...
}
# Higher thresholds = fewer, but higher-quality pseudo-labels
# Re-generate pseudo-labels with new thresholds
```

Fix 2: Use only Round 1 results
```python
# Skip Round 2 self-training
# Fine-tune directly on:
#   - BaDLAD labeled (33K images)
#   - BaDLAD pseudo-labeled Round 1 (180K images)
# Not the re-pseudo-labeled Round 2
```

Fix 3: Adjust learning rate for self-training
```python
# In train_synthetic.py, change:
results = model.train(
    data=...,
    lr0=0.0005,  # Was: 0.001, reduce if loss is unstable
    lrf=0.0005,  # Lower final LR too
    patience=5,  # Early stopping if val mAP doesn't improve for 5 epochs
)
```

**Verification after fix:**
```bash
# Check training curve
# If loss still decreases steadily, good
# If mAP curve is improving, good

# Expected: mAP should reach 64-65% after Round 1
# If not reaching target, check if pseudo-labels are too noisy
```
```

---

## 6. VERIFICATION CHECKLISTS

### Phase 2 Has:
- Implicit expected metrics
- No explicit "check this after step" lists

### Phase 3 Missing:
- **Post-step verification checklists** like Phase 1 could benefit from

**Should ADD:**

```
## Verification Checklist After Each Self-Training Round

### After Round 1 (End of Day 10)

File existence:
  [ ] doclayout_yolo_indic_selftraining_round1.pt exists
  [ ] File size is ~180 MB (same as baseline)

Checkpoint loading:
```python
from ultralytics import YOLO
model = YOLO('output/checkpoints/doclayout_yolo_indic_selftraining_round1.pt')
results = model.val(data='data/raw/badlad/test.yaml')
print(f"Round 1 mAP: {results.results_dict['mAP50-95']:.1%}")
# Expected: 64-65%
```

Training logs:
  [ ] output/logs/selftraining_log.json exists and is valid JSON
  [ ] Training log shows loss decreasing over 20 epochs
  [ ] No error messages in training output

Model performance:
  [ ] BaDLAD test mAP: 64-65% (up from 62% pretrained)
  [ ] Training loss on last epoch: < 0.90
  [ ] GPU memory usage: 35-38 GB (stable, not increasing)

If ANY check fails:
  - See "Failure Points to Watch" section above
  - Common issue: Pseudo-labels too noisy → increase confidence threshold
  - Check GPU logs for memory leaks
```

---

## 7. KEY METRICS TRACKING

### Phase 2 Has:
- Clear metrics to track

### Phase 3 Has:
- Similar level of detail

✅ **GOOD:** Both phases clearly specify metrics.

---

## 8. ABLATION STUDY DEPTH

### Phase 3 Ablation (Current):
Shows 5 experiments and results table.

✅ **GOOD:** Well-structured ablation

⚠️ **Gap:** Doesn't explain HOW to interpret ablation results.

**Should ADD:**

```
## Understanding Ablation Results

Each ablation compares two consecutive components:

Experiment 1 vs 2: Impact of Synthetic Data
  Baseline: 60.2% → +Synthetic: 66.1%
  → Improvement: +5.9 percentage points
  → Interpretation: Synthetic Indic data helps learning

Experiment 2 vs 3: Impact of Script-Head
  +Synthetic: 66.1% → +ScriptHead: 67.6%
  → Improvement: +1.5 percentage points
  → Interpretation: Auxiliary loss provides modest boost
  → Why modest? Script info somewhat redundant with layout features

Experiment 3 vs 4: Impact of Self-Training
  +ScriptHead: 67.6% → +SelfTraining: 72.8%
  → Improvement: +5.2 percentage points
  → Interpretation: Unlabeled data is valuable for Indic tasks
  → Why large? BaDLAD-unlabeled has 4M images (leverage scale)

Experiment 4 vs 5: Impact of Fine-Tuning
  +SelfTraining: 72.8% → Full (FineTuned): 75.5%
  → Improvement: +2.7 percentage points
  → Interpretation: IndicDLP-specific tuning adds final polish
  → Why smaller? Already adapted to Indic via pretraining + self-training

**Total Improvement Decomposition:**
Baseline (60.2%)
  + Synthetic Data        (+5.9%)
  + Script-Head           (+1.5%)
  + Self-Training         (+5.2%)
  + Fine-Tuning           (+2.7%)
─────────────────
= Full System (75.5%)

This shows: Self-training is biggest contributor (+5.2%),
followed by synthetic pretraining (+5.9%), then fine-tuning (+2.7%),
then script-awareness (+1.5%).

For publication: Include this breakdown in paper to show
which components contribute most to improvement.
```

---

## 9. PHASE 3 vs PHASE 2 STANDARDS COMPLIANCE TABLE

| Aspect | Phase 2 Standard | Phase 3 Current | Score | Gap |
|--------|---|---|---|---|
| **Structure** | 8 sections | 8+ sections (better organized by weeks) | 95% | Excellent structure |
| **Code Examples** | 11 blocks | 8 blocks | 75% | Missing 2-3 format examples |
| **Expected Outputs** | Explicit files shown | Vague in some sections | 70% | Need JSON formats |
| **JSON/CSV Formats** | Not all shown | Not shown for results | 40% | ADD: ablation, pseudo-label formats |
| **YAML Config Format** | N/A | Not shown | 0% | ADD: data mixing YAML file |
| **Failure Points** | Not detailed (improvement area) | One-liner + fix | 50% | Expand with diagnostic steps |
| **Verification Steps** | Not explicit | Not present | 0% | ADD: post-step verification |
| **Hyperparameter Table** | Present (Phase 2) | Mentioned in code | 75% | Could be clearer table |
| **Completion Checklist** | Present | Present | 90% | Good |
| **Metrics Tracking** | Present | Present | 85% | Good |
| **Ablation Interpretation** | N/A | Not explained | 40% | ADD: how to read ablation results |
| **Learning Rate Scheduling** | Not in Phase 2 | Mentioned briefly | 60% | Could expand |

**Overall Phase 3 Compliance: 74%**

---

## 10. RECOMMENDED IMPROVEMENTS FOR PHASE 3

### Priority 1 (Must Add):
1. ✅ Add YAML data.yaml example for self-training dataset mixing
2. ✅ Add complete JSON output structures (pseudo-labels, ablation results)
3. ✅ Expand failure points with diagnostic code
4. ✅ Add post-step verification checklists
5. ✅ Add "How to read ablation results" interpretation guide

### Priority 2 (Should Add):
1. ✅ Add class-balanced threshold computation detailed logic
2. ✅ Add GPU memory monitoring section
3. ✅ Add learning rate schedule explanation
4. ✅ Add sample numbers from Round 1 vs Round 2 comparison

### Priority 3 (Nice to Have):
1. ✅ Add visualization code for pseudo-label quality
2. ✅ Add pseudo-label statistics analysis code
3. ✅ Add convergence criteria explanation

---

## 11. SUMMARY OF PHASE 3 COMPLIANCE

**Strengths:**
- ✅ Excellent week-by-week organization
- ✅ Good step-level granularity
- ✅ Solid code examples
- ✅ Clear ablation structure
- ✅ Good completion checklist
- ✅ Good metrics to track

**Weaknesses:**
- ❌ JSON output formats not shown (pseudo-labels, ablation results)
- ❌ YAML config file not specified
- ❌ Failure points lack diagnostic depth
- ❌ No post-step verification checklists
- ❌ Ablation results interpretation missing

**Verdict:** Phase 3 is **74% compliant** with Phase 2 standards. It's well-structured but needs format examples and verification guidance.

**Estimated effort to reach 85%+ compliance:** ~3-4 hours of additions.

**Key addition:** JSON format specifications for pseudo-labels and ablation results are critical for reproducibility.