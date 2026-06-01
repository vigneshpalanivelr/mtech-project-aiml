# DAILY TRACKER: JUNE 1 - JULY 31, 2026
## Project Completion Checklist

---

## JUNE 1 STATUS — PHASE 1 IN PROGRESS
**Platform: Google Colab Pro (500 compute units)**

### Phase 1 Checklist:
- [ ] Baseline DocLayout-YOLO reproduced
- [ ] Baseline mAP: 70%+
- [ ] Zero-shot mAP: 75%+
- [ ] Checkpoint backed up to Google Drive
- [ ] GitHub updated
- [ ] College cluster confirmed unavailable → committed to Google Colab Pro (500 units)

**Work through Phase 1 first. See JUNE01.md for the Colab notebook setup.**

---

## WEEK 1 — PHASE 1: SETUP & BASELINE (June 1-8)
### Colab units target: ~20 | Units remaining: ~500

---

## WEEK 2-3: SYNTHETIC DATA GENERATION (June 9-22)
### Colab Target: ~30 compute units | Project Time: 14 days
### Environment: Google Colab Pro + Google Drive

**June 9-10 (SUN-MON) — Colab Setup + Test Generation**
- [ ] Open Colab, mount Drive, run JUNE01.md Cell 1 (session setup)
- [ ] Install packages: uharfbuzz, pillow, ultralytics, tqdm (Cell 1 handles this)
- [ ] Download all 10 Noto Sans fonts to Drive (JUNE01.md Cell 3)
- [ ] Download text corpus for all 10 scripts (JUNE01.md Cell 4)
- [ ] Write src/ modules to Drive using %%writefile (Cells 5-8)
- [ ] Generate 10 test documents and visually inspect (Cell 9)
- **Status:** Synthetic generation ready

**June 11-15 (TUE-SAT) — Full Generation (run off-peak 8 PM - 8 AM UTC)**
- [ ] Run full generation: 50,000 documents (JUNE01.md Cell 10)
  - Generation is CPU-heavy, ~3-4 hours total
  - Auto-resumes if Colab disconnects (tracks already_done count)
  - Each doc ~50 KB, total ~2.5 GB in Drive
- [ ] Run keep-alive in separate Colab tab during generation
- [ ] Monitor Drive storage: !du -sh /content/drive/MyDrive/doclayout-yolo-indic/
- **Status:** Synthetic dataset created

**June 16-22 (SUN-SAT) — COCO Manifest + Pretraining (off-peak)**
- [ ] Create COCO train/val manifests (JUNE01.md Cell 11)
- [ ] Create data.yaml for training (Cell 12)
- [ ] Start pretraining (30 epochs, JUNE01.md Cell 13)
  - Batch size: 16 (Colab GPU memory constraint)
  - Runs ~15-20 hours off-peak (8 PM - 8 AM UTC)
  - Uses ~20-25 Colab compute units
- [ ] Monitor validation mAP (should increase steadily)
- [ ] Save best checkpoint to Drive: output/checkpoints/doclayout_yolo_indic_pretrained.pt
- **Status:** Pretraining complete

**June 22 EOD CHECKPOINT:**
- [ ] 50K synthetic images generated in Drive ✓
- [ ] COCO annotations created ✓
- [ ] Pretraining complete ✓
- [ ] Pretrained mAP: 60%+ ✓
- [ ] Checkpoint backed up in Drive ✓
- [ ] GitHub updated ✓
- **Colab units used: ~50 | Units remaining: ~450**

---

## WEEK 4-5: SELF-TRAINING & FINE-TUNING (June 23-July 6)
### Colab Target: ~60 compute units | Project Time: 14 days
### Environment: Google Colab Pro + Google Drive

**June 23 (SUNDAY)**
- [ ] Download IndicDLP (40GB) to Colab runtime: !wget ... -O /content/indicdlp.tar.gz
  - Extract to Drive: /content/drive/MyDrive/doclayout-yolo-indic/data/raw/IndicDLP/
- [ ] Download BaDLAD partial (25GB) similarly to Drive
- [ ] Check Drive storage: !du -sh /content/drive/MyDrive/doclayout-yolo-indic/
  - Delete output/synthetic/images/ AFTER pretraining is confirmed (saves ~2.5 GB)
- [ ] Organize data/raw/ structure in Drive
- **Status:** Datasets ready for self-training

**June 24-27 (MON-THU)**
- [ ] In Colab: load Phase 2 pretrained checkpoint from Drive
- [ ] Run inference on BaDLAD-unlabeled (off-peak, ~3 hours)
  - No training, just inference → fast and cheap (~5 units)
  - Save predictions with confidence scores to Drive JSON
- [ ] Generate pseudo-labels with class-balanced thresholds
- [ ] Create pseudo-labeled dataset
- [ ] Check pseudo-label quality: visualize 10 random samples in Colab
- **Status:** Pseudo-labels ready

**June 28-Jul 2 (FRI-WED)**
- [ ] Train Round 1 self-training (20 epochs, off-peak)
  - Mix: 50% IndicDLP labeled + 50% pseudo-labeled BaDLAD
  - batch=16, runs ~20 hours = ~20 Colab units (off-peak)
  - Monitor: Check if validation mAP improves
- [ ] Evaluate Round 1 checkpoint from Drive
- [ ] Decision Point (June 30 evening):
  - If ahead of schedule: Proceed to Round 2
  - If on schedule: Skip Round 2, go to fine-tuning
  - If behind: Skip Round 2 definitely

**July 3-6 (THU-SUN)**
- [ ] Fine-tune on IndicDLP labeled (20 epochs, off-peak)
  - Load best checkpoint from self-training (from Drive)
  - Train on labeled images only, early stopping patience=3
  - ~20 hours = ~20 Colab units (off-peak)
- [ ] Evaluate final model
- [ ] Save final checkpoint to Drive: output/checkpoints/doclayout_yolo_indic_finetuned.pt
- **Status:** Phase 3 complete

**July 6 EOD CHECKPOINT:**
- [ ] Self-training complete ✓
- [ ] Fine-tuning complete ✓
- [ ] Final mAP on IndicDLP: 72-75% ✓
- [ ] Checkpoint saved to Drive ✓
- [ ] GitHub updated ✓
- **Colab units used: ~120 | Units remaining: ~380**

---

## WEEK 6-7: EVALUATION & ANALYSIS (July 7-20)
### Colab Target: ~15 compute units | Project Time: 14 days
### Environment: Google Colab Pro + Google Drive (inference only, no heavy training)

**July 7-8 (MON-TUE)**
- [ ] Evaluate on IndicDLP test (full evaluation)
  - Per-script breakdown
  - Per-class metrics
  - Confusion matrix
- [ ] Evaluate on BaDLAD test (cross-domain)
- [ ] Evaluate on DocLayNet (regression check)
- [ ] Save all results to JSON files
- **Status:** Evaluation complete

**July 9-12 (WED-SAT)**
- [ ] Run ablation comparisons (3 experiments):
  1. Baseline (English zero-shot)
  2. + Synthetic
  3. Full system
- [ ] Record all metrics
- [ ] Create comparison table
- **Status:** Ablation done

**July 13-15 (SUN-TUE)**
- [ ] Collect hard examples (misclassified images)
- [ ] Create failure visualization grid
- [ ] Analyze failure patterns by:
  - Script (which scripts fail most?)
  - Class (which object types fail?)
  - Domain (which document types?)
- [ ] Write failure analysis summary
- **Status:** Failure analysis complete

**July 16-18 (WED-FRI)**
- [ ] Compile all results into tables
- [ ] Create matplotlib figures/plots
- [ ] Organize results directory
- [ ] Prepare results summary document
- **Status:** Results ready for paper

**July 20 EOD CHECKPOINT:**
- [ ] Evaluation complete on 3 datasets ✓
- [ ] Ablation study done ✓
- [ ] Failure analysis complete ✓
- [ ] Results compiled ✓
- [ ] Figures ready ✓
- **Colab units used: ~145 | Units remaining: ~355**

---

## WEEK 8: FINAL SUBMISSION (July 21-31)
### GPU Target: 0 hours | Project Time: 11 days

**July 21-23 (MON-WED) - Paper Writing**
- [ ] Write paper skeleton (8 pages)
- [ ] Section 1: Introduction (1 page)
- [ ] Section 2: Background (0.5 pages)
- [ ] Section 3: Methodology (1.5 pages)
- [ ] Section 4: Experiments (1 page)
- [ ] Section 5: Results (2 pages)
- [ ] Section 6: Analysis (1 page)
- [ ] Section 7: Conclusion (0.5 pages)
- **Status:** Paper draft complete

**July 24-26 (THU-SAT) - Code Release**
- [ ] Final code cleanup (remove debug code)
- [ ] Update README with installation + usage + results
- [ ] Add requirements.txt
- [ ] Create reproducibility guide
- [ ] Final tests in Colab (verify code works end-to-end)
- [ ] Push final code to GitHub
- [ ] Tag release (v1.0)
- **Status:** Code released

**July 27-29 (SUN-TUE) - Dissertation**
- [ ] Format dissertation (BITS template)
- [ ] Compile all sections (intro, chapters, results, appendix)
- [ ] Add figures and tables
- [ ] Proofread (spell-check, grammar)
- [ ] Get supervisor signature
- [ ] Submit to department
- **Status:** Dissertation submitted

**July 30-31 (WED-THU) - Final Wrap-Up**
- [ ] Verify all submissions received
- [ ] Backup everything to Google Drive
- [ ] Archive project (GitHub + Drive)
- [ ] Write project summary (for records)
- [ ] Document lessons learned
- [ ] Celebrate! 🎉

**July 31 EOD FINAL CHECKPOINT:**
- [ ] Paper written ✓
- [ ] Code released ✓
- [ ] Dissertation submitted ✓
- [ ] Project COMPLETE ✓

---

## DAILY STATUS TEMPLATE (USE THIS FORMAT)

```
═══════════════════════════════════════════════════════════════
DATE: [Date]
PHASE: [Phase 1-5]
═══════════════════════════════════════════════════════════════

✓ COMPLETED TODAY:
  - [Task 1]
  - [Task 2]

⏳ IN PROGRESS:
  - [Task 3] (ETA: [time])

📋 PLANNED TOMORROW:
  - [Task 4]
  - [Task 5]

⚠️ BLOCKERS/ISSUES:
  - [None] / [Issue description]

📊 METRICS:
  - GPU hours used: [X]h
  - Storage used: [X]GB
  - Code commits: [X]
  - Results mAP: [X]%

💬 NOTES:
  [Any important notes]

═══════════════════════════════════════════════════════════════
```

---

## CRITICAL MILESTONES (CANNOT SLIP)

| Milestone | Date | Status | Action If Missed |
|-----------|------|--------|-----------------|
| Phase 1 complete | Jun 8 | In progress | Extend timeline if needed |
| Phase 2 complete | Jun 22 | Pending | Compress Phase 3 timeline |
| Phase 3 complete | Jul 6 | Pending | Skip ablations if needed |
| Phase 4 complete | Jul 20 | Pending | Fast-track paper writing |
| Submit | Jul 31 | HARD DEADLINE | PROJECT FAILS if missed |

**Platform: Google Colab Pro — 500 compute units available (~450 remaining after Phase 1)**

---

## WEEKLY METRICS TO TRACK

### Every Sunday Evening, Fill This Out:

**Week of June [X]:**
- GPU hours used: [20/20] ✓
- Dataset downloaded: [Yes/No]
- Code commits: [X]
- Phase complete: [Yes/No]
- On schedule: [Yes/No/Behind by X days]
- Issues: [None / List]

**Week of June [X]:**
- GPU hours used: [40/40] ✓
- Synthetic generated: [100K ✓]
- Pretraining complete: [Yes/No]
- On schedule: [Yes/No/Behind by X days]
- Issues: [None / List]

[etc. for all 8 weeks]

---

## EMERGENCY CONTACTS

| Name | Role | Contact | When to Use |
|------|------|---------|-----------|
| College IT | Tech support | [Phone/Email] | Cluster issues |
| Supervisor | Academic | [Email] | Guidance, approvals |
| [Self] | Project lead | [Phone] | Daily tracking |

---

## WHAT TO DO IF YOU GET BEHIND

**1-2 days behind:**
- Skip nice-to-haves (remove ablation experiments)
- Use simpler evaluation (2 datasets instead of 3)

**3-5 days behind:**
- Reduce synthetic images (100K → 75K)
- Skip self-training Round 2
- Minimal paper (focus on results only)

**>5 days behind:**
- Consider extending deadline (talk to supervisor)
- Switch focus to quality over quantity
- Get help from supervisor/classmates

---

**PRINT THIS AND TRACK DAILY!**

Your success depends on discipline and tracking.

Good luck! 💪📅