# DAILY TRACKER: JUNE 1 - JULY 31, 2026
## Project Completion Checklist

---

## TODAY - JUNE 1 (SATURDAY)
**Critical Setup Day - MUST DO TODAY**

### Morning (06:00-10:00)
- [ ] Read EMERGENCY_JUNE_JULY_TIMELINE.md (30 min)
- [ ] Call College IT for cluster access (30 min)
  - Phone: [IT number]
  - Ask: "Need cluster access by June 3 for M.Tech project"
- [ ] Email IT formally + CC supervisor (30 min)

### Afternoon (10:00-14:00)
- [ ] Download all phase context documents
- [ ] Test Phase 1 baseline code locally (1 hour)
- [ ] Create/update GitHub repo
- [ ] Push code to GitHub with README

### Evening (14:00-18:00)
- [ ] START D⁴LA download (10GB) - run overnight
  - Use wget or torrent
  - Target: External drive or /downloads/
- [ ] Notify supervisor about timeline change
- [ ] Review COLAB_VS_CLUSTER_DECISION_MATRIX.txt

### EOD June 1
- [ ] Cluster request submitted
- [ ] Code pushed to GitHub
- [ ] D⁴LA download started
- [ ] Supervisor notified
- **Status: ✓ Setup in progress**

---

## WEEK 1: SETUP & PHASE 1 (June 1-8)
### GPU Target: 20 hours | Project Time: 8 days

**June 2 (SUNDAY)**
- [ ] Verify D⁴LA download (check if complete)
- [ ] Follow up with College IT (if no response)
- [ ] Prepare backup: Set up Colab Pro account (just in case)
- [ ] Review Phase 1 commands (baseline training)
- **Status:** Waiting for cluster access

**June 3 (MONDAY)** ← CRITICAL MILESTONE
- [ ] Receive cluster credentials (EXPECTED)
- [ ] SSH into cluster
- [ ] Set up conda environment
  ```bash
  conda create -n doclayout python=3.10
  conda activate doclayout
  pip install ultralytics opencv-python pyyaml pycocotools
  ```
- [ ] Test GPU: `nvidia-smi`
- [ ] Mount scratch directory
- **Status:** Cluster should be ready by EOD

**June 4 (TUESDAY)**
- [ ] Transfer D⁴LA to cluster (or download directly)
- [ ] Download DocLayNet (30GB) to cluster
- [ ] Check storage usage: `du -sh /scratch/`
- [ ] Create data.yaml files
- **Status:** Datasets staged on cluster

**June 5 (WEDNESDAY)**
- [ ] Verify dataset integrity (file counts, sizes)
- [ ] Organize /data/raw/ structure
- [ ] Run first test training (1 epoch, quick test)
- [ ] Check loss curves (make sure no errors)
- **Status:** Ready for full training

**June 6 (THURSDAY)**
- [ ] Start baseline training (50 epochs, 20 GPU-hours)
  - This will take ~20 hours continuous
  - Monitor loss curve (check hourly first 2 hours)
- [ ] Set up monitoring script
- **Status:** Training in progress

**June 7-8 (FRI-SAT)**
- [ ] Monitor training progress (email alerts if GPU crashes)
- [ ] Once training complete: evaluate on D⁴LA test
- [ ] Evaluate zero-shot on DocLayNet
- [ ] Save results to Phase_1_RESULTS.json
- [ ] Backup checkpoint to Google Drive
- [ ] Delete D⁴LA + DocLayNet (free 40GB space)
- **Status:** Phase 1 COMPLETE ✓

**June 8 EOD CHECKPOINT:**
- [ ] Phase 1 complete
- [ ] Baseline mAP: 70%+ ✓
- [ ] Zero-shot mAP: 75%+ ✓
- [ ] Checkpoint backed up ✓
- [ ] GitHub updated ✓
- **GPU-hours used: 20 | Remaining: 140**

---

## WEEK 2-3: SYNTHETIC DATA GENERATION (June 9-22)
### GPU Target: 40 hours | Project Time: 14 days

**June 9-10 (SUN-MON)**
- [ ] Install HarfBuzz + fonts on cluster
- [ ] Prepare layout templates (15 templates)
- [ ] Download text corpora (Hindi, Tamil, Bengali, Telugu)
- [ ] Generate 100 test images
- [ ] Verify image quality (visual check)
- **Status:** Synthetic generation ready

**June 11-15 (TUE-SAT)**
- [ ] Generate 100K synthetic images in batches
  - Batch 1-10: 10K images each
  - Monitor storage (each batch ~3GB)
  - Delete after COCO annotation created
- [ ] Generate COCO annotations for each batch
- [ ] Merge all annotations into single file
- **Status:** Synthetic dataset created

**June 16-22 (SUN-SAT)**
- [ ] Create data.yaml for synthetic dataset
- [ ] Start pretraining (50 epochs, ~20 GPU-hours)
  - Batch size: 32
  - Learning rate: 0.01
  - Image size: 1280
- [ ] Monitor validation mAP (should increase steadily)
- [ ] Save best checkpoint
- [ ] Backup to Google Drive
- **Status:** Pretraining complete

**June 22 EOD CHECKPOINT:**
- [ ] 100K synthetic images generated ✓
- [ ] Pretraining complete ✓
- [ ] Pretrained mAP: 65%+ ✓
- [ ] Checkpoint backed up ✓
- [ ] GitHub updated ✓
- **GPU-hours used: 60 | Remaining: 100**

---

## WEEK 4-5: SELF-TRAINING & FINE-TUNING (June 23-July 6)
### GPU Target: 60 hours | Project Time: 14 days

**June 23 (SUNDAY)**
- [ ] Download IndicDLP (40GB) to cluster
- [ ] Download BaDLAD partial (25GB) to cluster
- [ ] Archive synthetic data to Google Drive (free space)
- [ ] Organize /data/raw/ for Phase 3
- **Status:** Datasets ready for self-training

**June 24-27 (MON-THU)**
- [ ] Load Phase 2 pretrained checkpoint
- [ ] Run inference on BaDLAD-unlabeled (200K images)
  - This is fast (no training, just inference)
  - Save predictions with confidence scores
- [ ] Generate pseudo-labels with class-balanced thresholds
- [ ] Create pseudo-labeled dataset
- [ ] Check pseudo-label quality (sample visualizations)
- **Status:** Pseudo-labels ready

**June 28-Jul 2 (FRI-WED)**
- [ ] Train Round 1 self-training (20 epochs, 20 GPU-hours)
  - Mix: 50% IndicDLP labeled + 50% pseudo-labeled BaDLAD
  - Monitor: Check if validation mAP improves
- [ ] Evaluate Round 1 checkpoint
- [ ] **Decision Point (June 30 evening):**
  - If ahead of schedule: Proceed to Round 2
  - If on schedule: Skip Round 2, go to fine-tuning
  - If behind: Skip Round 2 definitely

**July 3-6 (THU-SUN)**
- [ ] Fine-tune on IndicDLP labeled (20 epochs, 20 GPU-hours)
  - Load best checkpoint from self-training
  - Train on 40K labeled images only
  - Use early stopping (patience=3)
- [ ] Evaluate final model
- [ ] Save final checkpoint
- [ ] Backup to Google Drive
- **Status:** Phase 3 complete

**July 6 EOD CHECKPOINT:**
- [ ] Self-training complete ✓
- [ ] Fine-tuning complete ✓
- [ ] Final mAP on IndicDLP: 72-75% ✓
- [ ] Checkpoint backed up ✓
- [ ] GitHub updated ✓
- **GPU-hours used: 120 | Remaining: 40**

---

## WEEK 6-7: EVALUATION & ANALYSIS (July 7-20)
### GPU Target: 25 hours | Project Time: 14 days

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
- **GPU-hours used: 145 | Remaining: 15**

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
- [ ] Final tests on cluster (verify code works)
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
| Cluster access | Jun 3 | MUST HAVE | Activate Colab backup |
| Phase 1 complete | Jun 8 | MUST HAVE | Start Phase 2 immediately |
| Phase 2 complete | Jun 22 | MUST HAVE | Compress Phase 3 timeline |
| Phase 3 complete | Jul 6 | MUST HAVE | Skip ablations if needed |
| Phase 4 complete | Jul 20 | MUST HAVE | Fast-track paper writing |
| Submit | Jul 31 | HARD DEADLINE | PROJECT FAILS if missed |

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