# TASK — Build a dual-mode student package (TRAIN-from-scratch OR REPRODUCE-from-stored)

## Objective
Produce ONE portable folder, e.g. `FINAL_METHOD_DUAL_MODE/`, that gives the student TWO independent choices,
each with its OWN Word instructions file:
- **Choice A — Train from the beginning** → `INSTRUCTIONS_TRAIN_FROM_SCRATCH.docx`
- **Choice B — Reproduce from the stored result CSVs / frozen checkpoint** → `INSTRUCTIONS_REPRODUCE_FROM_STORED.docx`

Use the REAL source code only. No wrappers, no synthetic scripts, no hard-coded/copied result values, no mocks.
Shell `RUN_*.sh` may wrap the real `.py`, but must invoke it.

## Fixed context (do not change)
- Repo: `/Users/moahaimentalib/Desktop/f_flex_network_code_clean` · remote `github.com/moahaimen/f_flex_network_code`
- Final code branch: `hardening/periodic-keepmask-inf` (commit `5455988` = the `-inf` KEEP-mask hardening).
- Method: **Reward-Gated GNN-LPD Traffic Engineering (RG-GNN-LPD), FIX1 strict-all.**
  Failure rule (periodic AND 20-TM, identical): active link failure → KEEP removed via `float('-inf')` →
  DDQN selects only K50,K100,K200,K300,K500,K800.
- Governing report: `...FINAL_REPORT_FIX1/RG_GNN_LPD_FIX1_Final_Report_200VTL_N3976_FINAL_METHOD_AUDITED_PRREF_STRICTALL_MLU_UPGRADED.docx`
  DOCX SHA256 `65a43d5e8e648fed8f7ae2c7855dc82fb6797861f89eea15ef81f96e5209533d`.
- Section 13 (Mininet/SDN) is FROZEN from the report; not runnable. Baselines + reported ms are FROZEN.

====================================================================
## CHOICE B (REPRODUCE FROM STORED) — already built; reuse it
====================================================================
The self-contained package **`FINAL_METHOD_REAL_CODE_RERUN`** (zip 52 MB on Desktop) IS Choice B:
- `repo/` = real code + the FROZEN final checkpoints + real data (`.npz` TMs, topology, path caches) +
  stored strict full-MCF caches.
- `LIVE_REPRODUCIBLE_RESULTS/` = the stored per-cycle result CSVs from the last report.
- `manifest/GOVERNING_REPORT_VALUES.json` + `scripts/verify_core.py` = exact verifier.
- `RUN_NORMAL_FINAL_METHOD.sh`, `RUN_FAILURE_RESULTS_EXACT.sh`, `VERIFY_FINAL_METHOD_AGAINST_REPORT.sh`.
- Verified: **PASS 823 / FROZEN 438 / FAIL 0 / PENDING 0**, KEEP=0 on every active failure cycle; reproduces
  the report EXACTLY (e.g. pooled N=3976 = 98.519 / 98.390 / 88.657; vtl-200 = 94.982).
Copy this in as the `choiceB_reproduce/` subfolder (it is already portable and self-contained —
confirm 0 symlinks and 0 external file reads with an audit hook before shipping).

Real EVAL runners (used by Choice B):
- Normal N=3976 → `scripts/phase1_5/run_fix1_completed_metrics.py::eval_method` (cache_final; computes vtl 40-199 live)
- 20-TM 6-scenario failure → `scripts/phase1_5/run_keepmask_stress.py`
- Periodic single-link failure → `scripts/phase1_5/run_periodic_failure_masked.py`
- Strict full-MCF numerator → `scripts/phase1_5/strict_mcf_source_agg.py`

====================================================================
## CHOICE A (TRAIN FROM SCRATCH) — assemble
====================================================================
Include the REAL training scripts (verify each before shipping; do NOT invent):
- GNN-LPD scorer → **`scripts/phase1_5/train_gnn_dbbudget_selector.py`** → `gnn_dbbudget_selector.pt`
- DDQN (FIX1) → **`scripts/phase1_5/run_fulldata_gated_fix1.py`** (finetune + validation selection) → `fulldata_gated_model.pt`
- Frozen teacher (iter2) → **`scripts/phase1_5/run_fulldata_gated_preserved.py`** / `run_final_iter2.py` → `final_learned_4of5_iter2_model.pt`
- Experience / prepass precompute → `scripts/phase1_5/precompute_failure_experiences.py`, `bottleneck_precompute.py`
  (and any prepass builder the above import).
- Base env / models: `scripts/phase1_5/gnn_lpd_dqn_selective_db_lp.py`, `agnostic_lib.py`, `bottleneck_lib.py`,
  `te/*`, `phase1_reactive/*`.

Include the raw TRAINING INPUTS (topologies, traffic matrices, path caches) but do NOT ship the pre-trained
`.pt` checkpoints in Choice A (the student regenerates them). Data needed (copy real files, no symlinks):
`data/processed/*.npz`, `data/processed/path_cache/*_k3_paths.pkl`, `data/raw/topology/*`, `data/rocketfuel/*.txt`,
`data/topologyzoo/*.graphml`, `data/raw/traffic/cernet_tm.npz`.

Flow for the student in Choice A:
1. Train GNN scorer → 2. Precompute experiences/prepass → 3. Train DDQN (fix1 + teacher) →
4. Run the SAME eval runners (eval_method + failure runners) on THEIR fresh checkpoints → 5. See results.

### MANDATORY honesty caveat (must be printed in INSTRUCTIONS_TRAIN_FROM_SCRATCH.docx, bold, near the top)
Training is stochastic (experience-replay sampling, RNG seeds, CBC/HiGHS tie-breaks). A fresh training run
produces a DIFFERENT checkpoint → results will be **similar but NOT identical** to the governing report.
The report's EXACT numbers (98.519 pooled, KEEP=0, vtl-200 = 94.982, etc.) are tied to the specific FROZEN
checkpoint, which is why exact reproduction requires **Choice B**. Preserve `CBC_SEED=42` and the code's
`set_seed(42)`, but do NOT claim exact-report reproduction for Choice A. Do NOT retrain-then-overwrite the
frozen checkpoint used by Choice B.

====================================================================
## Deliverable structure
====================================================================
```
FINAL_METHOD_DUAL_MODE/
  choiceA_train_from_scratch/   real training code + raw inputs (NO pre-trained .pt) + RUN_TRAIN_*.sh
  choiceB_reproduce/            = FINAL_METHOD_REAL_CODE_RERUN (real code + frozen ckpts + data + stored CSVs + verifier)
  INSTRUCTIONS_TRAIN_FROM_SCRATCH.docx
  INSTRUCTIONS_REPRODUCE_FROM_STORED.docx
  README.md
```

## Requirements checklist
- [ ] Real `.py` only; RUN_*.sh call the real runners.
- [ ] Portable: no symlinks pointing outside the folder; copy real data in; verify 0 external file reads with a
      `sys.addaudithook` on `open` (the prior package had a data-symlink bug — do not repeat it).
- [ ] Each choice runs end-to-end from inside the folder with only Python 3.12 + the pinned packages installed.
- [ ] Two DOCX files (one per choice), each: prerequisites, exact copy-paste Terminal commands, expected runtime,
      expected outputs, and (for Choice A) the bold non-exact caveat.
- [ ] Choice B verifier still returns PASS 823 / FROZEN 438 / FAIL 0 / PENDING 0.
- [ ] Do NOT change the method, the frozen checkpoint, seeds, scenario definitions, the strict full-MCF PR
      numerator, or any report number.

## Return at completion
1. Folder path + both DOCX filenames.
2. List of real training `.py` files (Choice A) and real eval `.py` files (Choice B).
3. Checkpoint provenance: which script trains each `.pt`.
4. Confirmation Choice B reproduces the report exactly (verifier output).
5. Explicit statement that Choice A yields similar-not-identical results, with the reason.
6. Proof (audit) that neither choice reads any file outside the folder.
