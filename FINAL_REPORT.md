# Final Report — GNN-LPD DQN Selective DB-budgeted LP
**Method:** `gnn_lpd_dqn_selective_db_lp`  
**Date:** 2026-06-17  
**Evaluation tag:** `final_N3976`  
**Audit status:** PASSED (all 10 blocks)

---

## Compliance Statement

This is the **professor-compliant clean method**. Numbers in this report come from the
method's own evaluation and audit — they are NOT copied from the legacy accepted report.

The legacy accepted report (`reward_policy_selector_st_lam005/`) is audited separately
by `reproduce_legacy_reward_gated_gnn_lpd.py` / `audit_legacy_reward_gated_gnn_lpd_report.py`.
These two methods are strictly separated and their results must never be mixed.

---

## Method Summary

| Component | Implementation | Audit flag |
|---|---|---|
| Criticality selector | `GNNFlowSelector` (PyTorch GraphSAGE, 72,886 params) | `gnn_used=1, lpd_used=1` |
| Oracle label source | Full-OD DB-budgeted LP (`solve_selected_path_lp_dbbudget`, db=0.10) | `db_budgeted_oracle_used=True` |
| Action controller | Double DQN (K30/K40/K50 × DB-budget matrix, 9 actions) | `dqn_used=1` |
| LP solver | One-stage path LP on selected ODs; K-escalation 30→40→50→full-OD | `selected_od_lp_used=1` |
| Forbidden components | None active | `rf_gate=0, sticky=0, stage2=0, disturbance_fin=0, heuristic=0` |

**GNN training:**
- Labels: `solve_selected_path_lp_dbbudget` (NOT heuristic bottleneck ranking)
- `reroute_mass = demand × L1_split_distance(lp_split, ecmp_split)`; top-50% → `label_useful=1`
- Training: **6 topologies × 40 cycles = 279,360 rows** (includes CERNET); all LP calls Optimal
- Best val_prec@K = **0.7474** (25 epochs)

**DQN training:**
- Action space: KEEP / OPTIMIZE_K{30,40,50}_DB_{0.01,0.03} / FULL_OD_FALLBACK_{PR_SAFE,LOW_MLU}
- Trained on: abilene, cernet, geant, ebone (200 episodes, val_PR=1.0000 throughout)
- FULL_OD_FALLBACK excluded from random exploration to prevent slow-topology stalls
- Best val PR: **1.0000**

---

## SCOPE A — Direct FlexDATE Clean Comparison

**Topologies: Abilene, CERNET, GEANT, Sprintlink — N = 3,088 test cycles**

> Tiscali is excluded from the direct FlexDATE claim: no consistent FlexDATE reference
> row exists for Tiscali in the locked source material.

| Topology | N | Mean PR | FlexDATE PR | PR Margin | Mean DB | FlexDATE DB | DB Reduction | WIN BOTH |
|---|---|---|---|---|---|---|---|---|
| abilene | 2016 | **0.9967** | 0.958 | +3.9 pp | **0.00613** | 0.0513 | ×8.4 lower | ✅ |
| cernet | 200 | **0.9942** | 0.975 | +1.9 pp | **0.00321** | 0.0183 | ×5.7 lower | ✅ |
| geant | 672 | **0.9998** | 0.995 | +0.5 pp | **0.00378** | 0.0296 | ×7.8 lower | ✅ |
| sprintlink | 200 | **1.0000** | 0.999 | +0.1 pp | **0.00471** | 0.0510 | ×10.8 lower | ✅ |
| **Pooled** | **3,088** | **0.9975** | | | **0.00548** | | | **4/4 WIN BOTH** |

**Win summary:** The clean method wins BOTH PR and DB vs. FlexDATE on all 4 direct-comparison
topologies (Abilene, CERNET, GEANT, Sprintlink).

---

## SCOPE B — Full Internal Clean Evaluation

**All 8 topologies — N = 3,976 test cycles**

| Topology | N | Mean PR | Min PR | Mean DB | p95 DB | Full-OD% | Dec. ms (mean) |
|---|---|---|---|---|---|---|---|
| abilene | 2016 | 0.9967 | 0.9604 | 0.00613 | 0.01111 | 2.3% | 7.1 |
| cernet | 200 | 0.9942 | 0.9772 | 0.00321 | 0.00184 | 1.5% | 23.3 |
| geant | 672 | 0.9998 | 0.9962 | 0.00378 | 0.00750 | 3.1% | 18.9 |
| sprintlink | 200 | 1.0000 | 1.0000 | 0.00471 | 0.00290 | 11.5% | 155.5 |
| ebone | 200 | 1.0000 | 1.0000 | 0.00125 | 0.00193 | 0.0% | 14.2 |
| tiscali | 200 | 1.0000 | 0.9995 | 0.00714 | 0.01212 | 81.5% | 4340.4 |
| germany50 | 288 | 0.9981 | 0.9548 | 0.01061 | 0.01823 | 5.9% | 73.7 |
| vtlwavenet2011 | 200 | 0.9913 | 0.9882 | 0.00230 | 0.00082 | 0.5% | 128.3 |
| **Pooled** | **3,976** | **0.9974** | **0.9548** | **0.00545** | | **6.9%** | **246.6** |

All 8 topologies: min PR ≥ 0.9548, mean PR ≥ 0.9913, mean DB ≤ 0.011.

---

## Component Usage Statistics

| Metric | Value |
|---|---|
| GNN usage rate | **1.000** (every cycle, all 3,976 rows) |
| LPD usage rate | **1.000** (every cycle, all 3,976 rows) |
| Heuristic used rate | **0.000** |
| Full-OD fallback rate | 6.9% (pooled; 81.5% for tiscali — see note) |
| K-escalation rate | 6.3% |
| Mean decision time (pooled) | 246.6 ms |
| p95 decision time (pooled) | 657.2 ms |

**Tiscali latency note:** Tiscali (2352 ODs) was not in the DQN training set.
The DQN generalises conservatively (81.5% full-OD fallback). Full-OD LP for tiscali
takes ~4–9s per cycle, driving its mean to 4340 ms and p95 to 9256 ms.
Despite this, PR=1.000 and DB=0.007 are both within spec. This is a known limitation;
adding tiscali to DQN training (with GNN-score caching enabled) would resolve it.

---

## Audit Summary

**Phase 4 clean audit:** `audit_gnn_lpd_dqn_clean_method.py` — **ALL 10 BLOCKS PASS**

| Block | Check | Result |
|---|---|---|
| 1 | Required output files present | PASS |
| 2 | Per-cycle CSV non-empty (3,976 rows, 8 topologies) | PASS |
| 3 | `gnn_used=lpd_used=dqn_used=1` in every row | PASS |
| 3 | `stage2=rf_gate=sticky=disturbance_fin=heuristic=0` in every row | PASS |
| 4 | `criticality_backend=gnn_lpd` in every row | PASS |
| 5 | `method_audit.json` flags correct | PASS |
| 6 | GNN checkpoint meta: DB-budgeted oracle provenance | PASS |
| 7 | Label provenance JSON: `db_budgeted_oracle_used=True` | PASS |
| 8 | Static grep: no forbidden tokens active in method script | PASS |
| 9 | Hardcoded-result check (3,976 rows, non-trivial) | PASS |
| 10 | FlexDATE comparison: abilene/cernet/geant/sprintlink all WIN BOTH | PASS |

---

## Limitations

1. **Tiscali DQN generalisation:** 81.5% full-OD fallback rate (4.3s mean latency) because DQN
   was not trained on tiscali. Fix: add tiscali to DQN training with `--lp_time_limit 5`.

2. **DQN training coverage:** DQN trained on abilene/cernet/geant/ebone. Sprintlink and tiscali
   excluded from training due to OD count (1892 and 2352 respectively) causing slow episodes
   even with FULL_OD_FALLBACK excluded from random exploration.

3. **GNN val_prec@K (0.7474):** Adding CERNET (1640 ODs, harder topology) slightly reduced
   val_prec@K vs. the 5-topology baseline (0.7553). Result still exceeds 0.70 threshold and
   all eval PR targets are met.

---

## Reproducibility

```bash
# Phase 0a: Build oracle labels (including CERNET)
python scripts/phase1_5/build_dbbudget_oracle_labels.py \
  --topologies abilene cernet geant sprintlink tiscali ebone

# Phase 0b: Train GNN
python scripts/phase1_5/train_gnn_dbbudget_selector.py

# Phase 1: Precompute pathopt references (all eval topologies)
python scripts/phase1_5/gnn_lpd_dqn_selective_db_lp.py --mode precompute

# Phase 2: Train DQN
python scripts/phase1_5/gnn_lpd_dqn_selective_db_lp.py --mode train \
  --gnn_checkpoint results/gnn_lpd_dqn_selective_db_lp/models/gnn_dbbudget_selector.pt \
  --train_topos abilene cernet geant ebone --val_topos abilene cernet geant ebone \
  --episodes 200 --lp_time_limit 5

# Phase 3: Evaluate (N=3976)
python scripts/phase1_5/gnn_lpd_dqn_selective_db_lp.py --mode eval \
  --gnn_checkpoint results/gnn_lpd_dqn_selective_db_lp/models/gnn_dbbudget_selector.pt \
  --dqn_checkpoint results/gnn_lpd_dqn_selective_db_lp/dqn_best.pt \
  --eval_topos abilene cernet geant sprintlink ebone tiscali germany50 vtlwavenet2011 \
  --tag final_N3976

# Phase 4: Audit
python scripts/phase1_5/audit_gnn_lpd_dqn_clean_method.py \
  --eval_dir results/gnn_lpd_dqn_selective_db_lp/final_N3976
```

---

*Generated from method's own evaluation artifacts. Not copied from legacy report.*
