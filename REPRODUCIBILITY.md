# Reproducibility — Clean GNN-LPD-DQN Traffic Engineering

This document records the full pipeline that produced the committed evidence
artifacts. All commands are run from the repository root with the virtual
environment active (see `README_WINDOWS_RUN.md`). All paths are project-relative.

## Method

```
Traffic matrix + topology
  -> DB-budgeted LP-distilled GNN-LPD critical-OD selector
  -> DQN controller chooses K / action / DB budget
  -> selected critical ODs -> one-stage selected-flow DB-budgeted LP
  -> noncritical ODs remain on ECMP or previous routing
  -> capped K escalation if PR/MLU guard fails (K30 -> K40 -> K50)
  -> full-OD fallback only after the selected-K cap fails
```

Excluded by design (verified by the compliance audit): heuristic criticality,
RandomForest gate, sticky-gate reuse, Stage-2 DB LP, disturbance-finalization LP.

## Environment

* Python 3.9+
* Dependencies pinned in `requirements.txt`
* Solver: PuLP with HiGHS preferred, CBC fallback

## Pipeline

| Step | Command | Output |
|---|---|---|
| 1. Oracle labels | `python scripts/phase1_5/build_dbbudget_oracle_labels.py` | `results/gnn_lpd_dqn_selective_db_lp/labels/oracle_labels.csv` (279,360 rows) |
| 2. Train GNN-LPD selector | `python scripts/phase1_5/train_gnn_dbbudget_selector.py` | `results/gnn_lpd_dqn_selective_db_lp/models/gnn_dbbudget_selector.pt` |
| 3. Precompute path-optimal MLU | `python scripts/phase1_5/gnn_lpd_dqn_selective_db_lp.py --mode precompute` | `results/gnn_lpd_dqn_selective_db_lp/pathopt_ref/*.csv` |
| 4. Train DQN controller | `python scripts/phase1_5/gnn_lpd_dqn_selective_db_lp.py --mode train` | `results/gnn_lpd_dqn_selective_db_lp/dqn_best.pt` |
| 5. Full evaluation | `python scripts/phase1_5/gnn_lpd_dqn_selective_db_lp.py --mode eval` | `results/gnn_lpd_dqn_selective_db_lp/final_N3976/per_cycle.csv` (3976 rows) |
| 6. Compliance audit | `python scripts/phase1_5/audit_gnn_lpd_dqn_clean_method.py --eval_dir results/gnn_lpd_dqn_selective_db_lp/final_N3976` | audit PASS |
| 7. Failure validation | `python scripts/phase1_5/run_failure_validation_clean.py` | `results/gnn_lpd_dqn_selective_db_lp/failure_validation_clean/` (360 cycles) |
| 8. LP-derived SDN simulation | `python scripts/phase1_5/run_sdn_mininet_clean.py --mode simulate` | `results/gnn_lpd_dqn_selective_db_lp/sdn_mininet_clean/` (120 runs) |

## Datasets

| Topology | Role | Eval TMs |
|---|---|---|
| Abilene | seen training | 2,016 |
| GEANT | seen training | 672 |
| CERNET | seen training | 200 |
| Sprintlink | seen training | 200 |
| Tiscali | seen training | 200 |
| Ebone | seen training | 200 |
| Germany50 | zero-shot eval | 288 |
| VtlWavenet2011 | zero-shot eval | 200 |

* Main comparison subset: N = 3,288 (Abilene + CERNET + GEANT + Sprintlink + Tiscali)
* Full internal evaluation: N = 3,976 (all 8 topologies)

## Evidence row counts (verifiable)

| File | Rows |
|---|---|
| `final_N3976/per_cycle.csv` | 3,976 |
| `failure_validation_clean/failure_per_cycle.csv` | 360 (2 topologies × 9 scenarios × 20 cycles) |
| `sdn_mininet_clean/sdn_per_run.csv` | 120 (2 topologies × 6 scenarios × 10 runs) |

Re-verify counts and audit flags at any time:

```bash
python scripts/phase1_5/windows_smoke_test.py
```

## Audit provenance

* `results/gnn_lpd_dqn_selective_db_lp/final_N3976/method_audit.json` — full-eval flags
* `results/gnn_lpd_dqn_selective_db_lp/failure_validation_clean/failure_method_audit.json` — `"audit_result": "PASS"`
* `results/gnn_lpd_dqn_selective_db_lp/sdn_mininet_clean/sdn_method_audit.json` — `"audit_result": "PASS"`, `"mode": "simulate"`
* `results/gnn_lpd_dqn_selective_db_lp/labels/label_provenance.json` — oracle label definition
* `results/gnn_lpd_dqn_selective_db_lp/models/gnn_dbbudget_selector_meta.json` — GNN architecture/training metadata

In every evaluation, failure, and SDN row:
`gnn_used = lpd_used = dqn_used = 1` and
`heuristic_used = random_forest_gate_used = sticky_gate_used = stage2_used = disturbance_finalization_used = 0`.

## Scope of claims

* The four topologies with a source-locked FlexDATE reference (Abilene, CERNET,
  GEANT, Sprintlink) win both PR and DB.
* Tiscali is included in N = 3,288 and in the CDFs but is **not** claimed as a
  FlexDATE win/loss, because no source-locked FlexDATE reference row exists for it.
* The SDN section is an LP-derived operational simulation, not a live Mininet
  emulation; a `--mode mininet` entry point is provided for hardware validation.
