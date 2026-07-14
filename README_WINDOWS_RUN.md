# Windows Run Guide — Clean GNN-LPD-DQN Traffic Engineering

This guide lets you verify and reproduce the clean GNN-LPD-DQN traffic-engineering
results from a **fresh clone** on Windows (PowerShell or Command Prompt). The same
commands work on macOS/Linux by using the bash activate path noted below.

All paths in this repository are project-relative. No external download is required:
every model checkpoint, label file, and evidence artifact is committed directly
(largest file is ~20 MB, well under GitHub limits — Git LFS is **not** needed).

---

## 1. Clone

```bat
git clone https://github.com/moahaimen/final_trafic.git
cd final_trafic
```

## 2. Create and activate a virtual environment

Windows (PowerShell / CMD):

```bat
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Requires **Python 3.9 or newer**.

## 4. Run the smoke test (do this first)

```bat
python scripts\phase1_5\windows_smoke_test.py
```

Expected final line:

```
READY FOR STUDENT RUN
```

The smoke test checks the Python version, dependencies, source modules, evidence
artifacts, CSV row counts (3976 / 360 / 120), core imports, and that the GNN-LPD
and DQN checkpoints load.

## 5. Run the clean-method compliance audit

```bat
python scripts\phase1_5\audit_gnn_lpd_dqn_clean_method.py --eval_dir results\gnn_lpd_dqn_selective_db_lp\final_N3976
```

Expected:

* `CLEAN METHOD AUDIT PASSED`
* per-cycle rows = 3976
* no heuristic, no RandomForest gate, no sticky gate, no Stage-2, no disturbance finalization
* `criticality_backend = gnn_lpd` in every cycle

---

## Optional — regenerate evidence from the committed checkpoints

These steps re-run the analyses; they are **not** required to verify the committed
results, which are already present under `results/gnn_lpd_dqn_selective_db_lp/`.

Failure-scenario validation (Abilene + GEANT, 9 scenarios × 20 cycles = 360 rows):

```bat
python scripts\phase1_5\run_failure_validation_clean.py
```

LP-derived SDN-style operational simulation (6 scenarios × 10 runs × 2 topologies = 120 rows):

```bat
python scripts\phase1_5\run_sdn_mininet_clean.py --mode simulate
```

> The SDN section is an **LP-derived simulation**, not a live Mininet emulation.
> To produce real packet-level Mininet measurements, run the same script with
> `--mode mininet` on a Linux host with Open vSwitch and Mininet installed
> (`sudo apt install mininet`). The controller uses the same committed
> GNN-LPD and DQN checkpoints.

---

## Evidence artifacts (already committed)

| Artifact | Path |
|---|---|
| Full evaluation (N=3976) | `results/gnn_lpd_dqn_selective_db_lp/final_N3976/` |
| Failure validation (360 cycles) | `results/gnn_lpd_dqn_selective_db_lp/failure_validation_clean/` |
| LP-derived SDN simulation (120 runs) | `results/gnn_lpd_dqn_selective_db_lp/sdn_mininet_clean/` |
| GNN-LPD selector checkpoint | `results/gnn_lpd_dqn_selective_db_lp/models/gnn_dbbudget_selector.pt` |
| DQN checkpoint | `results/gnn_lpd_dqn_selective_db_lp/dqn_best.pt` |
| Oracle labels | `results/gnn_lpd_dqn_selective_db_lp/labels/oracle_labels.csv` |
| Final report (Word / PDF) | `STUDENT_PROFESSOR_FINAL_REPORT.docx` / `.pdf` |

See `REPRODUCIBILITY.md` for the full end-to-end pipeline and `FINAL_REPORT.md`
for the method summary.

---

## Exact first command to run

```bat
python -m venv .venv && .venv\Scripts\activate && python -m pip install --upgrade pip && pip install -r requirements.txt && python scripts\phase1_5\windows_smoke_test.py
```
