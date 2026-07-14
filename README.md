# s_mininet

S23 RG-GNN-LPD project code plus the FIX1 Mininet/OVS validation harness for
fresh live SDN reruns.

## What is in this repo

- S23 project folders copied from `moahaimen/s23_network`: `phase1_reactive/`,
  `phase2/`, `phase3/`, `rl/`, `eval/`, `tests/`, `configs/`, `data/`,
  `results/`, `scripts/`, `sdn/`, and `te/`
- `scripts/phase1_5/`: the live SDN/Mininet runner and related FIX1 build scripts
- `te/`: traffic-engineering solvers and path utilities used by the runner
- `sdn/`: SDN helper modules and controller/testbed adapters
- `artifacts/final_hardfix_handoff/`: final CSVs, raw logs, and validation notes from the hardfix handoff bundle
- `STUDENT_MININET_RUN_GUIDE.md`: the short Linux/Mininet command handoff for a student rerun

## Original code vs Mininet code

They are connected, but not the same thing.

- The original RG-GNN-LPD routing/controller logic lives mainly in `scripts/phase1_5/gnn_lpd_dqn_selective_db_lp.py` and the `te/` solvers.
- The live Mininet validation layer lives mainly in `scripts/phase1_5/run_sdn_mininet_clean.py` and `scripts/phase1_5/mininet_vm_bridge.py`.
- The Mininet runner imports the routing logic and applies it against live OVS/Mininet scenarios.

So the Mininet code is a live execution and measurement harness around the original routing/controller code, not a separate algorithm rewrite.

## Main hardfix changes

See `CHANGES_FROM_ORIGINAL.md` for the full list. The key updates were:

1. Correct transient-loss collection from VM-side iperf logs instead of shelling through Mininet host paths.
2. Incremental OpenFlow updates that modify/add only changed rules instead of overusing table replacement.
3. Backup-support rule preinstallation for Abilene single-link failure handling.
4. Split controller timing into decision/pipeline/legacy timing instead of one blended number.
5. Final hardfix artifact builders for the Abilene and GEANT live rerun outputs.

## Important note

This student branch is intended to preserve the S23 project structure while
adding the Mininet rerun harness and evidence files needed for live SDN
validation.
