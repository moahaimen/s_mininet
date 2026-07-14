# s_mininet

Focused FIX1 Mininet/OVS validation snapshot for the RG-GNN-LPD controller and its final live SDN rerun artifacts.

## What is in this repo

- `scripts/phase1_5/`: the live SDN/Mininet runner and related FIX1 build scripts
- `te/`: traffic-engineering solvers and path utilities used by the runner
- `configs/`: topology and phase config files
- `data/`: small bundled topology assets used by the live harness
- `sdn/`: SDN helper modules
- `artifacts/final_hardfix_handoff/`: final CSVs, raw logs, and validation notes from the hardfix handoff bundle

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

This repo is a curated Mininet-focused publish from the larger FIX1 snapshot. It is not the entire original workspace dump.
