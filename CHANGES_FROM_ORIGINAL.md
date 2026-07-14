# Changes From Original FIX1 Mininet Runner

This summarizes the practical code changes made to the live SDN/Mininet path relative to the earlier/original runner behavior.

## 1. Measurement correctness fix

- `collect_transient_udp_phase(...)` was changed to read transient iperf logs directly from VM-side `/tmp` log files.
- This replaced the earlier host-shell log collection path that could misread or miss transient output.
- Result: false `100%` transient loss readings were eliminated when the VM had valid iperf logs.

## 2. Incremental flow-update behavior

- Added `INCREMENTAL_REPLACE_THRESHOLD = 999999`.
- Added `_ovs_mod_flows_batch(...)`.
- `_apply_switch_flow_update(...)` now prefers modifying changed rules and adding new ones in batches instead of needlessly doing small-table replacements.
- Result: less disruptive rule updates and cleaner recovery timing.

## 3. Backup support preinstall for Abilene failure handling

- Added `preinstall_backup_support_flows(...)`.
- This preinstalls downstream support rules for the Abilene single-link failure case before the failure cut.
- Result: affected flows recover more cleanly without flushing unaffected traffic.

## 4. Controller timing split

- `compute_state_vector(...)` now emits timing such as `state_snapshot_ms` and `scorer_ranking_ms`.
- `run_fix1_cycle(...)` now records `ddqn_inference_ms`.
- Exported records now separate:
  - `controller_decision_ms`
  - `controller_pipeline_ms`
  - `legacy_controller_response_ms`
  - `rule_diff_build_ms`
  - `flow_mod_send_ms`
  - `barrier_wait_ms`
  - `probe_wait_ms`
  - `logging_ms`
- Result: startup-independent controller timing is more defensible and easier to audit.

## 5. Export and artifact changes

- Per-run and summary CSV exports were expanded to carry the new timing fields.
- Final packaging scripts were added/updated:
  - `scripts/phase1_5/build_final_hardfix_sdn_artifacts.py`
  - `scripts/phase1_5/build_abigeant_final_sdn.py`
  - `scripts/phase1_5/build_final_microaudit_sdn.py`
- Result: the final Abilene/GEANT live rerun outputs can be regenerated and audited consistently.

## Answer to the repo question

No, the original controller/routing code is not identical to the Mininet code.

- The original code is the routing/controller logic.
- The Mininet code is the live SDN harness that calls that logic, installs rules in OVS, injects failures, and measures QoS.
- The hardfix work changed the Mininet harness and artifact path much more than the core routing model itself.
