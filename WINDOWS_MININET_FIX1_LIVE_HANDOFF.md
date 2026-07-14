# WINDOWS MININET LIVE HANDOFF — FIX1 STRICT-ALL RG-GNN-LPD

This handoff is for a Windows laptop with access to a Linux VM or WSL2 environment that can actually run Mininet/OVS.

Goal:
- run a real live SDN/Mininet validation for the **current** final method
- write fresh SDN CSVs
- feed those fresh live numbers back into the FIX1 report

Do not use this handoff for the older K30/K40/K50/sticky artifact.

## 1. Canonical method to validate

Validate only this lineage:

- Public name:
  `Reward-Gated GNN-LPD Traffic Engineering (RG-GNN-LPD)`
- Final lineage:
  `FIX1 strict-all`
- Action space:
  - `KEEP`
  - `K50`
  - `K100`
  - `K200`
  - `K300`
  - `K500`
  - `K800`
- PR numerator:
  `strict full-MCF optimum MLU / achieved MLU`

Current final FIX1 artifact roots:

- Final refined controller folder:
  `/Users/moahaimentalib/Desktop/f_flex_network_code_clean/results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FULLDATA_GATED_PRESERVED_FIX1`
- Final refined checkpoint:
  `/Users/moahaimentalib/Desktop/f_flex_network_code_clean/results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FULLDATA_GATED_PRESERVED_FIX1/fulldata_gated_model.pt`
- Frozen teacher checkpoint:
  `/Users/moahaimentalib/Desktop/f_flex_network_code_clean/results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FROZEN_FINAL_LEARNED_RUNTIME_SAFE_ITER2/final_learned_4of5_iter2_model.pt`

## 2. What is wrong right now

The current SDN script is not yet suitable for fresh FIX1 live numbers.

### 2.1 Wrong controller lineage

File:
`scripts/phase1_5/run_sdn_mininet_clean.py`

At lines `75-89`, it imports the older:
- `scripts/phase1_5/gnn_lpd_dqn_selective_db_lp.py`
- `METHOD = "gnn_lpd_dqn_selective_db_lp"`
- `DQN_CKPT = results/gnn_lpd_dqn_selective_db_lp/dqn_best.pt`

That is not the current FIX1 strict-all controller/checkpoint.

### 2.2 Mininet mode is not implemented

Same file, lines `731-747`:
- `run_mininet()` prints “Mininet mode: not yet wired”
- then falls back to `run_simulate(args)`

So `--mode mininet` does not currently produce true live results.

### 2.3 Report refresh path is hardcoded as retained simulation

File:
`scripts/phase1_5/run_fix1_completed_metrics.py`

Lines `1169-1188` currently hardcode the SDN table as:
- `Mode = "simulate"`
- `Rerun on FIX1 yes/no = "no"`

So even if live data are generated, the report builder will still label them as retained simulation unless this is patched.

## 3. Scope to preserve

The existing report SDN section currently covers:
- topologies:
  - `abilene`
  - `geant`
- scenarios:
  - `normal`
  - `single_link_failure`
  - `two_link_failure`
  - `capacity_degradation_50`
  - `spike_x3`
  - `mixed_spike_failure`
- runs per scenario:
  `10`

If the goal is to replace the current report table fairly and with minimal change, keep this same SDN scope.

## 4. Hard rules

- Do not reuse the retained historical SDN CSV as if it were a fresh live rerun.
- Do not use the older source-scoped K30/K40/K50/sticky method.
- Do not use the older `dqn_best.pt` as the final controller.
- Do not label any row `Mode=live` or `Rerun on FIX1=yes` unless the run was truly live.
- Do not use the older `sdn/` directory output as final evidence unless you explicitly port it to the current FIX1 controller and verify it.

## 5. Recommended environment

Use Linux, not native Windows:

- Ubuntu VM, WSL2 with systemd/networking support, or a Linux laptop
- Mininet installed
- Open vSwitch installed
- Ryu installed if needed

Suggested setup:

```bash
sudo apt update
sudo apt install -y mininet openvswitch-switch python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install ryu
```

Quick checks:

```bash
mn --test pingall
ovs-vsctl show
python3 -c "import torch, pandas, pulp; print('python deps OK')"
```

## 6. What to implement before running

### Step A — wire the live runner to the current FIX1 controller

Modify:
`scripts/phase1_5/run_sdn_mininet_clean.py`

Required changes:

1. Stop loading the old `dqn_best.pt` path.
2. Load the current FIX1 strict-all checkpoint:
   - `FULLDATA_GATED_PRESERVED_FIX1/fulldata_gated_model.pt`
3. If TM0 behavior is meant to match the final report exactly, also load the frozen teacher:
   - `FROZEN_FINAL_LEARNED_RUNTIME_SAFE_ITER2/final_learned_4of5_iter2_model.pt`
4. Match the final action space:
   - `KEEP`, `K50`, `K100`, `K200`, `K300`, `K500`, `K800`
5. Match the final routing semantics:
   - teacher action only at TM0 if applicable
   - final network from TM1 onward
   - selected-flow LP only
   - no RandomForest
   - no Stage-2
   - no disturbance-finalization LP

### Step B — implement real `run_mininet()`

Current `run_mininet()` is a stub.

Implement it so it:
- creates or attaches to a real Mininet topology
- measures actual:
  - throughput
  - RTT
  - jitter
  - packet loss
  - install time
  - recovery time
- records controller/runtime metrics for the current FIX1 controller

### Step C — write live outputs to the same expected folder

Keep the output folder:

`results/gnn_lpd_dqn_selective_db_lp/sdn_mininet_clean/`

Required files:
- `sdn_per_run.csv`
- `sdn_summary.csv`
- `sdn_method_audit.json`

Each per-run row should preserve the current audit fields and also make the live mode explicit.

Minimum per-run expectations:
- `topology`
- `scenario`
- `run_id`
- `method`
- `mode` = `live`
- `throughput_mbps`
- `packet_loss_pct`
- `rtt_ms`
- `jitter_ms`
- `recovery_ms`
- `flow_rule_count`
- `install_ms`
- `controller_status`
- `decision_ms`
- `pr`
- `mlu`
- `db`
- `action_name`
- `active_od_count`
- `selected_od_count`
- audit flags

## 7. Do not use the old `sdn/` harness as final evidence without porting

These files exist:
- `sdn/mininet_testbed.py`
- `sdn/ryu_te_app.py`
- `sdn/run_sdn_simulation.py`

They are useful as reference scaffolding for:
- Mininet topology creation
- Ryu/OpenFlow plumbing
- traffic injection structure

But they are not automatically the current final FIX1 method.

Reasons:
- they belong to an older Phase-1 SDN pipeline
- they do not point at `FULLDATA_GATED_PRESERVED_FIX1/fulldata_gated_model.pt`
- they are not the report’s current strict-all controller lineage

Use them only if you explicitly port the current FIX1 controller into them.

## 8. After live run: patch the report-refresh path

After `sdn_summary.csv` and `sdn_per_run.csv` contain real live values, patch:

`scripts/phase1_5/run_fix1_completed_metrics.py`

Current lines `1169-1188` hardcode:
- `Mode = "simulate"`
- `Rerun on FIX1 yes/no = "no"`

Required behavior after live rerun:
- `Mode = "live"` for the new live rows
- `Rerun on FIX1 yes/no = "yes"`

Best approach:
- derive `Mode` from `sdn_per_run.csv`
- derive rerun flag from the actual run mode instead of hardcoding it

Then regenerate:
- `completed_metrics/sdn_operational_metrics_COMPLETED.csv`

## 9. Rebuild report after fresh live SDN numbers

From the repo root:

```bash
python3 scripts/phase1_5/run_fix1_completed_metrics.py
python3 scripts/phase1_5/build_fix1_completed_report.py
soffice --headless --convert-to pdf \
  --outdir results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1 \
  results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/RG_GNN_LPD_FIX1_Final_Report_200VTL_N3976_FINAL_METHOD_AUDITED_PRREF_STRICTALL_MLU_UPGRADED.docx
```

Then confirm in the rebuilt report:
- Section 13 rows now show `Mode=live`
- `Rerun on FIX1 yes/no = yes`
- the retained historical wording is updated so it no longer falsely claims the table is only retained simulation

## 10. Success criteria

This task is complete only if all of the following are true:

1. `run_sdn_mininet_clean.py --mode mininet` performs a real live run and does not fall back to simulate mode.
2. The live run uses the current FIX1 strict-all controller/checkpoints, not the old `dqn_best.pt`.
3. `sdn_per_run.csv` and `sdn_summary.csv` are regenerated from live execution.
4. `completed_metrics/sdn_operational_metrics_COMPLETED.csv` reflects those live numbers.
5. The final report’s SDN table shows the correct live values and correct labels.
6. The report no longer depends on retained historical SDN numbers for the rows you replaced.

## 11. One-line continuation summary

The current Mac report is still using retained historical SDN evidence because the Phase 1.5 Mininet script is only a simulation/live stub and still points at the older `dqn_best.pt` controller family. The Windows/Linux continuation task is to wire real Mininet mode to the current FIX1 strict-all controller (`fulldata_gated_model.pt`), generate fresh live `sdn_per_run.csv` and `sdn_summary.csv`, propagate `Mode=live` and `Rerun on FIX1=yes` into `sdn_operational_metrics_COMPLETED.csv`, and then rebuild the report.
