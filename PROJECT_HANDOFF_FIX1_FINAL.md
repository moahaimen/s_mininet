# PROJECT HANDOFF — FIX1 STRICT-ALL RG-GNN-LPD

This is the current authoritative handoff for the Phase 1.5 project.

Supersedes the older root notes:
- `FULL_HANDOFF_PHASE15_CURRENT_STATE.md`
- `CONTINUATION_HANDOFF.md`

Those older handoffs describe earlier method lineages and are now stale.

## 1. Canonical workspace

- Actual repo root: `/Users/moahaimentalib/Desktop/f_flex_network_code_clean`
- User shell cwd at the time of this handoff may be different.
  Do not continue from an outer project folder. Use the repo above.
- Branch: `main`
- Latest committed HEAD when this handoff was written: `15ad7fb Update README for FIX1 strict-all public method`
- Remote: `https://github.com/moahaimen/f_flex_network_code.git`
- Python with project deps on this Mac:
  `/opt/homebrew/Caskroom/miniforge/base/bin/python`

## 2. Final method identity

Use one lineage only.

- Public method name: `Reward-Gated GNN-LPD Traffic Engineering (RG-GNN-LPD)`
- Final lineage: `FIX1 strict-all`
- Final normal protocol: `N=3976`
- Final VtlWavenet2011 normal stream: `200`
- PR numerator: `strict full-MCF optimum MLU / achieved MLU`
- Final deployed action space:
  - `KEEP`
  - `K50`
  - `K100`
  - `K200`
  - `K300`
  - `K500`
  - `K800`
- Final report narrative:
  - learned FlexDATE wins: `3/5`
  - FlexDATE DB wins: `5/5`
  - Sprintlink is not a learned win
  - Tiscali is not a learned win

Do not mix in the older source-scoped artifact with:
- `fresh K30`
- `fresh K40`
- `fresh K50`
- `sticky reuse`

That older artifact is not the current final report lineage.

## 3. Current primary files

### Report candidate currently being edited

- DOCX:
  `/Users/moahaimentalib/Desktop/f_flex_network_code_clean/results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/RG_GNN_LPD_FIX1_Final_Report_200VTL_N3976_FINAL_METHOD_AUDITED_PRREF_STRICTALL_MLU_UPGRADED.docx`
- PDF:
  `/Users/moahaimentalib/Desktop/f_flex_network_code_clean/results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/RG_GNN_LPD_FIX1_Final_Report_200VTL_N3976_FINAL_METHOD_AUDITED_PRREF_STRICTALL_MLU_UPGRADED.pdf`

### Completed metrics folder

`/Users/moahaimentalib/Desktop/f_flex_network_code_clean/results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/completed_metrics`

### Packaged artifact

- ZIP:
  `/Users/moahaimentalib/Desktop/f_flex_network_code_clean/results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/FINAL_REPORT_FIX1_completed_metrics_package.zip`
- Manifest:
  `/Users/moahaimentalib/Desktop/f_flex_network_code_clean/results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/MANIFEST_FINAL_REPORT_FIX1.md`

### Key scripts

- Report builder:
  `/Users/moahaimentalib/Desktop/f_flex_network_code_clean/scripts/phase1_5/build_fix1_completed_report.py`
- Completed metrics builder:
  `/Users/moahaimentalib/Desktop/f_flex_network_code_clean/scripts/phase1_5/run_fix1_completed_metrics.py`
- Targeted repair script:
  `/Users/moahaimentalib/Desktop/f_flex_network_code_clean/scripts/phase1_5/run_fix1_targeted_repairs.py`
- FIX1 model fine-tune script:
  `/Users/moahaimentalib/Desktop/f_flex_network_code_clean/scripts/phase1_5/run_fulldata_gated_fix1.py`
- Current SDN/Mininet script:
  `/Users/moahaimentalib/Desktop/f_flex_network_code_clean/scripts/phase1_5/run_sdn_mininet_clean.py`
- Existing repro runbook:
  `/Users/moahaimentalib/Desktop/f_flex_network_code_clean/FIX1_STRICTALL_REPRO_RUNBOOK.md`

## 4. What is already completed

These are already present as executed artifacts in `completed_metrics/`:

- Normal 8-topology FIX1 table:
  `normal_8topo_200vtl_consistent.csv`
- Pooled normal table:
  `normal_pooled_3976_consistent.csv`
- FlexDATE overlap table:
  `flexdate_comparison_prref.csv`
- Real scorer ablation:
  `scorer_ablation_real_200VTL_N3976.csv`
- Real scorer-ablation per-topology table:
  `scorer_ablation_real_200VTL_N3976_per_topology.csv`
- Policy ablation:
  `policy_ablation_fix1_COMPLETED.csv`
- K sensitivity:
  `k_sensitivity_fix1_COMPLETED.csv`
- Per-topology K sensitivity:
  `k_sensitivity_per_topology_fix1_200VTL_N3976_AUDITED.csv`
- Solver reliability:
  `solver_reliability_fix1_COMPLETED.csv`
- Routing stability:
  `routing_stability_fix1_COMPLETED.csv`
- First-TM ablation:
  `first_tm_ablation_fix1_COMPLETED.csv`
- MLU per topology:
  `mlu_per_topology_fix1_COMPLETED.csv`
- Failure fairness table:
  `failure_baseline_comparison_fix1_COMPLETED.csv`
- Failure scenario tables:
  `failure_comparison_*_fix1_COMPLETED.csv`
- Failure event validation:
  `failure_event_validation_fix1.md`
- PR reference audit:
  `pr_reference_audit_N3976.md`
- SDN retained table:
  `sdn_operational_metrics_COMPLETED.csv`

Also already rerun at `N=3976`:
- `DDQN without GNN-LPD`
- `DDQN without bottleneck relief`

So the old mixed-`N=3816` issue for those two rows is no longer the main blocker.

## 5. Current source-of-truth metrics

Use the CSVs below, not memory and not older report prose.

### Current pooled normal row

Source:
`completed_metrics/normal_pooled_3976_consistent.csv`

Current values:
- `N = 3976`
- `Mean PR = 98.519`
- `PR>=0.90 = 98.390`
- `PR>=0.95 = 88.657`
- `Mean DB = 0.450`
- `P95 DB = 1.596`
- `Mean ms = 84.581`
- `P95 ms = 383.750`
- `Min PR = 72.535`

### Current FlexDATE overlap row set

Source:
`completed_metrics/flexdate_comparison_prref.csv`

Current values:
- Abilene: `our_PR = 0.984792`, `our_DB = 0.005753`, win = `True`
- CERNET: `our_PR = 0.994820`, `our_DB = 0.000405`, win = `True`
- GEANT: `our_PR = 0.999440`, `our_DB = 0.002738`, win = `True`
- Sprintlink: `our_PR = 0.995663`, `our_DB = 0.003465`, win = `False`
- Tiscali: `our_PR = 0.952156`, `our_DB = 0.002307`, win = `False`

Interpretation:
- learned wins = `3/5`
- DB wins = `5/5`

### Current action distribution

Source:
`completed_metrics/action_distribution_200vtl_consistent.csv`

This confirms the executed action space in the final artifact is:
- `KEEP`
- `K50`
- `K100`
- `K200`
- `K300`
- `K500`
- `K800`

## 6. Critical unresolved issue: CSV vs report mismatch in Section 6

This is the main continuation problem.

The student is correct: the current report and the executed CSVs still differ in the baseline section.

### 6.1 OSPF mismatch

Current OSPF executed file:

`completed_metrics/ospf_weighted_shortest_path_baseline_N3976.csv`

This file contains two variants:
- `weight_mode = unit`
- `weight_mode = inverse_capacity`

The report Section 6 currently shows one row only:
- `OSPF-weighted shortest-path routing`

But the current code path is still binding that row to the `unit` variant, not the required `inverse_capacity` variant.

#### Executed OSPF summaries

`unit`:
- `N = 3976`
- `Mean PR = 55.419`
- `PR>=0.90 = 8.174`
- `PR>=0.95 = 0.780`
- `Mean DB = 0.000`
- `P95 DB = 0.000`
- `Mean ms = 4.554`
- `P95 ms = 32.879`
- `Min PR = 25.111`

`inverse_capacity`:
- `N = 3976`
- `Mean PR = 73.635`
- `PR>=0.90 = 17.153`
- `PR>=0.95 = 2.238`
- `Mean DB = 0.000`
- `P95 DB = 0.000`
- `Mean ms = 3.469`
- `P95 ms = 32.383`
- `Min PR = 25.111`

Required interpretation:
- the report must choose one OSPF mode
- the requested/meaningful mode is `inverse_capacity`
- therefore the Section 6 OSPF row is currently wrong

### 6.2 ECMP mismatch

Executed ECMP file:

`completed_metrics/ecmp_per_cycle.csv`

Executed ECMP summary from that file:
- `N = 3976`
- `Mean PR = 53.269`
- `PR>=0.90 = 8.576`
- `PR>=0.95 = 1.107`
- `Mean DB = 0.000`
- `P95 DB = 0.000`
- `Mean ms = 0.000`
- `P95 ms = 0.000`
- `Min PR = 25.113`

But the current `completed_metrics/baseline_comparison_fix1_COMPLETED.csv` still contains the stale ECMP row:
- `Mean PR = 50.693`
- `PR>=0.90 = 6.539`
- `PR>=0.95 = 0.553`

So the current report Section 6 ECMP row is also wrong.

### 6.3 Exact code-backed cause

There are two code problems.

1. `run_fix1_completed_metrics.py` still takes the wrong OSPF subset:

- file:
  `/Users/moahaimentalib/Desktop/f_flex_network_code_clean/scripts/phase1_5/run_fix1_completed_metrics.py`
- lines:
  `953-955`

It does:
- read `ospf_weighted_shortest_path_baseline_N3976.csv`
- filter only `method == "OSPF-weighted shortest-path routing"`

That keeps the `unit` row name, not the `inverse_capacity` variant the report should use.

2. `run_fix1_targeted_repairs.py` refreshes DDQN rows but leaves stale ECMP/OSPF summary rows in the baseline table:

- file:
  `/Users/moahaimentalib/Desktop/f_flex_network_code_clean/scripts/phase1_5/run_fix1_targeted_repairs.py`
- baseline repair function:
  `191-214`
- OSPF filter:
  `371-372`

`build_policy_and_baseline()` only replaces:
- `DDQN without GNN-LPD`
- `DDQN without bottleneck relief`
- `Final RG-GNN-LPD`

It does not regenerate or replace the ECMP row.
It also keeps the wrong OSPF subset.

### 6.4 Required next fix

Next required fix:

1. Patch OSPF selection so Section 6 uses only:
- `weight_mode == "inverse_capacity"`

2. Regenerate/rebuild the ECMP baseline row directly from:
- `completed_metrics/ecmp_per_cycle.csv`

3. Rebuild:
- `completed_metrics/baseline_comparison_fix1_COMPLETED.csv`

4. Rebuild the report DOCX/PDF.

### 6.5 Minimal files to patch

- `/Users/moahaimentalib/Desktop/f_flex_network_code_clean/scripts/phase1_5/run_fix1_completed_metrics.py`
- `/Users/moahaimentalib/Desktop/f_flex_network_code_clean/scripts/phase1_5/run_fix1_targeted_repairs.py`
- then rerun `/Users/moahaimentalib/Desktop/f_flex_network_code_clean/scripts/phase1_5/build_fix1_completed_report.py`

## 7. Current SDN/Mininet status

The report currently does **not** contain fresh live FIX1 Mininet numbers.

What it contains now:
- retained historical SDN/Mininet QoS evidence only
- generated into:
  `completed_metrics/sdn_operational_metrics_COMPLETED.csv`

Two important facts:

1. The current Phase 1.5 SDN script is not truly live yet.

File:
`/Users/moahaimentalib/Desktop/f_flex_network_code_clean/scripts/phase1_5/run_sdn_mininet_clean.py`

Problems:
- lines `75-89`: it imports the older `gnn_lpd_dqn_selective_db_lp` action space/checkpoint family and sets:
  - `METHOD = "gnn_lpd_dqn_selective_db_lp"`
  - `DQN_CKPT = results/gnn_lpd_dqn_selective_db_lp/dqn_best.pt`
- lines `731-747`: `run_mininet()` is not wired and falls back to `run_simulate(args)`

So this script is not yet a real live FIX1 strict-all runner.

2. The completed-metrics builder hardcodes the SDN report row as simulated historical evidence.

File:
`/Users/moahaimentalib/Desktop/f_flex_network_code_clean/scripts/phase1_5/run_fix1_completed_metrics.py`

Lines `1169-1188` currently hardcode:
- `Mode = "simulate"`
- `Rerun on FIX1 yes/no = "no"`

So even if fresh live SDN numbers are generated, the report refresh path must be patched or it will still label them as retained simulation.

## 8. Windows/Mininet continuation plan

There is a separate Windows handoff for this:

`/Users/moahaimentalib/Desktop/f_flex_network_code_clean/WINDOWS_MININET_FIX1_LIVE_HANDOFF.md`

Use that file for the live SDN task.

## 9. Safe next-step sequence

Do these in order.

1. Fix the Section 6 baseline mismatch:
   - OSPF = `inverse_capacity` only
   - ECMP = current `ecmp_per_cycle.csv`

2. Rebuild the baseline CSV and report:

```bash
cd /Users/moahaimentalib/Desktop/f_flex_network_code_clean
/opt/homebrew/Caskroom/miniforge/base/bin/python scripts/phase1_5/run_fix1_targeted_repairs.py
/opt/homebrew/Caskroom/miniforge/base/bin/python scripts/phase1_5/build_fix1_completed_report.py
soffice --headless --convert-to pdf \
  --outdir results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1 \
  results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/RG_GNN_LPD_FIX1_Final_Report_200VTL_N3976_FINAL_METHOD_AUDITED_PRREF_STRICTALL_MLU_UPGRADED.docx
```

3. Validate Section 6 after rebuild against:
- `completed_metrics/ecmp_per_cycle.csv`
- `completed_metrics/ospf_weighted_shortest_path_baseline_N3976.csv`

4. Only after that, if the user wants fresh SDN live values, continue with the Windows/Mininet handoff.

## 10. Hard guardrails

- Do not work in the non-repo cwd.
- Do not mix old K30/K40/K50/sticky artifact numbers into FIX1 strict-all.
- Do not call the retained SDN table a fresh live FIX1 rerun.
- Do not assume the current report DOCX is authoritative if a CSV disagrees; the executed CSV is the source of truth.
- Do not use the older `sdn/` Ryu/Mininet code as final FIX1 evidence without explicitly porting it to the current FIX1 strict-all controller/checkpoints.
- Do not use GitHub `main` alone as the full truth for this report state; there are local uncommitted report/metrics changes in this checkout.
