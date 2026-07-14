# FIX1 Strict-All Reproduction Runbook

This runbook is for the current final report lineage:

- Method: `Reward-Gated GNN-LPD Traffic Engineering (RG-GNN-LPD)`
- Final report: `results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/RG_GNN_LPD_FIX1_Final_Report_200VTL_N3976_FINAL_METHOD_AUDITED_PRREF_STRICTALL.docx`
- Normal protocol: `N=3976`
- VtlWavenet2011 normal stream: `200`
- PR numerator: strict full-MCF optimum MLU

Use the exact repo commit that contains this file.

## 1. Required bundled artifacts

The following paths must exist before running the report pipeline:

- `results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FULLDATA_GATED_PRESERVED_FIX1/`
- `results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FROZEN_FINAL_LEARNED_RUNTIME_SAFE_ITER2/`
- `results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/STRICT_FULL_MCF_PR/_partial/`
- `results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/completed_metrics/`
- `results/gnn_lpd_dqn_selective_db_lp/pathopt_ref/`
- `results/gnn_lpd_dqn_selective_db_lp/sdn_mininet_clean/`
- `results/phase1_5_incremental/lp_distilled_pr_gnn_corrected_split/models/hgb_regressor_log_score.pkl`
- `results/phase1_5_incremental/lp_distilled_pr_gnn_corrected_split/models/hgb_specialist_path_lp.pkl`

Quick presence check:

```bash
python3 - <<'PY'
from pathlib import Path
root = Path('.').resolve()
required = [
    "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FULLDATA_GATED_PRESERVED_FIX1",
    "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FROZEN_FINAL_LEARNED_RUNTIME_SAFE_ITER2",
    "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/STRICT_FULL_MCF_PR/_partial",
    "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/completed_metrics",
    "results/gnn_lpd_dqn_selective_db_lp/pathopt_ref",
    "results/gnn_lpd_dqn_selective_db_lp/sdn_mininet_clean",
    "results/phase1_5_incremental/lp_distilled_pr_gnn_corrected_split/models/hgb_regressor_log_score.pkl",
    "results/phase1_5_incremental/lp_distilled_pr_gnn_corrected_split/models/hgb_specialist_path_lp.pkl",
]
missing = [p for p in required if not (root / p).exists()]
if missing:
    print("MISSING:")
    for p in missing:
        print(" -", p)
    raise SystemExit(1)
print("All required artifacts are present.")
PY
```

## 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install python-docx pypdf pdf2image
```

For PDF export, install LibreOffice and make sure `soffice` is on `PATH`.

## 3. Fast path: rebuild the final report from the completed metrics package

This is the fastest way to reproduce the exact current report outputs.

```bash
python3 scripts/phase1_5/audit_fix1_k_sensitivity.py
python3 scripts/phase1_5/audit_pr_reference.py
python3 scripts/phase1_5/build_fix1_completed_report.py
soffice --headless --convert-to pdf \
  --outdir results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1 \
  results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/RG_GNN_LPD_FIX1_Final_Report_200VTL_N3976_FINAL_METHOD_AUDITED_PRREF_STRICTALL.docx
```

Expected outputs:

- `results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/RG_GNN_LPD_FIX1_Final_Report_200VTL_N3976_FINAL_METHOD_AUDITED_PRREF_STRICTALL.docx`
- `results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/RG_GNN_LPD_FIX1_Final_Report_200VTL_N3976_FINAL_METHOD_AUDITED_PRREF_STRICTALL.pdf`

## 4. Slower path: regenerate the completed metrics CSV package from the bundled FIX1 artifacts

This uses the included frozen checkpoints, strict caches, path references, and completed artifact folders. It does not require retraining from scratch.

```bash
python3 scripts/phase1_5/run_fix1_completed_metrics.py
python3 scripts/phase1_5/run_fix1_failure_completed_metrics.py
python3 scripts/phase1_5/audit_fix1_k_sensitivity.py
python3 scripts/phase1_5/audit_pr_reference.py
python3 scripts/phase1_5/build_fix1_completed_report.py
soffice --headless --convert-to pdf \
  --outdir results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1 \
  results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/RG_GNN_LPD_FIX1_Final_Report_200VTL_N3976_FINAL_METHOD_AUDITED_PRREF_STRICTALL.docx
```

Main regenerated CSVs:

- `completed_metrics/normal_8topo_200vtl_consistent.csv`
- `completed_metrics/normal_pooled_3976_consistent.csv`
- `completed_metrics/flexdate_comparison_prref.csv`
- `completed_metrics/scorer_ablation_real_200VTL_N3976.csv`
- `completed_metrics/policy_ablation_fix1_COMPLETED.csv`
- `completed_metrics/k_sensitivity_fix1_COMPLETED.csv`
- `completed_metrics/k_sensitivity_per_topology_fix1_200VTL_N3976_AUDITED.csv`
- `completed_metrics/failure_scenarios_fix1_COMPLETED.csv`

## 5. Exact verification checks

### 5.1 Pooled normal row

```bash
python3 - <<'PY'
import pandas as pd
p = "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/completed_metrics/normal_pooled_3976_consistent.csv"
df = pd.read_csv(p).iloc[0]
print(df.to_dict())
assert int(df["N"]) == 3976
assert round(float(df["Mean PR"]), 3) == 98.627
assert round(float(df["Mean DB"]), 3) == 0.456
PY
```

### 5.2 Section 8 adaptive per-topology values must match Section 4

```bash
python3 - <<'PY'
import pandas as pd
base = "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/completed_metrics"
k = pd.read_csv(f"{base}/k_sensitivity_per_topology_fix1_200VTL_N3976_AUDITED.csv")
n = pd.read_csv(f"{base}/normal_8topo_200vtl_consistent.csv")
merged = k.merge(n[["topology", "fulldata_meanPR", "meanDB"]], left_on="Topology", right_on="topology", how="left")
for row in merged.itertuples():
    expected = f"{row.fulldata_meanPR*100:.3f}% / {row.meanDB*100:.3f}%"
    actual = getattr(row, "_17")
    print(row.Topology, actual, expected)
    assert actual == expected
print("Adaptive DDQN per-topology values match Section 4.")
PY
```

### 5.3 Strict PR reference audit

```bash
python3 scripts/phase1_5/audit_pr_reference.py
cat results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/completed_metrics/pr_reference_audit_N3976.md
```

## 6. Current final report identity

The final reported lineage is the FIX1 strict-all artifact, not the earlier fresh/sticky source-scoped artifact.

- Action space: `KEEP`, `K50`, `K100`, `K200`, `K300`, `K500`, `K800`
- Final method report file: `RG_GNN_LPD_FIX1_Final_Report_200VTL_N3976_FINAL_METHOD_AUDITED_PRREF_STRICTALL.docx`
- FlexDATE learned wins: `3/5`
- FlexDATE DB wins: `5/5`

## 7. Suggested handoff usage

- Professor review: run the fast path and the verification checks.
- Student verification: run the fast path first; only run the slower completed-metrics regeneration if she wants to confirm the CSV package rebuild from the bundled FIX1 artifacts.
