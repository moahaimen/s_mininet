# Student Run Instructions After Download

This file gives the exact commands to run the downloaded repository and rebuild the final strict-all report from the bundled FIX1 artifacts.

## What this package supports

This repository snapshot supports:

- smoke-test verification
- strict-all audit checks
- rebuilding the final report from the bundled completed metrics

This repository snapshot does **not** currently support full raw-data retraining from scratch on another machine, because some `data/` folders are still symlinks to a machine-local source path rather than bundled standalone datasets.

Use this package for **report/result reproduction from bundled artifacts**.

## 1. Clone the repository

```bash
git clone https://github.com/moahaimen/s23_network.git
cd s23_network
```

## 2. Create and activate a virtual environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install python-docx pypdf pdf2image
```

If PDF export is needed, install LibreOffice and make sure `soffice` is available on `PATH`.

## 4. Verify that the required bundled artifacts exist

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

## 5. Run the smoke test

```bash
python scripts/phase1_5/windows_smoke_test.py
```

Expected final line:

```text
READY FOR STUDENT RUN
```

## 6. Run the strict-all audits

```bash
python scripts/phase1_5/audit_fix1_k_sensitivity.py
python scripts/phase1_5/audit_pr_reference.py
```

## 7. Rebuild the final report from the bundled completed metrics

```bash
python scripts/phase1_5/build_fix1_completed_report.py
```

Expected output:

```text
results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/RG_GNN_LPD_FIX1_Final_Report_200VTL_N3976_FINAL_METHOD_AUDITED_PRREF_STRICTALL.docx
```

## 8. Export PDF

If LibreOffice is installed:

```bash
soffice --headless --convert-to pdf \
  --outdir results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1 \
  results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/RG_GNN_LPD_FIX1_Final_Report_200VTL_N3976_FINAL_METHOD_AUDITED_PRREF_STRICTALL.docx
```

Expected PDF:

```text
results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/RG_GNN_LPD_FIX1_Final_Report_200VTL_N3976_FINAL_METHOD_AUDITED_PRREF_STRICTALL.pdf
```

## 9. Optional verification checks

### 9.1 Pooled normal row

```bash
python3 - <<'PY'
import pandas as pd
p = "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/completed_metrics/normal_pooled_3976_consistent.csv"
df = pd.read_csv(p).iloc[0]
print(df.to_dict())
assert int(df["N"]) == 3976
assert round(float(df["Mean PR"]), 3) == 98.627
assert round(float(df["Mean DB"]), 3) == 0.456
print("Pooled normal row matches expected values.")
PY
```

### 9.2 Adaptive per-topology values must match Section 4

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
print("Adaptive per-topology values match the main normal table.")
PY
```

## 10. Important limitation

This repository snapshot is suitable for:

- reproducing the final strict-all report
- auditing the bundled final artifacts
- rebuilding the final DOCX/PDF from the committed results

This repository snapshot is **not** a complete fresh-training package for another machine, because the `data/` folders are not fully bundled standalone datasets in this snapshot.

If a full train-from-scratch package is needed, the raw datasets must be bundled as real in-repo files instead of symlinks.
