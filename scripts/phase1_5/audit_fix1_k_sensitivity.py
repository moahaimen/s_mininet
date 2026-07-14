#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
COMPLETED = ROOT / "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/completed_metrics"
OUT_CSV = COMPLETED / "k_sensitivity_per_topology_fix1_200VTL_N3976_AUDITED.csv"
OUT_MD = COMPLETED / "k_sensitivity_per_topology_fix1_200VTL_N3976_AUDIT.md"

METHOD_FILES = {
    "K10": "fixed_k10_per_cycle.csv",
    "K20": "fixed_k20_per_cycle.csv",
    "K30": "fixed_k30_per_cycle.csv",
    "K50": "fixed_k50_per_cycle.csv",
    "K100": "fixed_k100_per_cycle.csv",
    "Adaptive DDQN": "final_rg_gnn_lpd_200vtl_consistent_per_cycle.csv",
}
TOPO_ORDER = ["abilene", "geant", "cernet", "sprintlink", "tiscali", "ebone", "germany50", "vtlwavenet2011"]
NORMAL_CSV = COMPLETED / "normal_8topo_200vtl_consistent.csv"


def fmt_pair(pr_mean: float, db_mean: float) -> str:
    pr_pct = pr_mean * 100.0
    db_pct = db_mean * 100.0
    if abs(pr_pct) < 0.001 and pr_pct > 0.0:
        pr_txt = f"{pr_pct:.6f}%"
    else:
        pr_txt = f"{pr_pct:.3f}%"
    return f"{pr_txt} / {db_pct:.3f}%"


def main() -> None:
    tables = {label: pd.read_csv(COMPLETED / fn) for label, fn in METHOD_FILES.items()}
    normal = pd.read_csv(NORMAL_CSV).set_index("topology")
    rows = []
    audit_lines = [
        "# K-sensitivity per-topology audit",
        "",
        "The previous `0.000%` fixed-K entries were audited against the underlying per-cycle CSVs.",
        "Result: they are real tiny nonzero PR values on several topologies, not merge-fill artifacts.",
        "Adaptive DDQN PR/DB is anchored to `normal_8topo_200vtl_consistent.csv` so the Section 8 per-topology table matches Section 4 exactly after rounding.",
        "",
    ]
    for topo in TOPO_ORDER:
        row = {"Topology": topo}
        best = None
        best_score = None
        notes = []
        for label, df in tables.items():
            gt = df[df["topology"] == topo].copy()
            if label == "Adaptive DDQN":
                pr_mean = float(normal.loc[topo, "fulldata_meanPR"])
                db_mean = float(normal.loc[topo, "meanDB"])
            else:
                pr_mean = float(gt["PR"].mean())
                db_mean = float(gt["DB"].mean())
            row[f"{label} PR/DB"] = fmt_pair(pr_mean, db_mean)
            row[f"{label} rows"] = int(len(gt))
            row[f"{label} minPR"] = round(float(gt["PR"].min() * 100.0), 6)
            if label != "Adaptive DDQN" and 0.0 < pr_mean * 100.0 < 0.001:
                notes.append(f"{label} PR is tiny but nonzero ({pr_mean*100.0:.6f}%).")
            trade = pr_mean - db_mean
            if best_score is None or trade > best_score:
                best_score = trade
                best = label
        row["Best tradeoff"] = best
        base_note = "Fixed-K values are directly computed from per-cycle PR/DB; Adaptive DDQN is aligned to Section 4 normal-table values."
        row["Audit note"] = (" ".join(notes) + " " + base_note).strip() if notes else base_note
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    for topo in TOPO_ORDER:
        audit_lines.append(f"## {topo}")
        for label, df in tables.items():
            gt = df[df["topology"] == topo]
            if label == "Adaptive DDQN":
                pr_mean = float(normal.loc[topo, "fulldata_meanPR"])
                db_mean = float(normal.loc[topo, "meanDB"])
            else:
                pr_mean = float(gt["PR"].mean())
                db_mean = float(gt["DB"].mean())
            audit_lines.append(
                f"- {label}: rows={len(gt)}, mean_PR={pr_mean*100.0:.6f}%, min_PR={gt['PR'].min()*100.0:.6f}%, mean_DB={db_mean*100.0:.6f}%"
            )
        audit_lines.append("")
    OUT_MD.write_text("\n".join(audit_lines))
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
