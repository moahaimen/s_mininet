#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

import scripts.phase1_5.run_fix1_completed_metrics as R


ROOT = Path("/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
COMPLETED = ROOT / "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/completed_metrics"
WINDOWS = {
    "abilene": (2016, 4032),
    "geant": (672, 1344),
    "cernet": (200, 400),
    "sprintlink": (200, 400),
    "tiscali": (200, 400),
    "ebone": (200, 400),
    "germany50": (0, 288),
    "vtlwavenet2011": (0, 200),
}


def summary_row(name: str, df: pd.DataFrame) -> dict:
    s = R.summarize(df)
    return {"Method": name, **s, "N": int(len(df))}


def per_topo_row(name: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    for topo, g in df.groupby("topology", sort=False):
        s = R.summarize(g)
        rows.append({"Method": name, "Topology": topo, **s, "N": int(len(g))})
    return rows


def safe_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "plus")
        .replace("-", "_")
        .replace(",", "")
    )


def main() -> None:
    if not R.LPD_AVAILABLE:
        raise RuntimeError(f"LP-distilled models still unavailable: {R.LPD_LOAD_ERROR}")

    final_locked = COMPLETED / "final_rg_gnn_lpd_200vtl_consistent_per_cycle.csv"
    specs = [
        ("GNN-only fixed K50", dict(kind="fixed", ranking_mode="gnn", fixed_k=50, notes="Real scorer ablation: trained GNN score only.")),
        ("LPD-only fixed K50", dict(kind="fixed", ranking_mode="lpd", fixed_k=50, notes="Real scorer ablation: LP-distilled HGB score only.")),
        ("GNN+LPD fixed K50", dict(kind="fixed", ranking_mode="gnn_lpd", fixed_k=50, notes="Real scorer ablation: trained GNN + LP-distilled HGB fusion.")),
        ("GNN+LPD+bottleneck fixed K50", dict(kind="fixed", ranking_mode="gnn_lpd_bottleneck", fixed_k=50, notes="Real scorer ablation: trained GNN + LP-distilled HGB + bottleneck fusion.")),
        ("Final RG-GNN-LPD", dict(kind="adaptive", ranking_mode="cache_final", teacher_cycle0=True, notes="Final full RG-GNN-LPD method.")),
    ]

    per_cycle = {}
    summary_rows = []
    topo_rows = []
    for name, kwargs in specs:
        if name == "Final RG-GNN-LPD" and final_locked.exists():
            df = pd.read_csv(final_locked)
        else:
            df = R.eval_method(name, windows=WINDOWS, **kwargs)
        per_cycle[name] = df
        df.to_csv(COMPLETED / f"{safe_name(name)}_real_200vtl_n3976_per_cycle.csv", index=False)
        summary_rows.append(summary_row(name, df))
        topo_rows.extend(per_topo_row(name, df))

    summary_df = pd.DataFrame(summary_rows)
    topo_df = pd.DataFrame(topo_rows)
    summary_df.to_csv(COMPLETED / "scorer_ablation_real_200VTL_N3976.csv", index=False)
    topo_df.to_csv(COMPLETED / "scorer_ablation_real_200VTL_N3976_per_topology.csv", index=False)

    # Replace the report-facing scorer table.
    summary_df.to_csv(COMPLETED / "scorer_ablation_fix1_COMPLETED.csv", index=False)

    # Update baseline rows for the three scorer-based methods so no surrogate labels remain.
    base = pd.read_csv(COMPLETED / "baseline_comparison_fix1_COMPLETED.csv")
    base = base[
        ~base["Method"].isin(
            [
                "GNN-only fixed K50",
                "LPD-only fixed K50 (closest faithful: deployed rank surrogate)",
                "GNN+LPD fixed K50 (closest faithful: deployed rank surrogate)",
                "LPD-only fixed K50",
                "GNN+LPD fixed K50",
            ]
        )
    ].copy()
    inserts = summary_df[summary_df["Method"].isin(["GNN-only fixed K50", "LPD-only fixed K50", "GNN+LPD fixed K50"])].copy()
    parts = []
    inserted = False
    for _, row in base.iterrows():
        parts.append(row.to_dict())
        if row["Method"] == "Bottleneck Top-K (K50)" and not inserted:
            for _, ins in inserts.iterrows():
                parts.append(ins.to_dict())
            inserted = True
    if not inserted:
        for _, ins in inserts.iterrows():
            parts.append(ins.to_dict())
    pd.DataFrame(parts).to_csv(COMPLETED / "baseline_comparison_fix1_COMPLETED.csv", index=False)
    print(f"Wrote {COMPLETED / 'scorer_ablation_real_200VTL_N3976.csv'}")
    print(f"Wrote {COMPLETED / 'scorer_ablation_real_200VTL_N3976_per_topology.csv'}")


if __name__ == "__main__":
    main()
