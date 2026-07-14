#!/usr/bin/env python3
from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
OUT = ROOT / "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50"
COMPLETED = OUT / "FINAL_REPORT_FIX1/completed_metrics"
PREPASS = pickle.load(open(OUT / "_prepass.pkl", "rb"))
PATHOPT = ROOT / "results/gnn_lpd_dqn_selective_db_lp/pathopt_ref"
WINDOWS = {
    "abilene": (2016, 4032),
    "geant": (672, 1344),
    "cernet": (200, 400),
    "sprintlink": (200, 400),
    "tiscali": (200, 400),
    "ebone": (200, 400),
    "germany50": (0, 288),
    "vtlwavenet2011": (0, 40),
}


def pct(x: float) -> float:
    return round(float(x) * 100.0, 3)


def ms(x: float) -> float:
    return round(float(x), 3)


def summarize(df: pd.DataFrame) -> dict:
    return {
        "Mean PR": pct(df["PR"].mean()),
        "PR>=0.90": round(float((df["PR"] >= 0.90).mean() * 100.0), 3),
        "PR>=0.95": round(float((df["PR"] >= 0.95).mean() * 100.0), 3),
        "Mean DB": pct(df["DB"].mean()),
        "P95 DB": pct(np.percentile(df["DB"], 95)),
        "Mean ms": ms(df["decision_ms"].mean()),
        "P95 ms": ms(np.percentile(df["decision_ms"], 95)),
        "Min PR": pct(df["PR"].min()),
        "N": int(len(df)),
    }


def load_pathopt(topo: str) -> dict[int, float]:
    p = PATHOPT / f"pathopt_{topo}.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {int(r.timestep): float(r.pathopt_mlu) for r in df.itertuples()}


WRONG_REF = {t: load_pathopt(t) for t in WINDOWS}
RIGHT_REF = {
    topo: {int(k): float(v) for k, v in PREPASS[(topo, lo, hi)]["opt"].items()}
    for topo, (lo, hi) in WINDOWS.items()
    if (topo, lo, hi) in PREPASS and "opt" in PREPASS[(topo, lo, hi)]
}


def repair_per_cycle_csvs() -> dict[str, pd.DataFrame]:
    repaired = {}
    for p in sorted(COMPLETED.glob("*_per_cycle.csv")):
        df = pd.read_csv(p)
        if not {"topology", "tm_index", "PR"}.issubset(df.columns):
            repaired[p.name] = df
            continue
        if (df["tm_index"] < 0).all():
            repaired[p.name] = df
            continue
        changed = False
        for topo in ["cernet", "sprintlink", "tiscali", "ebone"]:
            mask = df["topology"] == topo
            if not mask.any():
                continue
            wr = WRONG_REF.get(topo, {})
            rr = RIGHT_REF.get(topo, {})
            for idx, row in df.loc[mask, ["tm_index", "PR"]].iterrows():
                t = int(row.tm_index)
                old_pr = float(row.PR)
                wrong = wr.get(t)
                right = rr.get(t)
                if wrong is None or right is None or old_pr <= 0:
                    continue
                achieved = wrong / old_pr
                if achieved <= 0:
                    continue
                new_pr = min(1.0, right / achieved)
                if not math.isclose(new_pr, old_pr, rel_tol=0.0, abs_tol=1e-12):
                    df.at[idx, "PR"] = new_pr
                    changed = True
        if changed:
            df.to_csv(p, index=False)
        repaired[p.name] = df
    return repaired


def rewrite_summaries(per_cycle: dict[str, pd.DataFrame]) -> None:
    method_to_df = {}
    for name, df in per_cycle.items():
        if "method" in df.columns and len(df):
            method_to_df[str(df["method"].iloc[0])] = df

    def row_for(method: str) -> dict:
        df = method_to_df[method]
        out = {"Method": method}
        out.update(summarize(df))
        return out

    baseline_methods = [
        "ECMP",
        "OSPF / shortest path (ECMP surrogate)",
        "Top-K Demand (K50)",
        "Bottleneck Top-K (K50)",
        "GNN-only fixed K50",
        "LPD-only fixed K50 (closest faithful: deployed rank surrogate)",
        "GNN+LPD fixed K50 (closest faithful: deployed rank surrogate)",
        "DDQN without GNN-LPD",
        "DDQN without bottleneck relief",
        "Final RG-GNN-LPD",
    ]

    # Keep imported non-per-cycle DDQN rows from existing summary file.
    old_base = pd.read_csv(COMPLETED / "baseline_comparison_fix1_COMPLETED.csv")
    imported = old_base[old_base["Method"].isin(["DDQN without GNN-LPD", "DDQN without bottleneck relief"])].copy()
    rows = [row_for(m) for m in baseline_methods if m not in set(imported["Method"])]
    rows.extend(imported.to_dict("records"))
    pd.DataFrame(rows)[old_base.columns].to_csv(COMPLETED / "baseline_comparison_fix1_COMPLETED.csv", index=False)

    scorer_methods = [
        "GNN-only fixed K50",
        "LPD-only fixed K50 (closest faithful: deployed rank surrogate)",
        "Bottleneck Top-K (K50)",
        "GNN+LPD fixed K50 (closest faithful: deployed rank surrogate)",
        "GNN+LPD+bottleneck fixed K50 (closest faithful: deployed rank surrogate)",
    ]
    pd.DataFrame([row_for(m) for m in scorer_methods]).to_csv(COMPLETED / "scorer_ablation_fix1_COMPLETED.csv", index=False)

    mapping = [
        ("Fixed K30", "Fixed K30"),
        ("Fixed K50", "Fixed K50"),
        ("Fixed K800", "Fixed K800"),
        ("DDQN gate", "Final RG-GNN-LPD"),
        ("DDQN without teacher cycle-0", "DDQN without teacher cycle-0"),
        ("DDQN with teacher cycle-0", "Final RG-GNN-LPD"),
    ]
    rows = []
    for label, method in mapping:
        out = row_for(method)
        out["Method"] = label
        rows.append(out)
    pd.DataFrame(rows).to_csv(COMPLETED / "policy_ablation_fix1_COMPLETED.csv", index=False)

    opt_rows = []
    purpose = {
        "ECMP only": "no optimization",
        "Ranking only, no LP": "ranking evidence but no feasible split optimization",
        "LP for selected ODs": "final efficient optimizer",
        "Full-OD LP": "optimal reference but slower",
    }
    source = {
        "ECMP only": "ECMP",
        "Ranking only, no LP": "Ranking only, no LP (ECMP carry)",
        "LP for selected ODs": "Final RG-GNN-LPD",
        "Full-OD LP": "Full-OD LP (closest faithful full-OD selected-flow LP)",
    }
    for label, method in source.items():
        out = row_for(method)
        out["Method"] = label
        out["Purpose"] = purpose[label]
        out["Notes"] = method
        opt_rows.append(out)
    pd.DataFrame(opt_rows).to_csv(COMPLETED / "optimization_ablation_fix1_COMPLETED.csv", index=False)

    k_methods = [
        ("K10", "Fixed K10"),
        ("K20", "Fixed K20"),
        ("K30", "Fixed K30"),
        ("K50", "Fixed K50"),
        ("K100", "Fixed K100"),
        ("Adaptive DDQN", "Final RG-GNN-LPD"),
    ]
    rows = []
    for label, method in k_methods:
        out = row_for(method)
        out["Method / Budget"] = label
        rows.append(out)
    pd.DataFrame(rows).to_csv(COMPLETED / "k_sensitivity_fix1_COMPLETED.csv", index=False)

    ft = pd.read_csv(COMPLETED / "first_tm_ablation_fix1_COMPLETED.csv")
    for version, method in [
        ("FIX1 with teacher cycle-0", "Final RG-GNN-LPD"),
        ("FIX1 without teacher cycle-0", "DDQN without teacher cycle-0"),
    ]:
        out = row_for(method)
        ft.loc[ft["Version"] == version, "Min PR"] = out["Min PR"]
        ft.loc[ft["Version"] == version, "Mean PR"] = out["Mean PR"]
        ft.loc[ft["Version"] == version, "Mean DB"] = out["Mean DB"]
        ft.loc[ft["Version"] == version, "P95 ms"] = out["P95 ms"]
    ft.to_csv(COMPLETED / "first_tm_ablation_fix1_COMPLETED.csv", index=False)


if __name__ == "__main__":
    per_cycle = repair_per_cycle_csvs()
    rewrite_summaries(per_cycle)
    print("Repaired completed_metrics PR references using FIX1 _prepass opt where available.")
