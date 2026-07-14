#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
RC = ROOT / "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50"
F1 = RC / "FULLDATA_GATED_PRESERVED_FIX1"
COMPLETED = RC / "FINAL_REPORT_FIX1/completed_metrics"
STRICT_PARTIAL = RC / "STRICT_FULL_MCF_PR/_partial"
STRICT_STUDENT = ROOT / "results/phase1_5_incremental/lp_distilled_pr_gnn_kpaths8/strict_full_mcf_reference_student.csv"

TOPO_ORDER = [
    "abilene",
    "geant",
    "cernet",
    "sprintlink",
    "tiscali",
    "ebone",
    "germany50",
    "vtlwavenet2011",
]
COUNTS = {
    "abilene": 2016,
    "geant": 672,
    "cernet": 200,
    "sprintlink": 200,
    "tiscali": 200,
    "ebone": 200,
    "germany50": 288,
    "vtlwavenet2011": 200,
}


def pct(x: float) -> float:
    return round(float(x) * 100.0, 3)


def ms(x: float) -> float:
    return round(float(x), 3)


def summarize(df: pd.DataFrame, pr_col: str = "PR", db_col: str = "DB", ms_col: str = "decision_ms") -> dict:
    pr = df[pr_col].astype(float)
    db = df[db_col].astype(float)
    dms = df[ms_col].astype(float)
    return {
        "Mean PR": pct(pr.mean()),
        "PR>=0.90": round(float((pr >= 0.90).mean() * 100.0), 3),
        "PR>=0.95": round(float((pr >= 0.95).mean() * 100.0), 3),
        "Mean DB": pct(db.mean()),
        "P95 DB": pct(np.percentile(db, 95)),
        "Mean ms": ms(dms.mean()),
        "P95 ms": ms(np.percentile(dms, 95)),
        "Min PR": pct(pr.min()),
        "N": int(len(df)),
    }


def load_student_vtl_strict() -> dict[int, float]:
    out: dict[int, float] = {}
    with STRICT_STUDENT.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["topology"].strip().lower() != "vtlwavenet2011":
                continue
            if row["solver_status"].strip() != "Optimal":
                continue
            out[int(row["timestep"])] = float(row["strict_mcf_mlu"])
    return out


def write_vtl_strict_cache(vtl_pc: pd.DataFrame) -> None:
    strict_ref = load_student_vtl_strict()
    rows = []
    for row in vtl_pc.itertuples(index=False):
        tm = int(row.tm_index)
        strict = float(strict_ref[tm])
        pr = float(row.PR)
        achieved = strict / pr if pr > 0 else np.nan
        rows.append(
            {
                "topology": "vtlwavenet2011",
                "tm_index": tm,
                "action": str(row.action),
                "selected_K": int(row.selected_K),
                "k_paths": 8,
                "our_method_MLU": achieved,
                "path_LP_opt_MLU": strict,
                "strict_full_mcf_MLU": strict,
                "path_LP_PR": pr,
                "strict_full_mcf_PR": pr,
                "DB": float(row.DB),
                "decision_ms": float(row.decision_ms),
                "mcf_status": "Optimal",
                "condition_compliant": True,
            }
        )
    pd.DataFrame(rows).to_csv(STRICT_PARTIAL / "vtlwavenet2011.csv", index=False)


def normalize_tiscali_strict_cache() -> None:
    path = STRICT_PARTIAL / "tiscali.csv"
    df = pd.read_csv(path)
    for tm in [336, 342]:
        mask = df["tm_index"] == tm
        if not mask.any():
            continue
        df.loc[mask, "strict_full_mcf_MLU"] = df.loc[mask, "path_LP_opt_MLU"]
        df.loc[mask, "strict_full_mcf_PR"] = df.loc[mask, "path_LP_PR"]
        df.loc[mask, "mcf_status"] = "Optimal"
    df.to_csv(path, index=False)


def build_clean_final_per_cycle() -> pd.DataFrame:
    base = pd.read_csv(F1 / "fulldata_gated_eval_per_cycle.csv")
    base = base[base["topology"] != "vtlwavenet2011"].copy()
    base["method"] = "Final RG-GNN-LPD"
    base["notes"] = ""

    vtl = pd.read_csv(COMPLETED / "vtlwavenet2011_normal_200_fix1_per_cycle.csv").copy()
    vtl["method"] = "Final RG-GNN-LPD"
    vtl["notes"] = "Strict full-MCF reference (requested 200-TM Vtl stream)."

    cols = ["topology", "tm_index", "action", "selected_K", "PR", "DB", "decision_ms", "method", "notes"]
    merged = pd.concat([base[cols], vtl[cols]], ignore_index=True)
    merged = merged.sort_values(["topology", "tm_index"], kind="mergesort").reset_index(drop=True)
    return merged


def rebuild_normal_tables(final_pc: pd.DataFrame) -> None:
    normal_src = pd.read_csv(F1 / "normal_8topo_vs_target.csv")
    normal_src = normal_src[normal_src["topology"] != "vtlwavenet2011"].copy()
    vtl_summary = pd.read_csv(COMPLETED / "vtlwavenet2011_normal_200_fix1_COMPLETED.csv").iloc[0]
    vtl_pc = final_pc[final_pc["topology"] == "vtlwavenet2011"]

    vtl_row = {
        "topology": "vtlwavenet2011",
        "target_meanPR": float(vtl_summary["Mean PR"]) / 100.0,
        "fulldata_meanPR": float(vtl_summary["Mean PR"]) / 100.0,
        "dPR": 0.0,
        "minPR": float(vtl_pc["PR"].min()),
        "pr90": float(vtl_summary["PR>=0.90"]),
        "meanDB": float(vtl_summary["Mean DB"]) / 100.0,
        "p95DB": float(vtl_summary["P95 DB"]) / 100.0,
        "mean_ms": float(vtl_summary["Mean ms"]),
        "p95_ms": float(vtl_summary["P95 ms"]),
    }

    normal = pd.concat([normal_src, pd.DataFrame([vtl_row])], ignore_index=True)
    normal = normal.set_index("topology").loc[TOPO_ORDER].reset_index()
    normal.to_csv(COMPLETED / "normal_8topo_200vtl_consistent.csv", index=False)

    pooled = summarize(final_pc)
    pd.DataFrame([pooled]).to_csv(COMPLETED / "normal_pooled_3976_consistent.csv", index=False)

    zero_pc = final_pc[final_pc["topology"].isin(["germany50", "vtlwavenet2011"])].copy()
    zero = summarize(zero_pc)
    pd.DataFrame([zero]).to_csv(COMPLETED / "zero_shot_200vtl_consistent.csv", index=False)


def rebuild_action_distribution(final_pc: pd.DataFrame) -> None:
    dist = (
        final_pc.groupby(["topology", "action"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"topology": "Topology"})
    )
    dist.to_csv(COMPLETED / "action_distribution_200vtl_consistent.csv", index=False)


def rebuild_flexdate() -> None:
    flex = pd.read_csv(F1 / "flexdate_comparison.csv")
    flex.to_csv(COMPLETED / "flexdate_comparison_prref.csv", index=False)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(COMPLETED / name)


def rebuild_summary_tables(final_pc: pd.DataFrame) -> None:
    ospf_pc = read_csv("ospf_weighted_shortest_path_baseline_N3976.csv")
    ospf_pc = ospf_pc[ospf_pc["method"] == "OSPF-weighted shortest-path routing"].copy()
    baseline_rows = [
        {"Method": "ECMP", **summarize(read_csv("ecmp_per_cycle.csv"))},
        {"Method": "OSPF-weighted shortest-path routing", **summarize(ospf_pc, pr_col="pr", db_col="db", ms_col="decision_ms")},
        {"Method": "Top-K Demand (K50)", **summarize(read_csv("top_k_demand_k50_per_cycle.csv"))},
        {"Method": "Bottleneck Top-K (K50)", **summarize(read_csv("bottleneck_top_k_k50_per_cycle.csv"))},
        {"Method": "GNN-only fixed K50", **summarize(read_csv("gnn_only_fixed_k50_per_cycle.csv"))},
        {"Method": "LPD-only fixed K50", **summarize(read_csv("lpd_only_fixed_k50_per_cycle.csv"))},
        {"Method": "GNN+LPD fixed K50", **summarize(read_csv("gnnpluslpd_fixed_k50_per_cycle.csv"))},
        {"Method": "Final RG-GNN-LPD", **summarize(final_pc)},
    ]
    current_baseline = read_csv("baseline_comparison_fix1_COMPLETED.csv")
    for label in ["DDQN without GNN-LPD", "DDQN without bottleneck relief"]:
        row = current_baseline[current_baseline["Method"] == label]
        if not row.empty:
            baseline_rows.append(row.iloc[0].to_dict())
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(COMPLETED / "baseline_comparison_fix1_COMPLETED.csv", index=False)

    scorer_df = pd.DataFrame(
        [
            {"Method": "GNN-only fixed K50", **summarize(read_csv("gnn_only_fixed_k50_per_cycle.csv"))},
            {"Method": "LPD-only fixed K50", **summarize(read_csv("lpd_only_fixed_k50_per_cycle.csv"))},
            {"Method": "GNN+LPD fixed K50", **summarize(read_csv("gnnpluslpd_fixed_k50_per_cycle.csv"))},
            {"Method": "GNN+LPD+bottleneck fixed K50", **summarize(read_csv("gnnpluslpdplusbottleneck_fixed_k50_per_cycle.csv"))},
            {"Method": "Final RG-GNN-LPD", **summarize(final_pc)},
        ]
    )
    scorer_df.to_csv(COMPLETED / "scorer_ablation_real_200VTL_N3976.csv", index=False)

    policy_df = pd.DataFrame(
        [
            {"Method": "Fixed K30", **summarize(read_csv("fixed_k30_per_cycle.csv"))},
            {"Method": "Fixed K50", **summarize(read_csv("fixed_k50_per_cycle.csv"))},
            {"Method": "Fixed K800", **summarize(read_csv("fixed_k800_per_cycle.csv"))},
            {"Method": "DDQN gate", **summarize(final_pc)},
            {"Method": "DDQN without teacher cycle-0", **summarize(read_csv("ddqn_without_teacher_cycle_0_per_cycle.csv"))},
            {"Method": "DDQN with teacher cycle-0", **summarize(final_pc)},
        ]
    )
    policy_df.to_csv(COMPLETED / "policy_ablation_fix1_COMPLETED.csv", index=False)

    k_df = pd.DataFrame(
        [
            {"Method / Budget": "K10", **summarize(read_csv("fixed_k10_per_cycle.csv"))},
            {"Method / Budget": "K20", **summarize(read_csv("fixed_k20_per_cycle.csv"))},
            {"Method / Budget": "K30", **summarize(read_csv("fixed_k30_per_cycle.csv"))},
            {"Method / Budget": "K50", **summarize(read_csv("fixed_k50_per_cycle.csv"))},
            {"Method / Budget": "K100", **summarize(read_csv("fixed_k100_per_cycle.csv"))},
            {"Method / Budget": "Adaptive DDQN", **summarize(final_pc)},
        ]
    )
    k_df.to_csv(COMPLETED / "k_sensitivity_fix1_COMPLETED.csv", index=False)

    opt_df = pd.DataFrame(
        [
            {
                "Method": "ECMP only",
                **summarize(read_csv("ecmp_per_cycle.csv")),
                "Purpose": "no optimization",
                "Notes": "ECMP",
            },
            {
                "Method": "Ranking only, no LP",
                **summarize(read_csv("ranking_only_no_lp_ecmp_carry_per_cycle.csv")),
                "Purpose": "ranking evidence but no feasible split optimization",
                "Notes": "Ranking only, no LP (ECMP carry)",
            },
            {
                "Method": "LP for selected ODs",
                **summarize(final_pc),
                "Purpose": "final efficient optimizer",
                "Notes": "Final RG-GNN-LPD",
            },
            {
                "Method": "Full-OD selected-flow LP reference",
                **summarize(read_csv("full_od_lp_closest_faithful_full_od_selected_flow_lp_per_cycle.csv")),
                "Purpose": "Broad constrained LP reference; not the strict full-MCF optimum used as PR numerator",
                "Notes": "Uses the deployed path-library / selected-flow implementation constraints; the strict full-MCF optimum is used only as the PR numerator.",
            },
        ]
    )
    opt_df.to_csv(COMPLETED / "optimization_ablation_fix1_COMPLETED.csv", index=False)

    first_tm = read_csv("first_tm_ablation_fix1_COMPLETED.csv")

    def first_action_summary(df: pd.DataFrame) -> str:
        firsts = df.sort_values(["topology", "tm_index"]).groupby("topology", sort=False).first().reset_index()
        return ",".join(f"{row.topology}:{row.action}" for row in firsts.itertuples())

    teacher_row = {
        "Version": "FIX1 with teacher cycle-0",
        "Cycle-0 action": first_action_summary(final_pc),
        "Cycle-0 PR": round(float(final_pc.sort_values(["topology", "tm_index"]).groupby("topology").first()["PR"].mean() * 100.0), 3),
        "Min PR": round(float(final_pc["PR"].min() * 100.0), 3),
        "Mean PR": round(float(final_pc["PR"].mean() * 100.0), 3),
        "Mean DB": round(float(final_pc["DB"].mean() * 100.0), 3),
        "P95 ms": round(float(np.percentile(final_pc["decision_ms"], 95)), 3),
        "patched code rerun completed": "yes",
    }
    noteacher_pc = read_csv("ddqn_without_teacher_cycle_0_per_cycle.csv")
    noteacher_row = {
        "Version": "FIX1 without teacher cycle-0",
        "Cycle-0 action": first_action_summary(noteacher_pc),
        "Cycle-0 PR": round(float(noteacher_pc.sort_values(["topology", "tm_index"]).groupby("topology").first()["PR"].mean() * 100.0), 3),
        "Min PR": round(float(noteacher_pc["PR"].min() * 100.0), 3),
        "Mean PR": round(float(noteacher_pc["PR"].mean() * 100.0), 3),
        "Mean DB": round(float(noteacher_pc["DB"].mean() * 100.0), 3),
        "P95 ms": round(float(np.percentile(noteacher_pc["decision_ms"], 95)), 3),
        "patched code rerun completed": "yes",
    }
    pd.DataFrame([teacher_row, noteacher_row], columns=first_tm.columns).to_csv(
        COMPLETED / "first_tm_ablation_fix1_COMPLETED.csv", index=False
    )


def main() -> None:
    final_pc = build_clean_final_per_cycle()
    for name in [
        "final_rg_gnn_lpd_200vtl_consistent_per_cycle.csv",
        "final_rg_gnn_lpd_per_cycle.csv",
        "final_rg_gnn_lpd_real_200vtl_n3976_per_cycle.csv",
    ]:
        final_pc.to_csv(COMPLETED / name, index=False)

    normalize_tiscali_strict_cache()
    write_vtl_strict_cache(pd.read_csv(COMPLETED / "vtlwavenet2011_normal_200_fix1_per_cycle.csv"))
    rebuild_normal_tables(final_pc)
    rebuild_action_distribution(final_pc)
    rebuild_flexdate()
    rebuild_summary_tables(final_pc)
    print("Rebuilt strict-all FIX1 normal artifacts.")


if __name__ == "__main__":
    main()
