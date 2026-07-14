#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.phase1_5 import run_fix1_completed_metrics as M


COMPLETED = M.COMPLETED


def save(df: pd.DataFrame, name: str) -> None:
    path = COMPLETED / name
    df.to_csv(path, index=False)
    print(f"[write] {path.relative_to(M.ROOT)}", flush=True)


def replace_or_append(df: pd.DataFrame, row: dict, key: str = "Method") -> pd.DataFrame:
    out = df.copy()
    if row[key] in set(out[key].astype(str)):
        out = out[out[key].astype(str) != str(row[key])].copy()
    out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
    return out


def summary_row(df: pd.DataFrame, label: str) -> dict:
    row = M.summarize(df)
    row["Method"] = label
    return row


def first_action_summary(pc: pd.DataFrame) -> str:
    first = pc.sort_values(["topology", "tm_index"]).groupby("topology").first()
    return ",".join(f"{topo}:{first.loc[topo, 'action']}" for topo in sorted(first.index))


def build_normal_tables(final_pc: pd.DataFrame) -> None:
    target_src = pd.read_csv(COMPLETED / "normal_8topo_200vtl_consistent.csv").set_index("topology")
    rows = []
    for topo in M.TOP_ORDER:
        g = final_pc[final_pc["topology"] == topo].copy()
        rows.append(
            {
                "topology": topo,
                "target_meanPR": float(target_src.loc[topo, "target_meanPR"]),
                "fulldata_meanPR": round(float(g["PR"].mean()), 6),
                "dPR": round(float(g["PR"].mean()) - float(target_src.loc[topo, "target_meanPR"]), 6),
                "minPR": round(float(g["PR"].min()), 6),
                "pr90": round(float((g["PR"] >= 0.90).mean() * 100.0), 3),
                "meanDB": round(float(g["DB"].mean()), 6),
                "p95DB": round(float(np.percentile(g["DB"], 95)), 6),
                "mean_ms": round(float(g["decision_ms"].mean()), 3),
                "p95_ms": round(float(np.percentile(g["decision_ms"], 95)), 3),
                "mean_mlu": round(float(g["achieved_mlu"].mean()), 6),
                "p95_mlu": round(float(np.percentile(g["achieved_mlu"], 95)), 6),
                "worst_mlu": round(float(g["achieved_mlu"].max()), 6),
            }
        )
    normal = pd.DataFrame(rows)
    save(normal, "normal_8topo_200vtl_consistent.csv")
    save(pd.DataFrame([M.summarize(final_pc)]), "normal_pooled_3976_consistent.csv")
    zero = final_pc[final_pc["topology"].isin(["germany50", "vtlwavenet2011"])].copy()
    save(pd.DataFrame([M.summarize(zero)]), "zero_shot_200vtl_consistent.csv")

    flex_targets = {
        "abilene": {"target_PR": 0.9580, "target_DB": 0.0513, "note": "source-locked"},
        "cernet": {"target_PR": 0.9750, "target_DB": 0.0183, "note": "source-locked"},
        "geant": {"target_PR": 0.9950, "target_DB": 0.0296, "note": "source-locked"},
        "sprintlink": {"target_PR": 0.9990, "target_DB": 0.0510, "note": "source-locked"},
        "tiscali": {"target_PR": 0.9990, "target_DB": 0.0510, "note": "informal"},
    }
    flex_rows = []
    for topo, tgt in flex_targets.items():
        g = final_pc[final_pc["topology"] == topo]
        our_pr = float(g["PR"].mean())
        our_db = float(g["DB"].mean())
        flex_rows.append(
            {
                "topology": topo,
                "target_PR": tgt["target_PR"],
                "our_PR": round(our_pr, 6),
                "target_DB": tgt["target_DB"],
                "our_DB": round(our_db, 6),
                "win": bool(our_pr >= tgt["target_PR"] and our_db <= tgt["target_DB"]),
                "note": tgt["note"],
            }
        )
    save(pd.DataFrame(flex_rows), "flexdate_comparison_prref.csv")

    act_rows = []
    for topo in M.TOP_ORDER:
        g = final_pc[final_pc["topology"] == topo]
        vc = g["action"].value_counts()
        act_rows.append(
            {
                "Topology": topo,
                "K100": int(vc.get("K100", 0)),
                "K200": int(vc.get("K200", 0)),
                "K300": int(vc.get("K300", 0)),
                "K50": int(vc.get("K50", 0)),
                "K500": int(vc.get("K500", 0)),
                "K800": int(vc.get("K800", 0)),
                "KEEP": int(vc.get("KEEP", 0)),
            }
        )
    save(pd.DataFrame(act_rows), "action_distribution_200vtl_consistent.csv")


def build_mlu_solver_stability(
    final_pc: pd.DataFrame,
    ecmp_pc: pd.DataFrame,
    ospf_pc: pd.DataFrame,
    failure_df: pd.DataFrame | None = None,
) -> None:
    mlu_rows = []
    for topo in M.TOP_ORDER:
        ge = ecmp_pc[ecmp_pc["topology"] == topo]
        go = ospf_pc[ospf_pc["topology"] == topo]
        gf = final_pc[final_pc["topology"] == topo]
        ecmp_mean = float(ge["achieved_mlu"].mean())
        ospf_mean = float(go["achieved_mlu"].mean())
        final_mean = float(gf["achieved_mlu"].mean())
        mlu_rows.append(
            {
                "Topology": topo,
                "ECMP Mean MLU": round(ecmp_mean, 6),
                "OSPF Mean MLU": round(ospf_mean, 6),
                "Final Mean MLU": round(final_mean, 6),
                "Final P95 MLU": round(float(np.percentile(gf["achieved_mlu"], 95)), 6),
                "Worst-case (Maximum) MLU": round(float(gf["achieved_mlu"].max()), 6),
                "Improvement vs ECMP": round(((ecmp_mean - final_mean) / ecmp_mean * 100.0) if ecmp_mean > 0 else np.nan, 3),
                "Improvement vs OSPF": round(((ospf_mean - final_mean) / ospf_mean * 100.0) if ospf_mean > 0 else np.nan, 3),
            }
        )
    save(pd.DataFrame(mlu_rows), "mlu_per_topology_fix1_COMPLETED.csv")

    final_lp = final_pc[final_pc["lp_executed"] == 1].copy()
    solver_rows = []
    failure_max = {}
    if failure_df is not None and not failure_df.empty and "Max disconnected ODs" in failure_df.columns:
        for topo in M.TOP_ORDER:
            gfail = failure_df[failure_df["Topology"].astype(str).str.lower() == topo]
            failure_max[topo] = int(gfail["Max disconnected ODs"].max()) if len(gfail) else 0
        failure_max["pooled"] = int(failure_df["Max disconnected ODs"].max())

    for topo in M.TOP_ORDER + ["pooled"]:
        g = final_pc if topo == "pooled" else final_pc[final_pc["topology"] == topo]
        glp = final_lp if topo == "pooled" else final_lp[final_lp["topology"] == topo]
        optimal = int((glp["solver_status"] == "Optimal").sum())
        not_solved = int((glp["solver_status"] == "Not Solved").sum())
        infeasible = int(glp["solver_status"].astype(str).str.contains("Infeasible|BudgetInfeasible", case=False, regex=True).sum())
        solver_rows.append(
            {
                "Topology": topo,
                "LP-triggered cycles": int(len(glp)),
                "Solver success rate": round((optimal / len(glp) * 100.0) if len(glp) else 100.0, 3),
                "Optimal LP %": round((optimal / len(glp) * 100.0) if len(glp) else 100.0, 3),
                "Infeasible cycles": infeasible,
                "Timeout/Not solved cycles": not_solved,
                "Capacity-violation cycles": int((g["capacity_overload"] > 1e-9).sum()),
                "Maximum capacity violation": round(float(g["capacity_overload"].max()), 6),
                "Disconnected ODs (normal protocol)": int(g["disconnected_ods"].max()),
                "Disconnected ODs (failure protocol max)": int(failure_max.get(topo, 0)),
            }
        )
    save(pd.DataFrame(solver_rows), "solver_reliability_fix1_COMPLETED.csv")

    stability_rows = []
    for topo in M.TOP_ORDER + ["pooled"]:
        g = final_pc if topo == "pooled" else final_pc[final_pc["topology"] == topo]
        stability_rows.append(
            {
                "Topology": topo,
                "Mean changed OD pairs per TM": round(float(g["changed_od_pairs"].mean()), 6),
                "Median changed OD pairs": round(float(g["changed_od_pairs"].median()), 6),
                "Mean changed paths": round(float(g["changed_paths"].mean()), 6),
                "Median changed paths": round(float(g["changed_paths"].median()), 6),
                "Rule updates per cycle": round(float(g["rule_updates"].mean()), 6),
                "KEEP action percentage": round(float(g["keep_action"].mean() * 100.0), 3),
                "Percentage of cycles with no routing change": round(float(g["no_routing_change"].mean() * 100.0), 3),
                "Maximum routing changes in one TM": int(g["changed_paths"].max()),
            }
        )
    save(pd.DataFrame(stability_rows), "routing_stability_fix1_COMPLETED.csv")


def build_policy_and_baseline(
    final_pc: pd.DataFrame,
    no_gnn: pd.DataFrame,
    no_bottleneck: pd.DataFrame,
    ecmp_pc: pd.DataFrame,
    ospf_pc: pd.DataFrame,
) -> None:
    baseline = pd.read_csv(COMPLETED / "baseline_comparison_fix1_COMPLETED.csv")
    baseline = baseline[
        ~baseline["Method"].isin(
            [
                "ECMP",
                "OSPF-weighted shortest-path routing",
                "DDQN without GNN-LPD",
                "DDQN without bottleneck relief",
                "Final RG-GNN-LPD",
            ]
        )
    ].copy()
    for label, df in [
        ("ECMP", ecmp_pc),
        ("OSPF-weighted shortest-path routing", ospf_pc),
        ("DDQN without GNN-LPD", no_gnn),
        ("DDQN without bottleneck relief", no_bottleneck),
        ("Final RG-GNN-LPD", final_pc),
    ]:
        baseline = replace_or_append(baseline, summary_row(df, label))
    order = [
        "ECMP",
        "OSPF-weighted shortest-path routing",
        "Top-K Demand (K50)",
        "Bottleneck Top-K (K50)",
        "GNN-only fixed K50",
        "LPD-only fixed K50",
        "GNN+LPD fixed K50",
        "DDQN without GNN-LPD",
        "DDQN without bottleneck relief",
        "Final RG-GNN-LPD",
    ]
    baseline["Method"] = pd.Categorical(baseline["Method"], categories=order, ordered=True)
    baseline = baseline.sort_values("Method").reset_index(drop=True)
    save(baseline, "baseline_comparison_fix1_COMPLETED.csv")

    policy = pd.read_csv(COMPLETED / "policy_ablation_fix1_COMPLETED.csv")
    policy = policy[~policy["Method"].isin(["DDQN gate", "DDQN without bottleneck relief", "DDQN with teacher cycle-0"])].copy()
    final_row = summary_row(final_pc, "DDQN gate")
    teacher_row = summary_row(final_pc, "DDQN with teacher cycle-0")
    nob_row = summary_row(no_bottleneck, "DDQN without bottleneck relief")
    for row in [final_row, nob_row, teacher_row]:
        policy = replace_or_append(policy, row)
    order = ["Fixed K30", "Fixed K50", "Fixed K800", "DDQN gate", "DDQN without bottleneck relief", "DDQN without teacher cycle-0", "DDQN with teacher cycle-0"]
    policy["Method"] = pd.Categorical(policy["Method"], categories=order, ordered=True)
    policy = policy.sort_values("Method").reset_index(drop=True)
    save(policy, "policy_ablation_fix1_COMPLETED.csv")

    first_tm = pd.read_csv(COMPLETED / "first_tm_ablation_fix1_COMPLETED.csv")
    final_summary = M.summarize(final_pc)
    first_tm.loc[first_tm["Version"] == "FIX1 with teacher cycle-0", "Cycle-0 action"] = first_action_summary(final_pc)
    first_tm.loc[first_tm["Version"] == "FIX1 with teacher cycle-0", "Min PR"] = final_summary["Min PR"]
    first_tm.loc[first_tm["Version"] == "FIX1 with teacher cycle-0", "Mean PR"] = final_summary["Mean PR"]
    first_tm.loc[first_tm["Version"] == "FIX1 with teacher cycle-0", "Mean DB"] = final_summary["Mean DB"]
    first_tm.loc[first_tm["Version"] == "FIX1 with teacher cycle-0", "P95 ms"] = final_summary["P95 ms"]
    save(first_tm, "first_tm_ablation_fix1_COMPLETED.csv")


def sync_final_method_rows(final_pc: pd.DataFrame, ecmp_pc: pd.DataFrame) -> None:
    final_summary = summary_row(final_pc, "Final RG-GNN-LPD")
    by_topo = []
    for topo in M.TOP_ORDER:
        g = final_pc[final_pc["topology"] == topo]
        s = M.summarize(g)
        s["Method"] = "Final RG-GNN-LPD"
        s["Topology"] = topo
        by_topo.append(s)
    by_topo_df = pd.DataFrame(by_topo)

    scorer_real = pd.read_csv(COMPLETED / "scorer_ablation_real_200VTL_N3976.csv")
    scorer_real = replace_or_append(scorer_real, final_summary)
    scorer_order = [
        "GNN-only fixed K50",
        "LPD-only fixed K50",
        "GNN+LPD fixed K50",
        "GNN+LPD+bottleneck fixed K50",
        "Final RG-GNN-LPD",
    ]
    scorer_real["Method"] = pd.Categorical(scorer_real["Method"], categories=scorer_order, ordered=True)
    scorer_real = scorer_real.sort_values("Method").reset_index(drop=True)
    save(scorer_real, "scorer_ablation_real_200VTL_N3976.csv")

    scorer_pt = pd.read_csv(COMPLETED / "scorer_ablation_real_200VTL_N3976_per_topology.csv")
    scorer_pt = scorer_pt[scorer_pt["Method"] != "Final RG-GNN-LPD"].copy()
    scorer_pt = pd.concat([scorer_pt, by_topo_df[scorer_pt.columns]], ignore_index=True)
    scorer_pt["Method"] = pd.Categorical(scorer_pt["Method"], categories=scorer_order, ordered=True)
    scorer_pt = scorer_pt.sort_values(["Method", "Topology"]).reset_index(drop=True)
    save(scorer_pt, "scorer_ablation_real_200VTL_N3976_per_topology.csv")

    scorer_legacy = pd.read_csv(COMPLETED / "scorer_ablation_fix1_COMPLETED.csv")
    scorer_legacy = replace_or_append(scorer_legacy, final_summary)
    legacy_order = [
        "GNN-only fixed K50",
        "LPD-only fixed K50",
        "GNN+LPD fixed K50",
        "GNN+LPD+bottleneck fixed K50",
        "Final RG-GNN-LPD",
    ]
    scorer_legacy["Method"] = pd.Categorical(scorer_legacy["Method"], categories=legacy_order, ordered=True)
    scorer_legacy = scorer_legacy.sort_values("Method").reset_index(drop=True)
    save(scorer_legacy, "scorer_ablation_fix1_COMPLETED.csv")

    ks = pd.read_csv(COMPLETED / "k_sensitivity_fix1_COMPLETED.csv")
    adaptive = final_summary.copy()
    adaptive["Method / Budget"] = "Adaptive DDQN"
    adaptive.pop("Method", None)
    ks = ks[ks["Method / Budget"] != "Adaptive DDQN"].copy()
    ks = pd.concat([ks, pd.DataFrame([adaptive])[ks.columns]], ignore_index=True)
    order = ["K10", "K20", "K30", "K50", "K100", "Adaptive DDQN"]
    ks["Method / Budget"] = pd.Categorical(ks["Method / Budget"], categories=order, ordered=True)
    ks = ks.sort_values("Method / Budget").reset_index(drop=True)
    save(ks, "k_sensitivity_fix1_COMPLETED.csv")

    kpt = pd.read_csv(COMPLETED / "k_sensitivity_per_topology_fix1_200VTL_N3976_AUDITED.csv")
    for _, row in by_topo_df.iterrows():
        topo = row["Topology"]
        mask = kpt["Topology"].astype(str).str.lower() == topo
        if not mask.any():
            continue
        kpt.loc[mask, "Adaptive DDQN PR/DB"] = f"{float(row['Mean PR']):.3f}% / {float(row['Mean DB']):.3f}%"
        kpt.loc[mask, "Adaptive DDQN rows"] = int(row["N"])
        kpt.loc[mask, "Adaptive DDQN minPR"] = float(row["Min PR"])
        if "Audit note" in kpt.columns:
            kpt.loc[mask, "Audit note"] = (
                "Fixed-K values are directly computed from per-cycle PR/DB; "
                "Adaptive DDQN is aligned to Section 4 normal-table values."
            )
    save(kpt, "k_sensitivity_per_topology_fix1_200VTL_N3976_AUDITED.csv")

    opt = pd.read_csv(COMPLETED / "optimization_ablation_fix1_COMPLETED.csv")
    ecmp_row = summary_row(ecmp_pc, "ECMP only")
    ecmp_row["Purpose"] = "no optimization"
    ecmp_row["Notes"] = "ECMP"
    opt_row = final_summary.copy()
    opt_row["Method"] = "LP for selected ODs"
    opt_row["Purpose"] = "final efficient optimizer"
    opt_row["Notes"] = "Final RG-GNN-LPD"
    opt = opt[~opt["Method"].isin(["ECMP only", "LP for selected ODs"])].copy()
    opt = pd.concat([opt, pd.DataFrame([ecmp_row, opt_row])[opt.columns]], ignore_index=True)
    opt_order = ["ECMP only", "Ranking only, no LP", "LP for selected ODs", "Full-OD selected-flow LP reference"]
    opt["Method"] = pd.Categorical(opt["Method"], categories=opt_order, ordered=True)
    opt = opt.sort_values("Method").reset_index(drop=True)
    save(opt, "optimization_ablation_fix1_COMPLETED.csv")


def build_completion_summary() -> None:
    text = "\n".join(
        [
            "# COMPLETED_METRICS_SUMMARY",
            "",
            "- Final RG-GNN-LPD N=3976 detailed rerun: completed.",
            "- DDQN without GNN-LPD N=3976 rerun: completed.",
            "- DDQN without bottleneck relief N=3976 rerun: completed.",
            "- ECMP MLU rerun for Section 4A: completed.",
            "- OSPF-weighted shortest-path routing MLU: reused from executed N=3976 artifact using weight_mode=inverse_capacity only.",
            "- Solver reliability and routing stability exports: completed from the detailed final-method rerun.",
            "- SDN metrics: retained historical QoS evidence only; no fresh live Mininet/OVS rerun produced by this script.",
            "- Real live Mininet blocker on this host: macOS checkout with no `mn`, `ovs-vsctl`, `ryu-manager`, Linux VM manager, or local OVS datapath available.",
            "",
            "## Commands used",
            "",
            "- `python scripts/phase1_5/run_fix1_targeted_repairs.py`",
        ]
    )
    (COMPLETED / "COMPLETED_METRICS_SUMMARY.md").write_text(text)
    print(f"[write] {(COMPLETED / 'COMPLETED_METRICS_SUMMARY.md').relative_to(M.ROOT)}", flush=True)


def main() -> None:
    final_pc = M.eval_method(
        "Final RG-GNN-LPD",
        windows=M.WINDOWS,
        kind="adaptive",
        ranking_mode="cache_final",
        teacher_cycle0=True,
        notes="Targeted N=3976 rerun with detailed MLU / solver / routing-stability exports.",
    )
    save(final_pc, "final_rg_gnn_lpd_real_200vtl_n3976_per_cycle.csv")
    save(final_pc, "final_rg_gnn_lpd_200vtl_consistent_per_cycle.csv")

    no_gnn = M.eval_agnostic_ddqn_200vtl()
    save(no_gnn, "ddqn_without_gnn_lpd_per_cycle.csv")

    no_bottleneck = M.eval_no_bottleneck_relief_200vtl()
    save(no_bottleneck, "ddqn_without_bottleneck_relief_per_cycle.csv")

    ecmp_pc = M.eval_method(
        "ECMP",
        windows=M.WINDOWS,
        kind="ecmp",
        notes="Targeted N=3976 rerun for executed ECMP MLU export.",
    )
    save(ecmp_pc, "ecmp_per_cycle.csv")

    ospf_pc = pd.read_csv(COMPLETED / "ospf_weighted_shortest_path_baseline_N3976.csv")
    ospf_pc = ospf_pc[ospf_pc["weight_mode"] == "inverse_capacity"].copy()
    ospf_pc = pd.DataFrame(
        {
            "method": "OSPF-weighted shortest-path routing",
            "topology": ospf_pc["topology"].astype(str),
            "tm_index": ospf_pc["tm_index"].astype(int),
            "PR": ospf_pc["pr"].astype(float),
            "DB": ospf_pc["db"].astype(float),
            "decision_ms": ospf_pc["decision_ms"].astype(float),
            "achieved_mlu": ospf_pc["achieved_mlu"].astype(float),
        }
    )

    build_normal_tables(final_pc)
    failure_df = pd.read_csv(COMPLETED / "failure_scenarios_fix1_COMPLETED.csv") if (COMPLETED / "failure_scenarios_fix1_COMPLETED.csv").exists() else None
    build_mlu_solver_stability(final_pc, ecmp_pc, ospf_pc, failure_df=failure_df)
    build_policy_and_baseline(final_pc, no_gnn, no_bottleneck, ecmp_pc, ospf_pc)
    sync_final_method_rows(final_pc, ecmp_pc)
    build_completion_summary()


if __name__ == "__main__":
    main()
