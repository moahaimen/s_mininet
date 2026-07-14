#!/usr/bin/env python
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


ROOT = Path("/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
OUT = ROOT / "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50"
FIX1 = OUT / "FULLDATA_GATED_PRESERVED_FIX1"
REPORT_OUT = OUT / "FINAL_REPORT_FIX1/missing_metrics"
SDN = ROOT / "results/gnn_lpd_dqn_selective_db_lp/sdn_mininet_clean"

REPORT_OUT.mkdir(parents=True, exist_ok=True)

CURRENT_NORMAL_COUNTS = {
    "abilene": 2016,
    "geant": 672,
    "cernet": 200,
    "sprintlink": 200,
    "tiscali": 200,
    "ebone": 200,
    "germany50": 288,
    "vtlwavenet2011": 40,
}

REQUESTED_PROTOCOL = [
    ("Abilene", "Real/SNDlib", "Seen", 2016, 2016, ""),
    ("GEANT", "Real/SNDlib", "Seen", 672, 672, ""),
    ("CERNET", "MGM", "Seen", 200, 200, ""),
    ("Sprintlink", "RocketFuel/MGM", "Seen", 200, 200, ""),
    ("Tiscali", "RocketFuel/MGM", "Seen", 200, 200, ""),
    ("Ebone", "RocketFuel/MGM", "Seen", 200, 200, ""),
    ("Germany50", "Real/SNDlib", "Zero-shot", 0, 288, ""),
    (
        "VtlWavenet2011",
        "Topology Zoo/MGM",
        "Zero-shot",
        0,
        200,
        "Current FIX1 normal headline artifact uses only 40 normal TMs; requested protocol is 200.",
    ),
]


def pct(x: Optional[float]) -> Optional[float]:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x) * 100.0, 3)


def ms(x: Optional[float]) -> Optional[float]:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), 3)


def line_no(path: Path, needle: str) -> Optional[int]:
    if not path.exists():
        return None
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        if needle in line:
            return i
    return None


def weighted_mean(series: Iterable[float], weights: Iterable[float]) -> float:
    series = np.array(list(series), dtype=float)
    weights = np.array(list(weights), dtype=float)
    return float(np.sum(series * weights) / np.sum(weights))


def summarize_pr_ms(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    return {
        "N": int(len(df)),
        "Mean PR": pct(df["PR"].mean()),
        "PR>=0.90": round(float((df["PR"] >= 0.90).mean() * 100.0), 3),
        "PR>=0.95": round(float((df["PR"] >= 0.95).mean() * 100.0), 3),
        "Mean ms": ms(df["decision_ms"].mean()),
        "P95 ms": ms(np.percentile(df["decision_ms"], 95)),
    }


def final_fix1_metrics() -> Dict[str, Optional[float]]:
    per_cycle = pd.read_csv(FIX1 / "fulldata_gated_eval_per_cycle.csv")
    agg = pd.read_csv(FIX1 / "normal_8topo_vs_target.csv")
    pr_ms = summarize_pr_ms(per_cycle)
    counts = agg["topology"].map(CURRENT_NORMAL_COUNTS).astype(int)
    mean_db = weighted_mean(agg["meanDB"], counts)
    return {
        **pr_ms,
        "Mean DB": pct(mean_db),
        "P95 DB": None,
    }


def top_rows_from_fix1() -> pd.DataFrame:
    df = pd.read_csv(FIX1 / "fulldata_gated_eval_per_cycle.csv")
    return df.sort_values(["topology", "tm_index"]).groupby("topology", as_index=False).first()


def top_rows_historical() -> pd.DataFrame:
    df = pd.read_csv(OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN/final_learned_4of5_iter2_eval_per_cycle.csv")
    return df.sort_values(["topology", "tm_index"]).groupby("topology", as_index=False).first()


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def main() -> None:
    final = final_fix1_metrics()
    normal_agg = pd.read_csv(FIX1 / "normal_8topo_vs_target.csv")
    periodic = pd.read_csv(OUT / "PERIODIC_FAILURE_FIX1/periodic_failure_summary.csv")
    sdn_summary = pd.read_csv(SDN / "sdn_summary.csv")
    sdn_per_run = pd.read_csv(SDN / "sdn_per_run.csv")

    # 1. Internal baselines under FIX1
    baseline_rows = [
        {
            "Method": "ECMP",
            "Status": "Missing",
            "Source artifact": "",
            "N": "",
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "No ECMP-only FIX1 rerun artifact exists. Historical non-FIX1 ECMP references exist in older condition-compliant comparisons only.",
        },
        {
            "Method": "OSPF / shortest path",
            "Status": "Missing",
            "Source artifact": "",
            "N": "",
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "Not rerun under FIX1 protocol.",
        },
        {
            "Method": "Top-K Demand",
            "Status": "Missing",
            "Source artifact": "",
            "N": "",
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "Not rerun under FIX1 protocol.",
        },
        {
            "Method": "Bottleneck Top-K",
            "Status": "Missing",
            "Source artifact": "",
            "N": "",
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "Not rerun under FIX1 protocol as a standalone baseline.",
        },
        {
            "Method": "GNN-only fixed K",
            "Status": "Missing",
            "Source artifact": OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN/rank_ablation.csv",
            "N": "",
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "Only historical Sprintlink-only rank-ablation rows exist; no FIX1 all-topology rerun.",
        },
        {
            "Method": "LPD-only fixed K",
            "Status": "Missing",
            "Source artifact": OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN/rank_ablation.csv",
            "N": "",
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "Only historical Sprintlink-only rank-ablation rows exist; no FIX1 all-topology rerun.",
        },
        {
            "Method": "GNN+LPD fixed K",
            "Status": "Missing",
            "Source artifact": "",
            "N": "",
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "No FIX1 fixed-K global rerun artifact exists.",
        },
        {
            "Method": "DDQN without GNN-LPD",
            "Status": "Missing",
            "Source artifact": OUT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN/agnostic_ddqn_summary.csv",
            "N": "",
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "Historical topology-agnostic DDQN artifacts exist, but not rerun under FIX1 protocol.",
        },
        {
            "Method": "DDQN without bottleneck relief",
            "Status": "Missing",
            "Source artifact": OUT / "BOTTLENECK_AWARE_DDQN/nobottleneck_ddqn_counters.json",
            "N": "",
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "A nobottleneck training artifact exists, but no FIX1 summary/eval rerun was emitted.",
        },
        {
            "Method": "Final RG-GNN-LPD",
            "Status": "Complete",
            "Source artifact": FIX1 / "fulldata_gated_eval_per_cycle.csv",
            "N": final["N"],
            "Mean PR": final["Mean PR"],
            "PR>=0.90": final["PR>=0.90"],
            "PR>=0.95": final["PR>=0.95"],
            "Mean DB": final["Mean DB"],
            "P95 DB": "",
            "Mean ms": final["Mean ms"],
            "P95 ms": final["P95 ms"],
            "Notes": "Mean DB is from weighted per-topology FIX1 aggregate DB. Overall P95 DB is not available because realized per-cycle DB is not logged in the FIX1 eval CSV.",
        },
    ]
    write_csv(
        REPORT_OUT / "baseline_comparison_fix1.csv",
        baseline_rows,
        [
            "Method",
            "Status",
            "Source artifact",
            "N",
            "Mean PR",
            "PR>=0.90",
            "PR>=0.95",
            "Mean DB",
            "P95 DB",
            "Mean ms",
            "P95 ms",
            "Notes",
        ],
    )

    # 2. Scorer ablation
    scorer_rows = [
        {
            "Scorer": "GNN only",
            "Status": "Missing",
            "Source artifact": OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN/rank_ablation.csv",
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "Only historical Sprintlink-only rows exist; not a FIX1 all-topology rerun.",
        },
        {
            "Scorer": "LPD only",
            "Status": "Missing",
            "Source artifact": OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN/rank_ablation.csv",
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "Only historical Sprintlink-only rows exist; not a FIX1 all-topology rerun.",
        },
        {
            "Scorer": "Bottleneck relief only",
            "Status": "Missing",
            "Source artifact": OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN/rank_ablation.csv",
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "Historical relief_only row exists only for Sprintlink/K500 and is not FIX1-compatible.",
        },
        {
            "Scorer": "GNN + LPD",
            "Status": "Missing",
            "Source artifact": "",
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "No FIX1 ablation isolates GNN+LPD without bottleneck ranking.",
        },
        {
            "Scorer": "GNN + LPD + bottleneck",
            "Status": "Complete",
            "Source artifact": FIX1 / "fulldata_gated_eval_per_cycle.csv",
            "Mean PR": final["Mean PR"],
            "PR>=0.90": final["PR>=0.90"],
            "PR>=0.95": final["PR>=0.95"],
            "Mean DB": final["Mean DB"],
            "P95 DB": "",
            "Mean ms": final["Mean ms"],
            "P95 ms": final["P95 ms"],
            "Notes": "Current FIX1 final method.",
        },
    ]
    write_csv(
        REPORT_OUT / "scorer_ablation_fix1.csv",
        scorer_rows,
        ["Scorer", "Status", "Source artifact", "Mean PR", "PR>=0.90", "PR>=0.95", "Mean DB", "P95 DB", "Mean ms", "P95 ms", "Notes"],
    )

    # 3. Policy ablation
    policy_rows = [
        {
            "Policy": "Fixed K30",
            "Status": "Missing",
            "Source artifact": "",
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "No FIX1 fixed-K30 rerun artifact exists.",
        },
        {
            "Policy": "Fixed K50",
            "Status": "Missing",
            "Source artifact": OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN/vtl_extended_200_per_cycle.csv",
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "Only historical VtlWavenet K50 extension exists; no FIX1 all-topology fixed-K50 rerun.",
        },
        {
            "Policy": "Fixed K800",
            "Status": "Missing",
            "Source artifact": OUT / "SPRINTLINK_4OF5_SEARCH/SPRINTLINK_4OF5_SEARCH_TABLE.csv",
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "Only deployable Sprintlink search artifacts exist outside the FIX1 learned comparison.",
        },
        {
            "Policy": "DDQN gate",
            "Status": "Complete",
            "Source artifact": FIX1 / "fulldata_gated_eval_per_cycle.csv",
            "Mean PR": final["Mean PR"],
            "PR>=0.90": final["PR>=0.90"],
            "PR>=0.95": final["PR>=0.95"],
            "Mean DB": final["Mean DB"],
            "P95 DB": "",
            "Mean ms": final["Mean ms"],
            "P95 ms": final["P95 ms"],
            "Notes": "Current FIX1 final policy.",
        },
        {
            "Policy": "DDQN without teacher cycle-0",
            "Status": "Missing",
            "Source artifact": OUT / "FROZEN_FIRST_CYCLE_OPT_ABLATION/comparison_summary.csv",
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "Only historical frozen Tier-A first-cycle ablation exists; not rerun on the FIX1 model.",
        },
        {
            "Policy": "DDQN with teacher cycle-0",
            "Status": "Complete",
            "Source artifact": FIX1 / "fulldata_gated_eval_per_cycle.csv",
            "Mean PR": final["Mean PR"],
            "PR>=0.90": final["PR>=0.90"],
            "PR>=0.95": final["PR>=0.95"],
            "Mean DB": final["Mean DB"],
            "P95 DB": "",
            "Mean ms": final["Mean ms"],
            "P95 ms": final["P95 ms"],
            "Notes": "Current FIX1 headline artifact uses frozen-teacher action preservation at cycle 0. Current code now also masks any cycle-0 KEEP, but the published FIX1 CSV predates that guard rerun.",
        },
    ]
    write_csv(
        REPORT_OUT / "policy_ablation_fix1.csv",
        policy_rows,
        ["Policy", "Status", "Source artifact", "Mean PR", "PR>=0.90", "PR>=0.95", "Mean DB", "P95 DB", "Mean ms", "P95 ms", "Notes"],
    )

    # 4. Optimization ablation
    opt_rows = [
        {
            "Method": "ECMP only",
            "Purpose": "no optimization",
            "Status": "Missing",
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "No ECMP-only FIX1 PR/DB rerun artifact exists.",
        },
        {
            "Method": "Ranking only, no LP",
            "Purpose": "ranking evidence but no feasible split optimization",
            "Status": "Missing",
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "No FIX1 ranking-only no-LP artifact exists.",
        },
        {
            "Method": "LP for selected ODs",
            "Purpose": "final efficient optimizer",
            "Status": "Complete",
            "Mean PR": final["Mean PR"],
            "PR>=0.90": final["PR>=0.90"],
            "PR>=0.95": final["PR>=0.95"],
            "Mean DB": final["Mean DB"],
            "P95 DB": "",
            "Mean ms": final["Mean ms"],
            "P95 ms": final["P95 ms"],
            "Notes": "Current FIX1 final method.",
        },
        {
            "Method": "Full-OD LP",
            "Purpose": "optimal reference but slower",
            "Status": "Partial",
            "Mean PR": 100.0,
            "PR>=0.90": 100.0,
            "PR>=0.95": 100.0,
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "PR is 100 by definition against its own strict full-MCF reference. FIX1 runtime and disturbance for a full-OD LP baseline were not logged as a comparable artifact.",
        },
    ]
    write_csv(
        REPORT_OUT / "optimization_ablation_fix1.csv",
        opt_rows,
        ["Method", "Purpose", "Status", "Mean PR", "PR>=0.90", "PR>=0.95", "Mean DB", "P95 DB", "Mean ms", "P95 ms", "Notes"],
    )

    # 5. First-TM ablation and cycle-0 keep audit
    topfix1 = top_rows_from_fix1()
    first_tm_rows = [
        {
            "Version": "FIX1 with teacher cycle-0",
            "Status": "Partial",
            "Cycle-0 action": "Per-topology; see cycle0_keep_guard_audit.md",
            "Cycle-0 PR": pct(topfix1["PR"].mean()),
            "Min PR": round(float(normal_agg["minPR"].min() * 100.0), 3),
            "Mean PR": final["Mean PR"],
            "Mean DB": final["Mean DB"],
            "P95 ms": final["P95 ms"],
            "Notes": "Published FIX1 artifact preserves frozen-teacher cycle-0 action, but predates the later cycle-0 KEEP guard patch.",
        },
        {
            "Version": "FIX1 without teacher cycle-0",
            "Status": "Missing",
            "Cycle-0 action": "",
            "Cycle-0 PR": "",
            "Min PR": "",
            "Mean PR": "",
            "Mean DB": "",
            "P95 ms": "",
            "Notes": "Only historical frozen Tier-A first-cycle-opt ablation exists; no current FIX1 rerun without teacher cycle-0.",
        },
    ]
    write_csv(
        REPORT_OUT / "first_tm_ablation_fix1.csv",
        first_tm_rows,
        ["Version", "Status", "Cycle-0 action", "Cycle-0 PR", "Min PR", "Mean PR", "Mean DB", "P95 ms", "Notes"],
    )

    hist_top = top_rows_historical()
    ebone_hist = hist_top[hist_top["topology"] == "ebone"].iloc[0]
    code_main = ROOT / "scripts/phase1_5/run_fulldata_gated_fix1.py"
    code_fail = ROOT / "scripts/phase1_5/run_periodic_failure_fix1.py"
    audit_lines = [
        "# Cycle-0 KEEP guard audit for FIX1",
        "",
        "## Code fact",
        "",
        f"- `{code_main}` line {line_no(code_main, 'if i == 0 and a == KEEP_IDX')}: cycle-0 KEEP is now explicitly masked in normal FIX1 eval code.",
        f"- `{code_fail}` line {line_no(code_fail, 'if i == 0 and a == KEEP_IDX')}: cycle-0 KEEP is now explicitly masked in periodic-failure FIX1 eval code.",
        "",
        "## Artifact fact",
        "",
        f"- The currently published historical pre-guard artifact `FINAL_LEARNED_4OF5_ITER2_DDQN/final_learned_4of5_iter2_eval_per_cycle.csv` shows `ebone` TM0 action `{ebone_hist['action']}` with PR `{ebone_hist['PR']:.6f}`.",
        "- The current published FIX1 artifact in `FULLDATA_GATED_PRESERVED_FIX1/fulldata_gated_eval_per_cycle.csv` predates a full rerun after the guard patch, so measured FIX1 tables do not yet reflect the new guard end-to-end.",
        "",
        "## Conclusion",
        "",
        "- Current code: cycle-0 KEEP is blocked.",
        "- Current published FIX1 metrics: stale relative to that guard; a fresh eval rerun is still required before the report can claim guarded cycle-0 behavior as measured output.",
    ]
    (REPORT_OUT / "cycle0_keep_guard_audit.md").write_text("\n".join(audit_lines))

    # 6. K sensitivity
    ks_rows = []
    for label in ["K10", "K20", "K30", "K50", "K100"]:
        ks_rows.append(
            {
                "Method / Budget": label,
                "Status": "Missing",
                "N": "",
                "Mean PR": "",
                "PR>=0.90": "",
                "PR>=0.95": "",
                "Mean DB": "",
                "P95 DB": "",
                "Mean ms": "",
                "P95 ms": "",
                "Notes": "No FIX1 fixed-K rerun artifact exists for this budget.",
            }
        )
    ks_rows.append(
        {
            "Method / Budget": "Adaptive DDQN",
            "Status": "Complete",
            "N": final["N"],
            "Mean PR": final["Mean PR"],
            "PR>=0.90": final["PR>=0.90"],
            "PR>=0.95": final["PR>=0.95"],
            "Mean DB": final["Mean DB"],
            "P95 DB": "",
            "Mean ms": final["Mean ms"],
            "P95 ms": final["P95 ms"],
            "Notes": "Current FIX1 final method. Fixed-K sweep was not rerun under the same protocol.",
        }
    )
    write_csv(
        REPORT_OUT / "k_sensitivity_fix1.csv",
        ks_rows,
        ["Method / Budget", "Status", "N", "Mean PR", "PR>=0.90", "PR>=0.95", "Mean DB", "P95 DB", "Mean ms", "P95 ms", "Notes"],
    )

    topo_rows = []
    for row in normal_agg.itertuples():
        topo_rows.append(
            {
                "Topology": row.topology,
                "K10 PR/DB": "",
                "K20 PR/DB": "",
                "K30 PR/DB": "",
                "K50 PR/DB": "",
                "K100 PR/DB": "",
                "Adaptive DDQN PR/DB": f"{pct(row.fulldata_meanPR):.3f}% / {pct(row.meanDB):.3f}%",
                "Best tradeoff": "Adaptive DDQN only available FIX1 row",
                "Notes": "Fixed-K sensitivity was not rerun under FIX1 for this topology.",
            }
        )
    write_csv(
        REPORT_OUT / "k_sensitivity_per_topology_fix1.csv",
        topo_rows,
        ["Topology", "K10 PR/DB", "K20 PR/DB", "K30 PR/DB", "K50 PR/DB", "K100 PR/DB", "Adaptive DDQN PR/DB", "Best tradeoff", "Notes"],
    )

    # 7. Failure scenarios under FIX1
    requested_scenarios = [
        "single_link_failure",
        "two_link_failure",
        "three_link_failure",
        "capacity_degradation_50",
        "spike",
        "mixed_spike_failure",
    ]
    topo_protocol = {
        "abilene": (2016, "single-link", "every 12 TMs", "full test stream"),
        "geant": (672, "single-link", "every 4 TMs", "full test stream"),
        "cernet": (200, "single-link", "every 2 TMs", "synthetic/MGM"),
        "sprintlink": (200, "single-link", "every 2 TMs", "synthetic/MGM"),
        "tiscali": (200, "single-link", "every 2 TMs", "synthetic/MGM"),
        "ebone": (200, "single-link", "every 2 TMs", "synthetic/MGM"),
        "germany50": (288, "single-link", "every 4 TMs", "zero-shot"),
        "vtlwavenet2011": (200, "single-link", "every 2 TMs", "zero-shot"),
    }
    failure_rows = []
    for topo in topo_protocol:
        ps = periodic[periodic["topology"] == topo]
        single = ps.iloc[0]
        failure_rows.append(
            {
                "Scenario": "single_link_failure",
                "Topology": topo,
                "Status": "Complete",
                "Mean PR": pct(single["mean_PR"]),
                "PR>=0.90": round(float(single["pr90"]), 3),
                "PR(fail)": pct(single["mean_PR_failcycles"]),
                "Mean DB": pct(single["mean_DB"]),
                "P95 DB if available": "",
                "Max disconnected ODs": int(single["max_disc"]),
                "Mean ms": ms(single["mean_ms"]),
                "P95 ms": ms(single["p95_ms"]),
                "Notes": "From PERIODIC_FAILURE_FIX1 full-stream single-link stress.",
            }
        )
        for sc in requested_scenarios[1:]:
            failure_rows.append(
                {
                    "Scenario": sc,
                    "Topology": topo,
                    "Status": "Missing",
                    "Mean PR": "",
                    "PR>=0.90": "",
                    "PR(fail)": "",
                    "Mean DB": "",
                    "P95 DB if available": "",
                    "Max disconnected ODs": "",
                    "Mean ms": "",
                    "P95 ms": "",
                    "Notes": "No FIX1 LP-eval artifact. Historical scenario-subset stress exists only in FAILURE_VALIDATION_ITER2_ALL8 and was not rerun on FIX1.",
                }
            )
    write_csv(
        REPORT_OUT / "failure_scenarios_fix1.csv",
        failure_rows,
        ["Scenario", "Topology", "Status", "Mean PR", "PR>=0.90", "PR(fail)", "Mean DB", "P95 DB if available", "Max disconnected ODs", "Mean ms", "P95 ms", "Notes"],
    )

    protocol_rows = []
    for topo, vals in topo_protocol.items():
        protocol_rows.append(
            {
                "Topology": topo,
                "Failure TMs": vals[0],
                "Failure type": vals[1],
                "Failure frequency": vals[2],
                "Notes": vals[3],
            }
        )
    write_csv(
        REPORT_OUT / "failure_protocol_fix1.csv",
        protocol_rows,
        ["Topology", "Failure TMs", "Failure type", "Failure frequency", "Notes"],
    )
    (REPORT_OUT / "failure_scenarios_missing.md").write_text(
        "\n".join(
            [
                "# Missing FIX1 failure scenarios",
                "",
                "- Full-stream FIX1 single-link failure exists in `PERIODIC_FAILURE_FIX1/`.",
                "- The requested additional scenarios `two_link_failure`, `three_link_failure`, `capacity_degradation_50`, `spike`, and `mixed_spike_failure` were not rerun on the FIX1 model in the LP-eval harness.",
                "- Historical subset scenario stress exists in `FAILURE_VALIDATION_ITER2_ALL8/failure_all8_summary.csv`, but it is not a FIX1 rerun and must not be presented as FIX1 final metrics.",
            ]
        )
    )

    # 8. VtlWavenet normal 200
    vtl_rows = [
        {
            "Topology": "vtlwavenet2011",
            "Status": "Missing",
            "N": 200,
            "Mean PR": "",
            "PR>=0.90": "",
            "PR>=0.95": "",
            "Mean DB": "",
            "P95 DB": "",
            "Mean ms": "",
            "P95 ms": "",
            "Notes": "Requested protocol is 200 normal TMs, but current FIX1 headline artifact evaluated only 40 normal TMs. Historical 200-TM file exists only for iter2 frozen model: FINAL_LEARNED_4OF5_ITER2_DDQN/vtl_extended_200_per_cycle.csv.",
        }
    ]
    write_csv(
        REPORT_OUT / "vtlwavenet2011_normal_200_fix1.csv",
        vtl_rows,
        ["Topology", "Status", "N", "Mean PR", "PR>=0.90", "PR>=0.95", "Mean DB", "P95 DB", "Mean ms", "P95 ms", "Notes"],
    )

    # 9. SDN / Mininet retained metrics
    sdn_rows = []
    required_scenarios = [
        "normal",
        "single_link_failure",
        "two_link_failure",
        "three_link_failure",
        "capacity_degradation_50",
        "spike",
        "mixed_spike_failure",
    ]
    for topo in ["abilene", "geant"]:
        subset = sdn_summary[sdn_summary["topology"] == topo]
        for sc in required_scenarios:
            row = subset[subset["scenario"] == sc]
            if row.empty:
                sdn_rows.append(
                    {
                        "Topology": topo,
                        "Scenario": sc,
                        "Throughput Mbps": "",
                        "RTT ms": "",
                        "Jitter ms": "",
                        "Packet loss %": "",
                        "Rule count": "",
                        "Install ms": "",
                        "Recovery ms": "",
                        "Disconnected ODs": "",
                        "Mode": "",
                        "Rerun on FIX1? yes/no": "no",
                        "Notes": "Scenario not present in retained SDN simulation artifact.",
                    }
                )
                continue
            r = row.iloc[0]
            mode = ",".join(sorted(set(sdn_per_run[(sdn_per_run["topology"] == topo) & (sdn_per_run["scenario"] == sc)]["mode"].astype(str))))
            sdn_rows.append(
                {
                    "Topology": topo,
                    "Scenario": sc,
                    "Throughput Mbps": ms(r["mean_throughput_mbps"]),
                    "RTT ms": ms(r["mean_rtt_ms"]),
                    "Jitter ms": ms(r["mean_jitter_ms"]),
                    "Packet loss %": ms(r["mean_packet_loss_pct"]),
                    "Rule count": ms(r["mean_flow_rules"]),
                    "Install ms": ms(r["mean_install_ms"]),
                    "Recovery ms": ms(r["mean_recovery_ms"]),
                    "Disconnected ODs": int(r["disconnected_ODs"]),
                    "Mode": mode or "simulate",
                    "Rerun on FIX1? yes/no": "no",
                    "Notes": "Retained SDN simulation artifact only; not rerun on FIX1.",
                }
            )
    write_csv(
        REPORT_OUT / "sdn_operational_metrics.csv",
        sdn_rows,
        ["Topology", "Scenario", "Throughput Mbps", "RTT ms", "Jitter ms", "Packet loss %", "Rule count", "Install ms", "Recovery ms", "Disconnected ODs", "Mode", "Rerun on FIX1? yes/no", "Notes"],
    )

    # 10. Dataset protocol tables
    dataset_rows = [
        {
            "Topology": topo,
            "Source": source,
            "Role": role,
            "Train TMs": train,
            "Test TMs": test,
            "Notes": note,
        }
        for topo, source, role, train, test, note in REQUESTED_PROTOCOL
    ]
    write_csv(
        REPORT_OUT / "dataset_protocol_table.csv",
        dataset_rows,
        ["Topology", "Source", "Role", "Train TMs", "Test TMs", "Notes"],
    )

    # 11. Final status
    status_md = [
        "# MISSING_METRICS_COMPLETED",
        "",
        "## Checklist",
        "",
        "- Internal baselines: not completed",
        "- Scorer ablation: not completed",
        "- Policy ablation: not completed",
        "- Optimization ablation: not completed",
        "- First-TM ablation: not completed",
        "- K sensitivity: not completed",
        "- Failure scenarios: not completed",
        "- VtlWavenet 200 normal eval: not completed",
        "- SDN metrics: retained only",
        "- Dataset protocol table: completed",
        "",
        "## Files created",
        "",
        "- baseline_comparison_fix1.csv",
        "- scorer_ablation_fix1.csv",
        "- policy_ablation_fix1.csv",
        "- optimization_ablation_fix1.csv",
        "- first_tm_ablation_fix1.csv",
        "- cycle0_keep_guard_audit.md",
        "- k_sensitivity_fix1.csv",
        "- k_sensitivity_per_topology_fix1.csv",
        "- failure_scenarios_fix1.csv",
        "- failure_protocol_fix1.csv",
        "- failure_scenarios_missing.md",
        "- vtlwavenet2011_normal_200_fix1.csv",
        "- sdn_operational_metrics.csv",
        "- dataset_protocol_table.csv",
        "",
        "## Key facts",
        "",
        "- Current FIX1 final metrics are sourced from `FULLDATA_GATED_PRESERVED_FIX1/`.",
        "- The current FIX1 eval CSV does not safely expose realized per-cycle disturbance for global P95 DB calculations; aggregate mean DB comes from `normal_8topo_vs_target.csv`.",
        "- Historical non-FIX1 artifacts exist for several ablations, but they were not copied into FIX1 tables as if they were FIX1 reruns.",
        "- Periodic single-link failure is the only full-stream failure artifact emitted specifically for FIX1.",
        "- Additional multi-failure/spike/capacity scenarios exist only as historical subset stress artifacts or retained SDN simulations.",
        "- Current code now includes a cycle-0 KEEP guard, but the published FIX1 metrics predate a full rerun after that patch.",
    ]
    (REPORT_OUT / "MISSING_METRICS_COMPLETED.md").write_text("\n".join(status_md))


if __name__ == "__main__":
    main()
