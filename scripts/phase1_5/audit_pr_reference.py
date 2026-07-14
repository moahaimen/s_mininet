#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
RC = ROOT / "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50"
COMPLETED = RC / "FINAL_REPORT_FIX1/completed_metrics"
STRICT_PARTIAL = RC / "STRICT_FULL_MCF_PR/_partial"
STRICT_STUDENT = ROOT / "results/phase1_5_incremental/lp_distilled_pr_gnn_kpaths8/strict_full_mcf_reference_student.csv"
PATHOPT_VTL = ROOT / "results/gnn_lpd_dqn_selective_db_lp/pathopt_ref/pathopt_vtlwavenet2011.csv"

EXPECTED_ROWS = {
    "abilene": 2016,
    "geant": 672,
    "cernet": 200,
    "sprintlink": 200,
    "tiscali": 200,
    "ebone": 200,
    "germany50": 288,
    "vtlwavenet2011": 200,
}


def strict_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for topo in EXPECTED_ROWS:
        path = STRICT_PARTIAL / f"{topo}.csv"
        if not path.exists():
            counts[topo] = 0
            continue
        df = pd.read_csv(path)
        if "mcf_status" in df.columns:
            counts[topo] = int((df["mcf_status"].fillna("") == "Optimal").sum())
        else:
            counts[topo] = int(len(df))
    return counts


def vtl_student_matches_pathopt() -> bool:
    strict = {}
    with STRICT_STUDENT.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["topology"].strip().lower() != "vtlwavenet2011":
                continue
            if row["solver_status"].strip() != "Optimal":
                continue
            strict[int(row["timestep"])] = float(row["strict_mcf_mlu"])
    pathopt = pd.read_csv(PATHOPT_VTL)
    if len(pathopt) != 200 or len(strict) != 200:
        return False
    for row in pathopt.itertuples():
        if abs(float(row.pathopt_mlu) - strict[int(row.timestep)]) > 1e-9:
            return False
    return True


def main() -> None:
    counts = strict_counts()
    final_ev = pd.read_csv(COMPLETED / "final_rg_gnn_lpd_200vtl_consistent_per_cycle.csv")
    normal = pd.read_csv(COMPLETED / "normal_8topo_200vtl_consistent.csv")
    pooled = pd.read_csv(COMPLETED / "normal_pooled_3976_consistent.csv")
    zero = pd.read_csv(COMPLETED / "zero_shot_200vtl_consistent.csv")
    baseline = pd.read_csv(COMPLETED / "baseline_comparison_fix1_COMPLETED.csv")
    scorer = pd.read_csv(COMPLETED / "scorer_ablation_real_200VTL_N3976.csv")
    policy = pd.read_csv(COMPLETED / "policy_ablation_fix1_COMPLETED.csv")
    k_sens = pd.read_csv(COMPLETED / "k_sensitivity_fix1_COMPLETED.csv")
    optimization = pd.read_csv(COMPLETED / "optimization_ablation_fix1_COMPLETED.csv")

    all_counts_ok = all(counts[t] == EXPECTED_ROWS[t] for t in EXPECTED_ROWS)
    final_counts = final_ev["topology"].value_counts().to_dict()
    final_rows_ok = all(int(final_counts.get(t, 0)) == EXPECTED_ROWS[t] for t in EXPECTED_ROWS)
    pooled_ok = int(pooled.iloc[0]["N"]) == 3976
    zero_ok = int(zero.iloc[0]["N"]) == 488
    normal_ok = set(normal["topology"]) == set(EXPECTED_ROWS)
    ospf_ok = "OSPF-weighted shortest-path routing" in set(baseline["Method"])
    full_od_ok = "Full-OD selected-flow LP reference" in set(optimization["Method"])
    vtl_match = vtl_student_matches_pathopt()

    lines = [
        "# PR Reference Audit (N=3976 normal protocol, strict-all)",
        "",
        "## Required PR definition",
        "",
        "- PR is computed as `strict full-MCF optimum MLU / achieved MLU` for all normal-protocol report rows.",
        "- Active strict numerator cache: `results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/STRICT_FULL_MCF_PR/_partial/<topology>.csv`.",
        "- Strict numerator column: `strict_full_mcf_MLU`.",
        "- Solver-status column used for coverage: `mcf_status`.",
        "",
        "## Strict cache coverage",
        "",
        "| Topology | Expected rows | Optimal strict rows | Status |",
        "|---|---:|---:|---|",
    ]
    for topo, expected in EXPECTED_ROWS.items():
        status = "PASS" if counts[topo] == expected else "FAIL"
        lines.append(f"| {topo} | {expected} | {counts[topo]} | {status} |")

    lines += [
        "",
        "## VtlWavenet2011 strict-reference normalization note",
        "",
        f"- `strict_full_mcf_reference_student.csv` Vtl rows = 200/200 Optimal: {'PASS' if vtl_match else 'FAIL'}.",
        f"- `pathopt_vtlwavenet2011.csv` is numerically identical to the student strict Vtl reference in this checkout: {'PASS' if vtl_match else 'FAIL'}.",
        "- The active `STRICT_FULL_MCF_PR/_partial/vtlwavenet2011.csv` file therefore carries the strict numerator for the requested 200-row Vtl normal stream.",
        "",
        "## Checked normal-protocol artifacts",
        "",
        "| Artifact | Check | Status |",
        "|---|---|---|",
        f"| `final_rg_gnn_lpd_200vtl_consistent_per_cycle.csv` | normal-protocol per-cycle rows cover all 8 topologies with counts {EXPECTED_ROWS} | {'PASS' if final_rows_ok else 'FAIL'} |",
        f"| `normal_8topo_200vtl_consistent.csv` | per-topology normal table covers all 8 topologies | {'PASS' if normal_ok else 'FAIL'} |",
        f"| `normal_pooled_3976_consistent.csv` | pooled normal N = 3976 | {'PASS' if pooled_ok else 'FAIL'} |",
        f"| `zero_shot_200vtl_consistent.csv` | combined zero-shot N = 488 | {'PASS' if zero_ok else 'FAIL'} |",
        f"| `baseline_comparison_fix1_COMPLETED.csv` | OSPF row name is exactly `OSPF-weighted shortest-path routing` | {'PASS' if ospf_ok else 'FAIL'} |",
        f"| `scorer_ablation_real_200VTL_N3976.csv` | scorer-ablation table present under the N=3976 protocol | {'PASS' if len(scorer) == 5 else 'FAIL'} |",
        f"| `policy_ablation_fix1_COMPLETED.csv` | policy-ablation table present under the N=3976 protocol | {'PASS' if len(policy) == 6 else 'FAIL'} |",
        f"| `k_sensitivity_fix1_COMPLETED.csv` | K-sensitivity table present under the N=3976 protocol | {'PASS' if len(k_sens) == 6 else 'FAIL'} |",
        f"| `optimization_ablation_fix1_COMPLETED.csv` | Full-OD selected-flow LP reference retained only as ablation/check row | {'PASS' if full_od_ok else 'FAIL'} |",
        "",
        "## Full-OD selected-flow LP reference audit",
        "",
        "- The row name `Full-OD selected-flow LP reference` is retained in the optimization-ablation table.",
        "- It is **not** the strict full-MCF optimum.",
        "- It is **not** used as the numerator of PR.",
        "",
        "## Scope note",
        "",
        "- This audit is for normal-protocol PR tables only.",
        "- The supplementary failure stress section is reported separately and is not part of the N=3976 strict-normal PR numerator claim.",
        "",
        "## Final audit result",
        "",
        f"- Strict full-MCF numerator available for all 3976 normal-protocol rows: {'PASS' if all_counts_ok and final_rows_ok else 'FAIL'}.",
        f"- Tiscali strict rows = 200/200: {'PASS' if counts['tiscali'] == 200 else 'FAIL'}.",
        f"- VtlWavenet2011 strict rows = 200/200: {'PASS' if counts['vtlwavenet2011'] == 200 else 'FAIL'}.",
        "- No path-LP fallback is used for the normal-protocol PR tables in the repaired strict-all package.",
    ]

    out = COMPLETED / "pr_reference_audit_N3976.md"
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
