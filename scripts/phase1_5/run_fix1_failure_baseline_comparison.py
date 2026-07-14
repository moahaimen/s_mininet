#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path("/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
sys.path.insert(0, str(ROOT))

import scripts.phase1_5.run_fix1_completed_metrics as M  # noqa: E402
from scripts.phase1_5.run_final_iter2 import bottleneck_rank  # noqa: E402
from te.baselines import clone_splits, ecmp_splits  # noqa: E402
from te.disturbance import compute_disturbance  # noqa: E402
from te.lp_solver import solve_all_od_path_lp  # noqa: E402
from te.paths import PathLibrary  # noqa: E402


OUT = ROOT / "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50"
COMPLETED = OUT / "FINAL_REPORT_FIX1/completed_metrics"
LOGS = COMPLETED / "logs"
COMPLETED.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    "single_link_failure",
    "two_link_failure",
    "three_link_failure",
    "capacity_degradation_50",
    "spike",
    "mixed_spike_failure",
]

TOPOS = {
    "abilene": (2016, 2036),
    "geant": (672, 692),
    "cernet": (200, 220),
    "sprintlink": (200, 220),
    "tiscali": (200, 220),
    "ebone": (200, 220),
    "germany50": (0, 20),
    "vtlwavenet2011": (0, 20),
}

DISCONNECTED_RULE = "count_disconnected_ods(path-sum<=1e-9 or zero-path OD)"
FAILURE_SEED = 0
REF_WEIGHT_MODE = "unit"

OUT_PER_CYCLE = COMPLETED / "failure_baseline_comparison_fix1_per_cycle.csv"
OUT_SUMMARY = COMPLETED / "failure_baseline_comparison_fix1_COMPLETED.csv"
OUT_EVENTS = COMPLETED / "failure_event_catalog_fix1_COMPLETED.csv"
OUT_VALIDATION = COMPLETED / "failure_event_validation_fix1.md"


@dataclass
class BaseCtx:
    topo: str
    lo: int
    hi: int
    ds: object
    pl0: PathLibrary
    caps0: np.ndarray
    struct: dict


def pr_of(opt_mlu: float, achieved_mlu: float) -> float:
    return float(min(1.0, opt_mlu / achieved_mlu)) if achieved_mlu > 0 else 0.0


def pct(x: float) -> float:
    return round(float(x) * 100.0, 3)


def ms(x: float) -> float:
    return round(float(x), 3)


def scenario_label(name: str) -> str:
    return {
        "single_link_failure": "single-link failure",
        "two_link_failure": "two-link failure",
        "three_link_failure": "three-link failure",
        "capacity_degradation_50": "capacity-50%",
        "spike": "spike",
        "mixed_spike_failure": "mixed spike+failure",
    }[name]


def pick_failed_links(caps: np.ndarray, n: int) -> np.ndarray:
    return np.argsort(caps)[::-1][: min(n, int(np.sum(caps > 0)))]


def failed_edge_ids(caps: np.ndarray, scenario: str) -> list[int]:
    n = {
        "single_link_failure": 1,
        "two_link_failure": 2,
        "three_link_failure": 3,
        "mixed_spike_failure": 1,
    }.get(scenario, 0)
    if n <= 0:
        return []
    return [int(x) for x in pick_failed_links(caps, n).tolist()]


def capacity_scale_for(scenario: str) -> float:
    return 0.5 if scenario == "capacity_degradation_50" else 1.0


def spike_factor_for(scenario: str) -> float:
    return 3.0 if scenario in ("spike", "mixed_spike_failure") else 1.0


def n_failed_links(scenario: str) -> int:
    return {
        "single_link_failure": 1,
        "two_link_failure": 2,
        "three_link_failure": 3,
        "mixed_spike_failure": 1,
    }.get(scenario, 0)


def modify_caps(caps: np.ndarray, scenario: str) -> np.ndarray:
    out = caps.copy()
    if scenario == "single_link_failure":
        out[pick_failed_links(caps, 1)] = 0.0
    elif scenario == "two_link_failure":
        out[pick_failed_links(caps, 2)] = 0.0
    elif scenario == "three_link_failure":
        out[pick_failed_links(caps, 3)] = 0.0
    elif scenario == "capacity_degradation_50":
        out *= 0.5
    elif scenario == "mixed_spike_failure":
        out[pick_failed_links(caps, 1)] = 0.0
    return out


def prune_pathlib(pl: PathLibrary, caps: np.ndarray) -> PathLibrary:
    node_paths, edge_paths, edge_idx_paths, costs = [], [], [], []
    for i in range(len(pl.edge_idx_paths_by_od)):
        keep = [j for j, ep in enumerate(pl.edge_idx_paths_by_od[i]) if all(float(caps[e]) > 0 for e in ep)]
        node_paths.append([pl.node_paths_by_od[i][j] for j in keep])
        edge_paths.append([pl.edge_paths_by_od[i][j] for j in keep])
        edge_idx_paths.append([pl.edge_idx_paths_by_od[i][j] for j in keep])
        costs.append([pl.costs_by_od[i][j] for j in keep])
    return PathLibrary(
        od_pairs=pl.od_pairs,
        node_paths_by_od=node_paths,
        edge_paths_by_od=edge_paths,
        edge_idx_paths_by_od=edge_idx_paths,
        costs_by_od=costs,
    )


def build_failure_ospf_path_library(ds, caps: np.ndarray, weight_mode: str = REF_WEIGHT_MODE) -> PathLibrary:
    graph = nx.DiGraph()
    for node in ds.nodes:
        graph.add_node(node)

    edge_to_idx = {edge: idx for idx, edge in enumerate(ds.edges)}
    for edge_idx, (src, dst) in enumerate(ds.edges):
        cap = float(caps[edge_idx])
        if cap <= 0:
            continue
        if weight_mode == "unit":
            weight = 1.0
        elif weight_mode == "inverse_capacity":
            weight = float(100.0 / max(cap, 1e-12))
        else:
            raise ValueError(weight_mode)
        graph.add_edge(src, dst, weight=weight, capacity=cap)

    node_paths_by_od = []
    edge_paths_by_od = []
    edge_idx_paths_by_od = []
    costs_by_od = []
    for src, dst in ds.od_pairs:
        if src == dst or not nx.has_path(graph, src, dst):
            node_paths_by_od.append([])
            edge_paths_by_od.append([])
            edge_idx_paths_by_od.append([])
            costs_by_od.append([])
            continue
        paths = [list(path) for path in nx.all_shortest_paths(graph, src, dst, weight="weight")]
        edge_paths = [[(path[i], path[i + 1]) for i in range(len(path) - 1)] for path in paths]
        edge_idx_paths = [[edge_to_idx[e] for e in ep] for ep in edge_paths]
        costs = []
        for path in paths:
            total = 0.0
            for i in range(len(path) - 1):
                total += float(graph[path[i]][path[i + 1]]["weight"])
            costs.append(total)
        node_paths_by_od.append(paths)
        edge_paths_by_od.append(edge_paths)
        edge_idx_paths_by_od.append(edge_idx_paths)
        costs_by_od.append(costs)
    return PathLibrary(
        od_pairs=list(ds.od_pairs),
        node_paths_by_od=node_paths_by_od,
        edge_paths_by_od=edge_paths_by_od,
        edge_idx_paths_by_od=edge_idx_paths_by_od,
        costs_by_od=costs_by_od,
    )


def equal_shortest_splits(pl: PathLibrary) -> list[np.ndarray]:
    splits: list[np.ndarray] = []
    for costs in pl.costs_by_od:
        if not costs:
            splits.append(np.zeros(0, dtype=float))
            continue
        arr = np.asarray(costs, dtype=float)
        min_cost = float(arr.min())
        idx = np.where(np.abs(arr - min_cost) <= 1e-12)[0]
        vec = np.zeros(len(arr), dtype=float)
        vec[idx] = 1.0 / float(len(idx))
        splits.append(vec)
    return splits


def make_base_ctx(topo: str, lo: int, hi: int) -> BaseCtx:
    env = M._make_envs([topo], {topo: (lo, hi)}, M.GNNLPD, hi - lo, 30)[0]
    return BaseCtx(
        topo=topo,
        lo=lo,
        hi=hi,
        ds=env.ctx["ds"],
        pl0=env.ctx["pl"],
        caps0=np.asarray(env.ctx["caps"], float),
        struct=M.A.struct_feats(env.ctx["ds"]),
    )


def make_failure_ctx(base: BaseCtx, pl: PathLibrary, caps: np.ndarray):
    return SimpleNamespace(
        topo=base.topo,
        ds=base.ds,
        pl=pl,
        caps=np.asarray(caps, float),
        ecmp=ecmp_splits(pl),
        struct=base.struct,
        prepass=None,
        raw_cache=None,
        rank_cache=None,
    )


def summarize(df: pd.DataFrame) -> dict:
    disc_col = "disconnected_ods" if "disconnected_ods" in df.columns else "Disconnected ODs"
    return {
        "Mean PR": pct(df["PR"].mean()),
        "PR>=0.90": round(float((df["PR"] >= 0.90).mean() * 100.0), 3),
        "PR>=0.95": round(float((df["PR"] >= 0.95).mean() * 100.0), 3),
        "Mean DB": pct(df["DB"].mean()),
        "P95 DB": pct(np.percentile(df["DB"], 95)),
        "Mean ms": ms(df["decision_ms"].mean()),
        "P95 ms": ms(np.percentile(df["decision_ms"], 95)),
        "Disconnected ODs": int(df[disc_col].max()),
        "N": int(len(df)),
    }


def ranked_list_for_mode(fctx, topo: str, tm: np.ndarray, prev_tm: np.ndarray | None, mode: str) -> tuple[list[int], np.ndarray]:
    if mode == "cache_final":
        neural = M.neural_scores(fctx, tm)
        return bottleneck_rank(tm, fctx.ecmp, fctx.pl, fctx.caps, neural), neural
    ranked, info = M.ranking_from_mode(fctx, topo, -1, tm, prev_tm, mode)
    neural = info.get("neural")
    if neural is None:
        neural = M.neural_scores(fctx, tm)
    return ranked, neural


def run_selected_lp_failure(fctx, selected_ods: list[int], tm: np.ndarray, accepted, db_budget: float, action_k: int) -> dict:
    return M.run_selected_lp(fctx, selected_ods, tm, accepted, db_budget, time_limit=60, action_k=action_k)


def eval_fixed_method(method_name: str, topo: str, scenario: str, fctx, tm: np.ndarray, accepted, prev_tm: np.ndarray | None, ranking_mode: str, fixed_k: int) -> dict:
    ranked, _neural = ranked_list_for_mode(fctx, topo, tm, prev_tm, ranking_mode)
    selected = list(ranked[:fixed_k])
    lpinfo = run_selected_lp_failure(fctx, selected, tm, accepted, 0.051, action_k=fixed_k)
    splits = lpinfo["splits"]
    routing = lpinfo["routing"]
    return {
        "method": method_name,
        "splits": splits,
        "routing": routing,
        "mlu": float(lpinfo["mlu"]),
        "decision_ms": float(lpinfo["decision_lp_ms"]) + float(M.GNN_MS[topo]),
        "selected_k": int(len([od for od in selected if tm[od] > 0])),
        "action": f"K{fixed_k}",
        "solver_status": str(lpinfo["solver_status"]),
    }


def eval_ecmp_method(method_name: str, topo: str, fctx, tm: np.ndarray) -> dict:
    splits = clone_splits(fctx.ecmp)
    routing = M.apply_routing(tm, splits, fctx.pl, fctx.caps)
    return {
        "method": method_name,
        "splits": splits,
        "routing": routing,
        "mlu": float(routing.mlu),
        "decision_ms": 0.0,
        "selected_k": 0,
        "action": "ECMP",
        "solver_status": "BASELINE_NO_LP",
    }


def eval_ospf_method(method_name: str, topo: str, ospf_pl: PathLibrary, ospf_splits: list[np.ndarray], caps: np.ndarray, tm: np.ndarray) -> dict:
    t0 = time.perf_counter()
    routing = M.apply_routing(tm, ospf_splits, ospf_pl, caps)
    decision_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "method": method_name,
        "splits": clone_splits(ospf_splits),
        "routing": routing,
        "mlu": float(routing.mlu),
        "decision_ms": float(decision_ms),
        "selected_k": 0,
        "action": "OSPF",
        "solver_status": "BASELINE_NO_LP",
    }


def eval_final_method(method_name: str, topo: str, cycle_idx: int, fctx, tm: np.ndarray, accepted, prev_tm: np.ndarray | None) -> dict:
    ranked, neural = ranked_list_for_mode(fctx, topo, tm, prev_tm, "cache_final")
    raw, emlu = M.compute_raw_for_state(fctx, topo, -1, tm, prev_tm, ranked, neural)
    keep_mlu = float(M.apply_routing(tm, accepted, fctx.pl, fctx.caps).mlu)
    state = M.standardize(raw, keep_mlu, emlu)
    action_idx = M.action_from_final(fctx, cycle_idx, state, teacher_cycle0=True)
    kind, k, _ = M.ACTIONS[action_idx]
    if kind == "keep":
        splits = accepted
        routing = M.apply_routing(tm, splits, fctx.pl, fctx.caps)
        return {
            "method": method_name,
            "splits": splits,
            "routing": routing,
            "mlu": float(routing.mlu),
            "decision_ms": float(M.GNN_MS[topo]) + 0.5,
            "selected_k": 0,
            "action": M.ANAME[action_idx],
            "solver_status": "KEEP_NO_LP",
        }
    selected = list(ranked[:k])
    db_budget = 1.0 if (cycle_idx == 0 and int(k) >= 300) else 0.051
    lpinfo = run_selected_lp_failure(fctx, selected, tm, accepted, db_budget, action_k=int(k))
    return {
        "method": method_name,
        "splits": lpinfo["splits"],
        "routing": lpinfo["routing"],
        "mlu": float(lpinfo["mlu"]),
        "decision_ms": float(lpinfo["decision_lp_ms"]) + float(M.GNN_MS[topo]),
        "selected_k": int(len([od for od in selected if tm[od] > 0])),
        "action": M.ANAME[action_idx],
        "solver_status": str(lpinfo["solver_status"]),
    }


def method_specs():
    return [
        {
            "name": "ECMP",
            "runner": lambda **kw: eval_ecmp_method("ECMP", kw["topo"], kw["fctx"], kw["tm"]),
        },
        {
            "name": "OSPF-weighted shortest-path routing",
            "runner": lambda **kw: eval_ospf_method(
                "OSPF-weighted shortest-path routing",
                kw["topo"],
                kw["ospf_pl"],
                kw["ospf_splits"],
                kw["fctx"].caps,
                kw["tm"],
            ),
        },
        {
            "name": "Top-K Demand (K50)",
            "runner": lambda **kw: eval_fixed_method(
                "Top-K Demand (K50)",
                kw["topo"],
                kw["scenario"],
                kw["fctx"],
                kw["tm"],
                kw["accepted"],
                kw["prev_tm"],
                "demand",
                50,
            ),
        },
        {
            "name": "Bottleneck Top-K (K50)",
            "runner": lambda **kw: eval_fixed_method(
                "Bottleneck Top-K (K50)",
                kw["topo"],
                kw["scenario"],
                kw["fctx"],
                kw["tm"],
                kw["accepted"],
                kw["prev_tm"],
                "bottleneck",
                50,
            ),
        },
        {
            "name": "GNN-only fixed K50",
            "runner": lambda **kw: eval_fixed_method(
                "GNN-only fixed K50",
                kw["topo"],
                kw["scenario"],
                kw["fctx"],
                kw["tm"],
                kw["accepted"],
                kw["prev_tm"],
                "gnn",
                50,
            ),
        },
        {
            "name": "GNN+LPD fixed K50",
            "runner": lambda **kw: eval_fixed_method(
                "GNN+LPD fixed K50",
                kw["topo"],
                kw["scenario"],
                kw["fctx"],
                kw["tm"],
                kw["accepted"],
                kw["prev_tm"],
                "gnn_lpd",
                50,
            ),
        },
        {
            "name": "Final RG-GNN-LPD",
            "runner": lambda **kw: eval_final_method(
                "Final RG-GNN-LPD",
                kw["topo"],
                kw["cycle_idx"],
                kw["fctx"],
                kw["tm"],
                kw["accepted"],
                kw["prev_tm"],
            ),
        },
    ]


def build_outputs_from_frames(pc: pd.DataFrame, events: pd.DataFrame, expected_methods: list[str]) -> None:
    summary_rows = []
    for scenario in SCENARIOS:
        gsc = pc[pc["Scenario"] == scenario].copy()
        for method in expected_methods:
            gm = gsc[gsc["Method"] == method].copy()
            s = summarize(gm)
            summary_rows.append(
                {
                    "Scenario": scenario,
                    "Method": method,
                    **s,
                }
            )
        pd.DataFrame([r for r in summary_rows if r["Scenario"] == scenario]).to_csv(
            COMPLETED / f"failure_comparison_{scenario}_fix1_COMPLETED.csv",
            index=False,
        )
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)

    final_pc = pc[pc["Method"] == "Final RG-GNN-LPD"].copy()
    final_legacy = []
    for (scenario, topo), g in final_pc.groupby(["Scenario", "Topology"], sort=False):
        final_legacy.append(
            {
                "Scenario": scenario,
                "Topology": topo,
                "Mean PR": round(float(g["PR"].mean() * 100.0), 3),
                "PR>=0.90": round(float(g["PR>=0.90 flag"].mean() * 100.0), 3),
                "PR(fail)": round(float(g["PR"].mean() * 100.0), 3) if scenario != "spike" else "",
                "Mean DB": round(float(g["DB"].mean() * 100.0), 3),
                "P95 DB": round(float(np.percentile(g["DB"], 95) * 100.0), 3),
                "Max disconnected ODs": int(g["Disconnected ODs"].max()),
                "Mean ms": round(float(g["decision_ms"].mean()), 3),
                "P95 ms": round(float(np.percentile(g["decision_ms"], 95)), 3),
            }
        )
    pd.DataFrame(final_legacy).to_csv(COMPLETED / "failure_scenarios_fix1_COMPLETED.csv", index=False)

    final_pc_legacy = final_pc.copy()
    final_pc_legacy["Mean PR"] = final_pc_legacy["PR"]
    final_pc_legacy["PR>=0.90 flag"] = final_pc_legacy["PR>=0.90 flag"]
    final_pc_legacy["PR(fail)"] = np.where(final_pc_legacy["Scenario"] == "spike", np.nan, final_pc_legacy["PR"])
    final_pc_legacy["Mean DB"] = final_pc_legacy["DB"]
    final_pc_legacy["disconnected_ODs"] = final_pc_legacy["Disconnected ODs"]
    final_pc_legacy["failed_links"] = final_pc_legacy["failed_links"]
    final_pc_legacy[
        ["Scenario", "Topology", "tm_index", "Mean PR", "PR>=0.90 flag", "PR(fail)", "Mean DB", "decision_ms", "selected_K", "action", "disconnected_ODs", "failed_links"]
    ].to_csv(COMPLETED / "failure_scenarios_fix1_per_cycle.csv", index=False)

    grouped = pc.groupby("failure_event_key", sort=False)
    bad_method_events = []
    bad_event_specs = []
    for event_key, g in grouped:
        methods_here = sorted(g["Method"].unique().tolist())
        if methods_here != sorted(expected_methods):
            bad_method_events.append((event_key, methods_here))
        if not (
            g["failed_edge_ids"].fillna("").nunique() == 1
            and g["failed_links"].nunique() == 1
            and g["capacity_scale"].nunique() == 1
            and g["spike_factor"].nunique() == 1
            and g["scenario_opt_mlu"].round(12).nunique() == 1
            and g["pr_reference_type"].nunique() == 1
            and g["disconnected_rule"].nunique() == 1
        ):
            bad_event_specs.append(event_key)

    validation_lines = [
        "# Failure Event Validation",
        "",
        f"- Expected methods per event: {', '.join(expected_methods)}",
        f"- Unique events: {len(events)}",
        f"- Per-cycle rows: {len(pc)}",
        f"- Expected rows: {len(events) * len(expected_methods)}",
        f"- Failure seed: {FAILURE_SEED}",
        f"- PR reference type: scenario_path_library_optimum",
        f"- Disconnected-OD rule: `{DISCONNECTED_RULE}`",
        "",
        "## Fairness checks",
        "",
        f"- Every event executed by all requested methods: {'PASS' if not bad_method_events else 'FAIL'}",
        f"- Failed-link IDs / counts / capacity-scale / spike-factor / PR-reference identical across methods per event: {'PASS' if not bad_event_specs else 'FAIL'}",
        f"- Deterministic event generator: PASS (highest-capacity directed links selected; no random sampling in the failure generator)",
        "",
        "## Event generator details",
        "",
        "- single-link failure: highest-capacity directed link zeroed",
        "- two-link failure: top-2 highest-capacity directed links zeroed",
        "- three-link failure: top-3 highest-capacity directed links zeroed",
        "- capacity-50%: all capacities scaled by 0.5",
        "- spike: TM demand scaled by 3.0 with no link failure",
        "- mixed spike+failure: TM demand scaled by 3.0 plus highest-capacity directed link zeroed",
        "",
        "## Artifact paths",
        "",
        f"- Per-cycle comparison: `{OUT_PER_CYCLE.relative_to(ROOT)}`",
        f"- Scenario summary: `{OUT_SUMMARY.relative_to(ROOT)}`",
        f"- Event catalog: `{OUT_EVENTS.relative_to(ROOT)}`",
    ]
    if bad_method_events:
        validation_lines += ["", "## FAIL details — missing methods per event", ""]
        validation_lines.extend([f"- `{event}` -> {methods}" for event, methods in bad_method_events[:20]])
    if bad_event_specs:
        validation_lines += ["", "## FAIL details — inconsistent shared event specs", ""]
        validation_lines.extend([f"- `{event}`" for event in bad_event_specs[:20]])
    OUT_VALIDATION.write_text("\n".join(validation_lines) + "\n")


def main() -> None:
    per_cycle_rows: list[dict] = []
    event_rows: list[dict] = []
    specs = method_specs()
    expected_methods = [s["name"] for s in specs]

    for topo, (lo, hi) in TOPOS.items():
        base = make_base_ctx(topo, lo, hi)
        for scenario in SCENARIOS:
            caps = modify_caps(base.caps0, scenario)
            pl = prune_pathlib(base.pl0, caps)
            fctx = make_failure_ctx(base, pl, caps)
            ospf_pl = build_failure_ospf_path_library(base.ds, caps, weight_mode=REF_WEIGHT_MODE)
            ospf_splits = equal_shortest_splits(ospf_pl)
            accepted = {
                "ECMP": clone_splits(fctx.ecmp),
                "OSPF-weighted shortest-path routing": None,
                "Top-K Demand (K50)": clone_splits(fctx.ecmp),
                "Bottleneck Top-K (K50)": clone_splits(fctx.ecmp),
                "GNN-only fixed K50": clone_splits(fctx.ecmp),
                "GNN+LPD fixed K50": clone_splits(fctx.ecmp),
                "Final RG-GNN-LPD": clone_splits(fctx.ecmp),
            }
            prev_tm = {name: None for name in expected_methods}

            fail_ids = failed_edge_ids(base.caps0, scenario)
            fail_id_text = ",".join(str(x) for x in fail_ids) if fail_ids else ""
            cap_scale = capacity_scale_for(scenario)
            spike_factor = spike_factor_for(scenario)

            for cycle_idx, t in enumerate(range(lo, hi)):
                tm = np.asarray(base.ds.tm[t], float) * spike_factor
                opt = float(solve_all_od_path_lp(tm, pl, caps, time_limit_sec=60).mlu)
                event_key = f"{topo}|{scenario}|{int(t)}"
                event_rows.append(
                    {
                        "Topology": topo,
                        "Scenario": scenario,
                        "tm_index": int(t),
                        "failure_event_key": event_key,
                        "failed_edge_ids": fail_id_text,
                        "failed_links": int(n_failed_links(scenario)),
                        "capacity_scale": float(cap_scale),
                        "spike_factor": float(spike_factor),
                        "pr_reference_type": "scenario_path_library_optimum",
                        "scenario_opt_mlu": float(opt),
                        "failure_seed": FAILURE_SEED,
                    }
                )
                for spec in specs:
                    name = spec["name"]
                    result = spec["runner"](
                        topo=topo,
                        scenario=scenario,
                        cycle_idx=cycle_idx,
                        fctx=fctx,
                        ospf_pl=ospf_pl,
                        ospf_splits=ospf_splits,
                        tm=tm,
                        accepted=accepted[name],
                        prev_tm=prev_tm[name],
                    )
                    splits = result["splits"]
                    routing = result["routing"]
                    mlu = float(result["mlu"])
                    pr = pr_of(opt, mlu)
                    db = float(compute_disturbance(accepted[name], splits, tm))
                    disconnected = int(M.count_disconnected_ods(tm, splits, fctx.pl))
                    per_cycle_rows.append(
                        {
                            "Method": name,
                            "Topology": topo,
                            "Scenario": scenario,
                            "tm_index": int(t),
                            "failure_event_key": event_key,
                            "failed_edge_ids": fail_id_text,
                            "failed_links": int(n_failed_links(scenario)),
                            "capacity_scale": float(cap_scale),
                            "spike_factor": float(spike_factor),
                            "failure_seed": FAILURE_SEED,
                            "pr_reference_type": "scenario_path_library_optimum",
                            "scenario_opt_mlu": float(opt),
                            "PR": float(pr),
                            "PR>=0.90 flag": int(pr >= 0.90),
                            "PR>=0.95 flag": int(pr >= 0.95),
                            "DB": float(db),
                            "decision_ms": round(float(result["decision_ms"]), 3),
                            "Disconnected ODs": disconnected,
                            "selected_K": int(result["selected_k"]),
                            "action": str(result["action"]),
                            "solver_status": str(result["solver_status"]),
                            "achieved_mlu": float(mlu),
                            "disconnected_rule": DISCONNECTED_RULE,
                        }
                    )
                    accepted[name] = splits
                    prev_tm[name] = tm
                print(f"[done] {topo} {scenario} tm={t}", flush=True)

    pc = pd.DataFrame(per_cycle_rows)
    events = pd.DataFrame(event_rows)
    pc.to_csv(OUT_PER_CYCLE, index=False)
    events.to_csv(OUT_EVENTS, index=False)
    build_outputs_from_frames(pc, events, expected_methods)

    print(f"Wrote {OUT_PER_CYCLE}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_EVENTS}")
    print(f"Wrote {OUT_VALIDATION}")


if __name__ == "__main__":
    main()
