#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

import scripts.phase1_5.run_fix1_completed_metrics as R
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import apply_routing
from te.disturbance import compute_disturbance
from te.paths import PathLibrary


ROOT = Path("/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
COMPLETED = ROOT / "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FINAL_REPORT_FIX1/completed_metrics"
OUT_PER_CYCLE = COMPLETED / "ospf_weighted_shortest_path_baseline_N3976.csv"
OUT_SUMMARY = COMPLETED / "ospf_weighted_shortest_path_summary_N3976.csv"

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
TOPO_ORDER = list(WINDOWS)
REF_BW = 100.0


def summarize(df: pd.DataFrame) -> dict:
    return {
        "Mean PR (%)": round(float(df["pr"].mean() * 100.0), 3),
        "PR>=0.90 (% cycles)": round(float((df["pr"] >= 0.90).mean() * 100.0), 3),
        "PR>=0.95 (% cycles)": round(float((df["pr"] >= 0.95).mean() * 100.0), 3),
        "Mean DB (%)": round(float(df["db"].mean() * 100.0), 3),
        "P95 DB (%)": round(float(np.percentile(df["db"], 95) * 100.0), 3),
        "Mean ms": round(float(df["decision_ms"].mean()), 3),
        "P95 ms": round(float(np.percentile(df["decision_ms"], 95)), 3),
        "Min PR (%)": round(float(df["pr"].min() * 100.0), 3),
        "N": int(len(df)),
    }


def build_ospf_path_library(ctx: R.Ctx, weight_mode: str) -> PathLibrary:
    graph = nx.DiGraph()
    for node in ctx.ds.nodes:
        graph.add_node(node)

    edge_to_idx = {edge: idx for idx, edge in enumerate(ctx.ds.edges)}
    for edge_idx, (src, dst) in enumerate(ctx.ds.edges):
        cap = float(ctx.ds.capacities[edge_idx])
        if weight_mode == "unit":
            weight = 1.0
        elif weight_mode == "inverse_capacity":
            weight = float(REF_BW / max(cap, 1e-12))
        else:
            raise ValueError(weight_mode)
        graph.add_edge(src, dst, weight=weight, capacity=cap)

    node_paths_by_od = []
    edge_paths_by_od = []
    edge_idx_paths_by_od = []
    costs_by_od = []
    for src, dst in ctx.ds.od_pairs:
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
        od_pairs=list(ctx.ds.od_pairs),
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


def run() -> None:
    rows = []
    summary_rows = []
    for weight_mode in ["unit", "inverse_capacity"]:
        method_name = "OSPF-weighted shortest-path routing"
        if weight_mode == "inverse_capacity":
            method_name = "OSPF-weighted shortest-path routing (inverse-capacity sensitivity)"
        mode_rows = []
        for topo in TOPO_ORDER:
            lo, hi = WINDOWS[topo]
            ctx = R.get_ctx(topo, lo, hi)
            pl = build_ospf_path_library(ctx, weight_mode)
            splits = equal_shortest_splits(pl)
            accepted = None
            for t in range(lo, hi):
                tm = np.asarray(ctx.ds.tm[t], float)
                t0 = time.perf_counter()
                routing = apply_routing(tm, splits, pl, ctx.caps)
                decision_ms = (time.perf_counter() - t0) * 1000.0
                opt = ctx.refs.get(int(t), ctx.prepass["opt"][t] if ctx.prepass is not None else float("nan"))
                db = 0.0 if accepted is None else float(compute_disturbance(accepted, splits, tm))
                row = {
                    "topology": topo,
                    "tm_index": int(t),
                    "method": method_name,
                    "weight_mode": weight_mode,
                    "achieved_mlu": float(routing.mlu),
                    "opt_mlu": float(opt),
                    "pr": float(R.pr_of(opt, float(routing.mlu))),
                    "db": db,
                    "decision_ms": round(float(decision_ms), 6),
                }
                rows.append(row)
                mode_rows.append(row)
                accepted = splits
            print(f"[done] {weight_mode} {topo}", flush=True)
        sdf = pd.DataFrame(mode_rows)
        summary_rows.append({"Method": method_name, **summarize(sdf)})
    pd.DataFrame(rows).to_csv(OUT_PER_CYCLE, index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
    print(f"Wrote {OUT_PER_CYCLE}")
    print(f"Wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    run()
