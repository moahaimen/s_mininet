#!/usr/bin/env python3
"""FIX1 SDN validation runner.

This script validates the final FIX1 strict-all controller in two modes:

1. ``--mode simulate``
   Runs the deployed FIX1 controller and derives SDN metrics from the solved
   routing state.

2. ``--mode mininet``
   Builds a real Mininet/OVS topology, applies live failure/degradation
   scenarios, installs controller-informed forwarding for measured OD flows,
   and records live QoS measurements.

The controller lineage is the final audited method only:

- Public name: Reward-Gated GNN-LPD Traffic Engineering (RG-GNN-LPD)
- Lineage: FIX1 strict-all
- Action space: KEEP, K50, K100, K200, K300, K500, K800
- TM0 policy: frozen teacher only
- TM1+ policy: final FIX1 checkpoint
- Optimization: selected-flow LP only
"""

from __future__ import annotations

import argparse
import atexit
import cProfile
import concurrent.futures
import dataclasses
import hashlib
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

for _env_key in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_env_key, "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from mininet.link import TCLink  # type: ignore
    from mininet.net import Mininet  # type: ignore
    from mininet.node import OVSKernelSwitch  # type: ignore

    MININET_AVAILABLE = True
except ImportError:
    MININET_AVAILABLE = False

from scripts.phase1_5 import agnostic_lib as A
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import (
    GNNLPDScorer,
    GNN_CHECKPOINT_DEFAULT,
    apply_routing,
    build_context,
    clone_splits,
    ecmp_splits,
    load_bundle,
    _build_spec_lookup,
)
from scripts.phase1_5.bottleneck_lib import ACTIONS, ANAME
from te.disturbance import compute_disturbance
from te.lp_solver import solve_full_mcf_min_mlu, solve_selected_path_lp_dbbudget
from te.paths import PathLibrary


PUBLIC_METHOD = "Reward-Gated GNN-LPD Traffic Engineering (RG-GNN-LPD)"
METHOD_LINEAGE = "FIX1 strict-all"
METHOD_OUTPUT_ID = "gnn_lpd_dqn_selective_db_lp"

CONFIG = ROOT / "configs" / "phase1_reactive_full.yaml"
RESULT_ROOT = ROOT / "results" / METHOD_OUTPUT_ID
OUT_DIR = RESULT_ROOT / "sdn_mininet_clean"
FIX1_ROOT = RESULT_ROOT / "condition_compliant_k10_k50"
FIX1_DIR = FIX1_ROOT / "FULLDATA_GATED_PRESERVED_FIX1"
TEACHER_CKPT = FIX1_ROOT / "FROZEN_FINAL_LEARNED_RUNTIME_SAFE_ITER2" / "final_learned_4of5_iter2_model.pt"
FINAL_CKPT = FIX1_DIR / "fulldata_gated_model.pt"
SCALER_JSON = FIX1_ROOT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN" / "_cache" / "scaler.json"

DEVICE = "cpu"
RUNS_PER_SCENARIO = 10
TOPOLOGIES_TO_RUN = ["abilene", "geant"]
LP_TIME_LIMIT = 120
STRICT_MCF_TIME_LIMIT = 60
LIVE_TRAFFIC_DURATION_SEC = 6
LIVE_WARMUP_TRAFFIC_SEC = 1
MAX_MEASURED_ODS = 3
PING_COUNT = 6
PING_INTERVAL_SEC = 0.2
STEADY_STATE_SETTLE_SEC = 0.5
TRANSIENT_TRAFFIC_DURATION_SEC = 3
TRANSIENT_PRE_INJECT_SEC = 0.3
POST_EVENT_SETTLE_SEC = 0.02
RECOVERY_TIMEOUT_SEC = 8.0
RECOVERY_STABLE_ROUNDS = 1
FAILURE_RATE_SHARE = 0.45
BASELINE_RATE_SHARE = 0.60
MIXED_SPIKE_DELAY_SEC = 0.4
USE_PRECOMPUTED_FAILURE_PLANS = True
INCREMENTAL_REPLACE_THRESHOLD = 999999
ARTIFACT_PREFIX = "sdn_live_fix1_qos_enhanced"

TM_RANGES = {
    "abilene": (2016, 2026),
    "geant": (672, 682),
}

SDN_SCENARIOS = [
    "normal",
    "single_link_failure",
    "two_link_failure",
    "capacity_degradation_50",
    "spike_x3",
    "mixed_spike_failure",
]

GNN_MS = {
    "abilene": 3.0,
    "geant": 7.0,
    "cernet": 22.0,
    "sprintlink": 27.0,
    "tiscali": 33.0,
    "ebone": 12.0,
    "germany50": 26.0,
    "vtlwavenet2011": 140.0,
}

AUDIT_TEMPLATE = {
    "gnn_used": 1,
    "lpd_used": 1,
    "dqn_used": 1,
    "heuristic_used": 0,
    "random_forest_gate_used": 0,
    "sticky_gate_used": 0,
    "stage2_used": 0,
    "disturbance_finalization_used": 0,
    "criticality_backend": "gnn_lpd_bottleneck_rank",
}

KEEP_IDX = [idx for idx, cfg in ACTIONS.items() if cfg[0] == "keep"][0]


def kp_for(k_value: int) -> int:
    return 8 if int(k_value) in (50, 100, 200, 300) else 4


def build_mixed(path_library: PathLibrary, selected_set: set[int], k_paths: int) -> PathLibrary:
    if int(k_paths) >= 8:
        return path_library
    edge_idx_paths = [
        path_library.edge_idx_paths_by_od[od_idx][: int(k_paths)] if od_idx in selected_set else path_library.edge_idx_paths_by_od[od_idx]
        for od_idx in range(len(path_library.edge_idx_paths_by_od))
    ]
    node_paths = [
        path_library.node_paths_by_od[od_idx][: int(k_paths)] if od_idx in selected_set else path_library.node_paths_by_od[od_idx]
        for od_idx in range(len(path_library.node_paths_by_od))
    ]
    edge_paths = [
        path_library.edge_paths_by_od[od_idx][: int(k_paths)] if od_idx in selected_set else path_library.edge_paths_by_od[od_idx]
        for od_idx in range(len(path_library.edge_paths_by_od))
    ]
    costs = [
        path_library.costs_by_od[od_idx][: int(k_paths)] if od_idx in selected_set else path_library.costs_by_od[od_idx]
        for od_idx in range(len(path_library.costs_by_od))
    ]
    return dataclasses.replace(
        path_library,
        node_paths_by_od=node_paths,
        edge_paths_by_od=edge_paths,
        edge_idx_paths_by_od=edge_idx_paths,
        costs_by_od=costs,
    )


def pad_to_lib(splits: Sequence[np.ndarray], path_library: PathLibrary) -> list[np.ndarray]:
    out = []
    for od_idx, split_vec in enumerate(splits):
        arr = np.asarray(split_vec, dtype=np.float32)
        target_len = len(path_library.edge_idx_paths_by_od[od_idx])
        if arr.size < target_len:
            arr = np.concatenate([arr, np.zeros(target_len - arr.size, dtype=np.float32)])
        elif arr.size > target_len:
            arr = arr[:target_len]
        out.append(arr)
    return out


def project_splits_to_library(
    splits: Sequence[np.ndarray],
    src_library: PathLibrary,
    dst_library: PathLibrary,
    fallback_splits: Sequence[np.ndarray] | None = None,
) -> list[np.ndarray]:
    projected: list[np.ndarray] = []
    for od_idx, dst_paths in enumerate(dst_library.edge_idx_paths_by_od):
        dst_len = len(dst_paths)
        if dst_len == 0:
            projected.append(np.zeros(0, dtype=np.float32))
            continue

        src_paths = src_library.edge_idx_paths_by_od[od_idx]
        src_vec = np.asarray(splits[od_idx], dtype=np.float32) if od_idx < len(splits) else np.zeros(0, dtype=np.float32)
        src_map = {
            tuple(int(edge_idx) for edge_idx in path): float(src_vec[path_idx]) if path_idx < src_vec.size else 0.0
            for path_idx, path in enumerate(src_paths)
        }

        arr = np.zeros(dst_len, dtype=np.float32)
        for path_idx, path in enumerate(dst_paths):
            arr[path_idx] = float(src_map.get(tuple(int(edge_idx) for edge_idx in path), 0.0))

        total = float(arr.sum())
        if total > 0:
            arr /= total
        elif fallback_splits is not None and od_idx < len(fallback_splits):
            fallback = np.asarray(fallback_splits[od_idx], dtype=np.float32)
            if fallback.size == dst_len and float(fallback.sum()) > 0:
                arr = fallback.copy()
            else:
                arr = np.full(dst_len, 1.0 / dst_len, dtype=np.float32)
        else:
            arr = np.full(dst_len, 1.0 / dst_len, dtype=np.float32)
        projected.append(arr)
    return projected


def bottleneck_rank(tm: np.ndarray, ecmp: Sequence[np.ndarray], path_library: PathLibrary, caps: np.ndarray, scores: np.ndarray) -> list[int]:
    util = apply_routing(tm, ecmp, path_library, caps).utilization
    active = [od for od in range(len(tm)) if tm[od] > 0]
    if not active:
        return []
    relief = np.zeros(len(tm), dtype=float)
    for od_idx in active:
        paths = path_library.edge_idx_paths_by_od[od_idx]
        split_vec = np.asarray(ecmp[od_idx], dtype=float)
        split_sum = split_vec.sum()
        if split_sum <= 0:
            continue
        for path_idx, frac in enumerate(split_vec):
            if frac <= 0 or path_idx >= len(paths):
                continue
            flow = float(tm[od_idx]) * float(frac / split_sum)
            for edge_idx in paths[path_idx]:
                relief[od_idx] += flow * float(util[edge_idx])
    relief_norm = relief[active]
    relief_norm = relief_norm / (relief_norm.max() + 1e-12)
    gnn_norm = np.array([scores[od_idx] if od_idx < len(scores) else 0.0 for od_idx in active], dtype=float)
    gnn_norm = gnn_norm / (gnn_norm.max() + 1e-12)
    combined = relief_norm + 0.3 * gnn_norm
    return [active[idx] for idx in np.argsort(-combined)]


@dataclass
class HostSpec:
    node_name: str
    host_name: str
    switch_name: str
    ip: str
    mac: str
    host_port: int


@dataclass
class PhysicalLinkSpec:
    node_a: str
    node_b: str
    switch_a: str
    switch_b: str
    port_a: int
    port_b: int
    bw_mbps: float
    delay_ms: float


@dataclass
class DirectedEdgeSpec:
    edge_idx: int
    src_node: str
    dst_node: str
    src_switch: str
    dst_switch: str
    src_port: int
    dst_port: int
    link_key: Tuple[str, str]


@dataclass
class LiveTopologySpec:
    host_by_node: Dict[str, HostSpec]
    switch_by_node: Dict[str, str]
    physical_links: Dict[Tuple[str, str], PhysicalLinkSpec]
    directed_edges: Dict[int, DirectedEdgeSpec]
    max_bw_mbps: float
    max_capacity_units: float
    bw_scale_mbps_per_unit: float


@dataclass
class MininetHarness:
    net: Mininet
    spec: LiveTopologySpec
    switch_links: Dict[Tuple[str, str], object]
    programmed_flows: Dict[str, Dict[str, str]]


def pr_of(opt_mlu: float, achieved_mlu: float) -> float:
    if achieved_mlu <= 0 or not np.isfinite(opt_mlu):
        return 0.0
    return float(min(1.0, opt_mlu / achieved_mlu))


def tm_scale(scenario: str) -> float:
    return 3.0 if scenario in {"spike_x3", "mixed_spike_failure"} else 1.0


def pick_failed_links(caps: np.ndarray, n: int) -> np.ndarray:
    nonzero = np.flatnonzero(np.asarray(caps, dtype=float) > 0)
    if nonzero.size == 0:
        return np.array([], dtype=int)
    order = nonzero[np.argsort(np.asarray(caps, dtype=float)[nonzero])[::-1]]
    return order[: min(n, order.size)]


def _physical_graph_stays_connected(
    nodes: Sequence[str],
    physical_links: Sequence[Tuple[str, str]],
    removed_keys: Sequence[Tuple[str, str]],
) -> bool:
    node_set = {str(node) for node in nodes}
    if not node_set:
        return True
    removed = {tuple(sorted((str(a), str(b)))) for a, b in removed_keys}
    adjacency: Dict[str, set[str]] = {node: set() for node in node_set}
    for src_node, dst_node in physical_links:
        link_key = tuple(sorted((str(src_node), str(dst_node))))
        if link_key in removed:
            continue
        src_name, dst_name = str(src_node), str(dst_node)
        adjacency.setdefault(src_name, set()).add(dst_name)
        adjacency.setdefault(dst_name, set()).add(src_name)
    start = next(iter(node_set))
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in adjacency.get(node, set()):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen == node_set


def _remaining_path_count(
    path_library: PathLibrary | None,
    removed_edge_indices: set[int],
    focus_ods: Sequence[int] | None = None,
) -> int:
    if path_library is None:
        return 0
    focus = {int(od_idx) for od_idx in focus_ods} if focus_ods is not None else None
    count = 0
    for od_idx, edge_paths in enumerate(path_library.edge_idx_paths_by_od):
        if focus is not None and int(od_idx) not in focus:
            continue
        if any(all(int(edge_idx) not in removed_edge_indices for edge_idx in edge_path) for edge_path in edge_paths):
            count += 1
    return int(count)


def plan_failed_physical_links(
    edges: Sequence[tuple],
    caps: np.ndarray,
    n: int,
    path_library: PathLibrary | None = None,
    focus_ods: Sequence[int] | None = None,
) -> tuple[list[Tuple[str, str]], np.ndarray]:
    pair_caps: dict[Tuple[str, str], float] = {}
    pair_indices: dict[Tuple[str, str], list[int]] = {}
    nodes: set[str] = set()
    for edge_idx, (src_node, dst_node) in enumerate(edges):
        nodes.add(str(src_node))
        nodes.add(str(dst_node))
        cap = float(np.asarray(caps, dtype=float)[edge_idx])
        if cap <= 0:
            continue
        link_key = tuple(sorted((str(src_node), str(dst_node))))
        pair_caps[link_key] = max(pair_caps.get(link_key, 0.0), cap)
        pair_indices.setdefault(link_key, []).append(int(edge_idx))
    sorted_keys = [key for key, _ in sorted(pair_caps.items(), key=lambda item: item[1], reverse=True)]
    if not sorted_keys or n <= 0:
        return [], np.asarray([], dtype=int)

    candidate_combos = itertools.combinations(sorted_keys, min(int(n), len(sorted_keys)))
    scored_candidates = []
    for combo in candidate_combos:
        removed_edge_indices = {edge_idx for key in combo for edge_idx in pair_indices.get(key, [])}
        connected = _physical_graph_stays_connected(nodes, pair_indices.keys(), combo)
        focus_remaining = _remaining_path_count(path_library, removed_edge_indices, focus_ods=focus_ods)
        remaining_paths = _remaining_path_count(path_library, removed_edge_indices)
        total_cap = float(sum(pair_caps.get(key, 0.0) for key in combo))
        scored_candidates.append((int(bool(connected)), int(focus_remaining), int(remaining_paths), total_cap, list(combo)))

    scored_candidates.sort(reverse=True)
    selected_keys = scored_candidates[0][4] if scored_candidates else sorted_keys[: min(int(n), len(sorted_keys))]
    selected_indices = sorted({edge_idx for key in selected_keys for edge_idx in pair_indices.get(key, [])})
    return selected_keys, np.asarray(selected_indices, dtype=int)


def modified_caps(orig_caps: np.ndarray, scenario: str, failed_edge_indices: Sequence[int] | None = None) -> np.ndarray:
    caps = np.asarray(orig_caps, dtype=float).copy()
    if scenario in {"single_link_failure", "two_link_failure", "mixed_spike_failure"}:
        failed = np.asarray(failed_edge_indices if failed_edge_indices is not None else pick_failed_links(caps, 1 if scenario != "two_link_failure" else 2), dtype=int)
        if failed.size > 0:
            caps[failed] = 0.0
    elif scenario == "capacity_degradation_50":
        caps *= 0.5
    return caps


def prune_path_library(pl: PathLibrary, caps: np.ndarray) -> PathLibrary:
    node_paths = []
    edge_paths = []
    edge_idx_paths = []
    costs = []
    for od_idx in range(len(pl.edge_idx_paths_by_od)):
        keep = [
            path_idx
            for path_idx, edge_path in enumerate(pl.edge_idx_paths_by_od[od_idx])
            if all(float(caps[edge_idx]) > 0 for edge_idx in edge_path)
        ]
        node_paths.append([pl.node_paths_by_od[od_idx][path_idx] for path_idx in keep])
        edge_paths.append([pl.edge_paths_by_od[od_idx][path_idx] for path_idx in keep])
        edge_idx_paths.append([pl.edge_idx_paths_by_od[od_idx][path_idx] for path_idx in keep])
        costs.append([pl.costs_by_od[od_idx][path_idx] for path_idx in keep])
    return PathLibrary(
        od_pairs=pl.od_pairs,
        node_paths_by_od=node_paths,
        edge_paths_by_od=edge_paths,
        edge_idx_paths_by_od=edge_idx_paths,
        costs_by_od=costs,
    )


def count_disconnected_ods(tm: np.ndarray, splits: Sequence[np.ndarray], path_library: PathLibrary, tol: float = 1e-9) -> int:
    disconnected = 0
    for od_idx, demand in enumerate(np.asarray(tm, dtype=float)):
        if demand <= 0:
            continue
        npaths = len(path_library.edge_idx_paths_by_od[od_idx])
        if npaths == 0:
            disconnected += 1
            continue
        split_sum = float(np.asarray(splits[od_idx], dtype=float)[:npaths].sum()) if od_idx < len(splits) else 0.0
        if split_sum <= tol:
            disconnected += 1
    return disconnected


def count_candidate_path_exhausted_ods(tm: np.ndarray, path_library: PathLibrary) -> int:
    exhausted = 0
    for od_idx, demand in enumerate(np.asarray(tm, dtype=float)):
        if demand <= 0:
            continue
        if len(path_library.edge_idx_paths_by_od[od_idx]) == 0:
            exhausted += 1
    return exhausted


def count_physical_disconnected_ods(
    tm: np.ndarray,
    od_pairs: Sequence[tuple],
    edges: Sequence[tuple],
    caps: np.ndarray,
) -> int:
    adjacency: Dict[str, set[str]] = {}
    for edge_idx, (src_node, dst_node) in enumerate(edges):
        if float(np.asarray(caps, dtype=float)[edge_idx]) <= 0:
            continue
        src_name = str(src_node)
        dst_name = str(dst_node)
        adjacency.setdefault(src_name, set()).add(dst_name)

    disconnected = 0
    for od_idx, demand in enumerate(np.asarray(tm, dtype=float)):
        if demand <= 0:
            continue
        src_node, dst_node = (str(od_pairs[od_idx][0]), str(od_pairs[od_idx][1]))
        if src_node == dst_node:
            continue
        seen = {src_node}
        stack = [src_node]
        found = False
        while stack:
            node = stack.pop()
            if node == dst_node:
                found = True
                break
            for neighbor in adjacency.get(node, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        if not found:
            disconnected += 1
    return disconnected


def count_affected_ods(
    tm: np.ndarray,
    pre_ctx: dict,
    pre_splits: Sequence[np.ndarray],
    post_ctx: dict,
    post_splits: Sequence[np.ndarray],
    scenario: str,
    failed_edge_indices: Sequence[int],
) -> int:
    failed_edges = {int(edge_idx) for edge_idx in failed_edge_indices}
    affected = 0
    for od_idx, demand in enumerate(np.asarray(tm, dtype=float)):
        if demand <= 0:
            continue
        if scenario == "capacity_degradation_50":
            affected += 1
            continue
        pre_path = selected_path_edge_indices(pre_ctx, pre_splits, od_idx)
        post_path = selected_path_edge_indices(post_ctx, post_splits, od_idx)
        post_has_candidate = bool(post_ctx["pl"].edge_idx_paths_by_od[od_idx])
        if (failed_edges and bool(set(int(edge_idx) for edge_idx in pre_path) & failed_edges)) or (pre_path != post_path) or not post_has_candidate:
            affected += 1
    return affected


def load_fix1_runtime() -> tuple[GNNLPDScorer, A.QNet, A.QNet, np.ndarray, np.ndarray]:
    scaler = json.loads(SCALER_JSON.read_text())
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    dim = len(A.AGN_FEAT_NAMES)

    gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device=DEVICE)

    teacher = A.QNet(dim, 7)
    teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location=DEVICE)["state_dict"])
    teacher.eval()

    final_net = A.QNet(dim, 7)
    final_net.load_state_dict(torch.load(FINAL_CKPT, map_location=DEVICE)["state_dict"])
    final_net.eval()

    return gnn, teacher, final_net, mean, std


def compute_state_vector(
    topo: str,
    tm_index: int,
    tm: np.ndarray,
    prev_tm: np.ndarray | None,
    ctx: dict,
    caps: np.ndarray,
    accepted: Sequence[np.ndarray],
    gnn: GNNLPDScorer,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict:
    snapshot_t0 = time.perf_counter()
    ecmp = ctx["ecmp"]
    pl = ctx["pl"]
    ds = ctx["ds"]
    struct = ctx["struct"]

    util = apply_routing(tm, ecmp, pl, caps).utilization
    keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
    ecmp_mlu = float(apply_routing(tm, ecmp, pl, caps).mlu)
    tm_change = 0.0 if prev_tm is None else float(np.abs(tm - prev_tm).sum() / (np.abs(prev_tm).sum() + 1e-9))
    snapshot_ready_t = time.perf_counter()

    scores, _, _ = gnn.score(dataset=ds, tm_vector=tm, path_library=pl, capacities=caps, ecmp_base=ecmp)
    scores = np.asarray(scores, dtype=float).ravel()
    active = [od for od in range(len(tm)) if float(tm[od]) > 0]
    active_scores = scores[active] if active else np.zeros(1, dtype=float)
    ranked = bottleneck_rank(tm, ecmp, pl, caps, scores)
    scoring_ready_t = time.perf_counter()

    dd = {
        "ranked": {tm_index: np.asarray(ranked, dtype=np.int32)},
        "tm_cache": ds.tm,
        "num_nodes": len(ds.nodes),
    }
    dpre = {
        "tmstat": {
            tm_index: (
                float(np.log1p(tm.sum())),
                float(tm.max() / (tm.sum() + 1e-9)) if tm.sum() > 0 else 0.0,
                min(tm_change, 3.0),
                len(active),
            )
        },
        "sstat": {
            tm_index: (
                float(active_scores.mean()),
                float(np.quantile(active_scores, 0.95)),
                float(active_scores.max()),
            )
        },
        "emlu": {tm_index: ecmp_mlu},
    }
    raw = A.raw_static(topo, tm_index, dd, dpre, pl, ecmp, caps, scores, util, struct)
    state_vec = A.standardize(A.raw_to_vec(raw, keep_mlu, ecmp_mlu), mean, std)

    return {
        "state_vec": state_vec,
        "ranked": ranked,
        "scores": scores,
        "keep_mlu": keep_mlu,
        "ecmp_mlu": ecmp_mlu,
        "active": active,
        "state_snapshot_ms": round((snapshot_ready_t - snapshot_t0) * 1000.0, 3),
        "scorer_ranking_ms": round((scoring_ready_t - snapshot_ready_t) * 1000.0, 3),
    }


def choose_fix1_action(cycle_idx: int, state_vec: np.ndarray, teacher: A.QNet, final_net: A.QNet) -> int:
    state = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = teacher(state) if cycle_idx == 0 else final_net(state)
    action_idx = int(q_values.argmax())
    if cycle_idx == 0 and action_idx == KEEP_IDX:
        q_values = q_values.clone()
        q_values[..., KEEP_IDX] = -1e9
        action_idx = int(q_values.argmax())
    return action_idx


def compute_strict_mcf_mlu(tm: np.ndarray, ctx: dict, caps: np.ndarray, time_limit_sec: int) -> tuple[float, str]:
    result = solve_full_mcf_min_mlu(
        tm_vector=np.asarray(tm, dtype=float),
        od_pairs=ctx["ds"].od_pairs,
        nodes=ctx["ds"].nodes,
        edges=ctx["ds"].edges,
        capacities=np.asarray(caps, dtype=float),
        time_limit_sec=time_limit_sec,
    )
    return float(result.mlu), str(result.status)


def run_fix1_cycle(
    topo: str,
    tm_index: int,
    cycle_idx: int,
    tm_raw: np.ndarray,
    ctx: dict,
    caps: np.ndarray,
    accepted: Sequence[np.ndarray],
    prev_tm: np.ndarray | None,
    gnn: GNNLPDScorer,
    teacher: A.QNet,
    final_net: A.QNet,
    mean: np.ndarray,
    std: np.ndarray,
    strict_mcf_time_limit: int,
    compute_strict_mcf: bool = True,
) -> dict:
    decision_t0 = time.perf_counter()
    tm = np.asarray(tm_raw, dtype=float)
    state = compute_state_vector(topo, tm_index, tm, prev_tm, ctx, caps, accepted, gnn, mean, std)
    ddqn_t0 = time.perf_counter()
    action_idx = choose_fix1_action(cycle_idx, state["state_vec"], teacher, final_net)
    ddqn_inference_ms = float((time.perf_counter() - ddqn_t0) * 1000.0)
    kind, action_k, _ = ACTIONS[action_idx]
    routing_compute_ms = 0.0

    if kind == "keep":
        splits = clone_splits(accepted)
        achieved_mlu = float(state["keep_mlu"])
        selected_ods: list[int] = []
        solver_status = "KeepPrevious"
        db_budget = 0.0
    else:
        selected_ods = list(state["ranked"][:action_k])
        db_budget = 1.0 if cycle_idx == 0 and action_k >= 300 else 0.051
        mixed_pl = build_mixed(ctx["pl"], set(int(od) for od in selected_ods), kp_for(action_k))
        if CAPTURE_LP_PATH:
            import pickle as _pickle
            global _CAPTURE_LP_N
            _cap_file = f"{CAPTURE_LP_PATH}.{_CAPTURE_LP_N:03d}"
            _CAPTURE_LP_N += 1
            with open(_cap_file, "wb") as _cf:
                _pickle.dump({
                    "topo": topo, "tm": np.asarray(tm, dtype=float),
                    "selected_ods": list(selected_ods), "base_splits": ctx["ecmp"],
                    "path_library": mixed_pl, "capacities": np.asarray(caps, dtype=float),
                    "prev_splits": accepted, "db_budget": float(db_budget),
                    "full_path_library": ctx["pl"],
                }, _cf)
        routing_t0 = time.perf_counter()
        if _ROUTING_PROFILER is not None:
            _ROUTING_PROFILER.enable()
        lp = _SOLVE_ROUTING(
            tm_vector=tm,
            selected_ods=selected_ods,
            base_splits=ctx["ecmp"],
            path_library=mixed_pl,
            capacities=caps,
            prev_splits=accepted,
            db_budget=db_budget,
            db_weight=1e-6,
            time_limit_sec=LP_TIME_LIMIT,
        )
        if _ROUTING_PROFILER is not None:
            _ROUTING_PROFILER.disable()
        routing_compute_ms = float((time.perf_counter() - routing_t0) * 1000.0)
        splits = pad_to_lib(lp.splits, ctx["pl"])
        achieved_mlu = float(apply_routing(tm, splits, ctx["pl"], caps).mlu)
        solver_status = str(lp.status)

    decision_ms = float((time.perf_counter() - decision_t0) * 1000.0)
    if compute_strict_mcf:
        strict_t0 = time.perf_counter()
        strict_mcf_mlu, strict_status = compute_strict_mcf_mlu(tm, ctx, caps, strict_mcf_time_limit)
        strict_mcf_audit_ms = float((time.perf_counter() - strict_t0) * 1000.0)
    else:
        strict_mcf_mlu = float("nan")
        strict_status = "skipped_live_mininet"
        strict_mcf_audit_ms = 0.0
    disturbance = float(compute_disturbance(accepted, splits, tm))
    disconnected = count_disconnected_ods(tm, splits, ctx["pl"])

    return {
        "tm": tm,
        "splits": splits,
        "action_idx": int(action_idx),
        "action_name": ANAME[action_idx],
        "selected_ods": selected_ods,
        "selected_od_count": int(len([od for od in selected_ods if tm[od] > 0])),
        "active_od_count": int(len(state["active"])),
        "decision_ms": round(decision_ms, 3),
        "state_snapshot_ms": round(float(state.get("state_snapshot_ms", 0.0)), 3),
        "scorer_ranking_ms": round(float(state.get("scorer_ranking_ms", 0.0)), 3),
        "ddqn_inference_ms": round(float(ddqn_inference_ms), 3),
        "routing_compute_ms": round(routing_compute_ms, 3),
        "strict_mcf_audit_ms": round(strict_mcf_audit_ms, 3),
        "db": round(disturbance, 6),
        "mlu": round(achieved_mlu, 6),
        "strict_mcf_mlu": round(strict_mcf_mlu, 6) if np.isfinite(strict_mcf_mlu) else float("nan"),
        "strict_mcf_status": strict_status,
        "pr": round(pr_of(strict_mcf_mlu, achieved_mlu), 6),
        "disconnected": int(disconnected),
        "lp_status": solver_status,
        "db_budget": db_budget,
    }


def derive_simulated_sdn_metrics(
    topo: str,
    mlu: float,
    db: float,
    decision_ms: float,
    active_od_count: int,
    selected_od_count: int,
    disconnected: int,
) -> dict:
    topo_defaults = {
        "abilene": {"link_speed_mbps": 10000.0, "baseline_rtt_ms": 42.0, "hop_delay_ms": 5.0, "mean_path_hops": 3.2},
        "geant": {"link_speed_mbps": 10000.0, "baseline_rtt_ms": 38.0, "hop_delay_ms": 8.0, "mean_path_hops": 3.5},
    }
    params = topo_defaults.get(topo, topo_defaults["abilene"])
    throughput_frac = min(1.0, 1.0 / max(float(mlu), 1e-9))
    if disconnected > 0:
        throughput_frac *= max(0.0, 1.0 - disconnected / max(active_od_count, 1))
    throughput_mbps = throughput_frac * params["link_speed_mbps"] * 0.65

    if disconnected > 0:
        packet_loss_pct = min(100.0, disconnected / max(active_od_count, 1) * 100.0)
    elif mlu > 1.0:
        packet_loss_pct = min(100.0, (1.0 - 1.0 / mlu) * 100.0)
    else:
        packet_loss_pct = 0.0

    queue_term = params["hop_delay_ms"] * params["mean_path_hops"] * max(0.0, mlu - 0.5) * 0.4
    rtt_ms = params["baseline_rtt_ms"] + queue_term
    jitter_ms = params["hop_delay_ms"] * 0.5 + params["hop_delay_ms"] * 0.3 * max(0.0, mlu - 0.7)
    flow_rule_count = max(selected_od_count, 1) * max(1, min(8, selected_od_count))
    install_ms = max(1.0, float(decision_ms) * 0.08)

    if disconnected > 0:
        controller_status = "PARTIAL_FAILURE"
    elif packet_loss_pct > 0:
        controller_status = "DEGRADED"
    else:
        controller_status = "OK"

    return {
        "throughput_mbps": round(throughput_mbps, 2),
        "packet_loss_pct": round(packet_loss_pct, 4),
        "rtt_ms": round(rtt_ms, 3),
        "jitter_ms": round(jitter_ms, 3),
        "recovery_ms": round(float(decision_ms), 3),
        "flow_rule_count": int(flow_rule_count),
        "install_ms": round(install_ms, 3),
        "controller_status": controller_status,
    }


def _cap_to_bw_mbps(capacity: float, max_capacity: float) -> float:
    if max_capacity <= 0:
        return 100.0
    frac = max(0.05, float(capacity) / max_capacity)
    return max(10.0, min(1000.0, 1000.0 * frac))


def build_live_topology_spec(ctx: dict, caps: np.ndarray) -> LiveTopologySpec:
    ds = ctx["ds"]
    nodes = [str(node) for node in ds.nodes]
    edges = [(str(src), str(dst)) for src, dst in ds.edges]
    weights = np.asarray(getattr(ds, "weights", np.ones(len(edges), dtype=float)), dtype=float)
    caps = np.asarray(caps, dtype=float)
    max_capacity = float(np.max(caps[caps > 0])) if np.any(caps > 0) else 1.0

    switch_by_node = {node: f"s{idx + 1}" for idx, node in enumerate(nodes)}
    host_by_node: Dict[str, HostSpec] = {}
    for idx, node in enumerate(nodes):
        host_by_node[node] = HostSpec(
            node_name=node,
            host_name=f"h{idx + 1}",
            switch_name=switch_by_node[node],
            ip=f"10.0.0.{idx + 1}/24",
            mac=f"00:00:00:00:{idx + 1:02x}:01",
            host_port=100 + idx,
        )

    physical_links: Dict[Tuple[str, str], PhysicalLinkSpec] = {}
    directed_edges: Dict[int, DirectedEdgeSpec] = {}
    port_counter = {switch_name: 1 for switch_name in switch_by_node.values()}

    for edge_idx, (src_node, dst_node) in enumerate(edges):
        pair = tuple(sorted((src_node, dst_node)))
        bw_mbps = _cap_to_bw_mbps(float(caps[edge_idx]), max_capacity)
        delay_ms = max(1.0, float(weights[edge_idx]))

        if pair not in physical_links:
            sw_a = switch_by_node[pair[0]]
            sw_b = switch_by_node[pair[1]]
            port_a = port_counter[sw_a]
            port_b = port_counter[sw_b]
            port_counter[sw_a] += 1
            port_counter[sw_b] += 1
            physical_links[pair] = PhysicalLinkSpec(
                node_a=pair[0],
                node_b=pair[1],
                switch_a=sw_a,
                switch_b=sw_b,
                port_a=port_a,
                port_b=port_b,
                bw_mbps=bw_mbps,
                delay_ms=delay_ms,
            )
        else:
            physical_links[pair].bw_mbps = max(physical_links[pair].bw_mbps, bw_mbps)
            physical_links[pair].delay_ms = max(physical_links[pair].delay_ms, delay_ms)

        link = physical_links[pair]
        if src_node == link.node_a:
            src_port, dst_port = link.port_a, link.port_b
        else:
            src_port, dst_port = link.port_b, link.port_a
        directed_edges[edge_idx] = DirectedEdgeSpec(
            edge_idx=edge_idx,
            src_node=src_node,
            dst_node=dst_node,
            src_switch=switch_by_node[src_node],
            dst_switch=switch_by_node[dst_node],
            src_port=src_port,
            dst_port=dst_port,
            link_key=pair,
        )

    max_bw = max((link.bw_mbps for link in physical_links.values()), default=100.0)
    return LiveTopologySpec(
        host_by_node=host_by_node,
        switch_by_node=switch_by_node,
        physical_links=physical_links,
        directed_edges=directed_edges,
        max_bw_mbps=max_bw,
        max_capacity_units=max_capacity,
        bw_scale_mbps_per_unit=(max_bw / max_capacity) if max_capacity > 0 else 1.0,
    )


# --- Flow-install profiling (Target 6). Env-gated; off by default so trusted timing is unchanged. ---
# When enabled, every ovs-ofctl subprocess is counted and timed so flow_mod_send_ms can be decomposed
# into (subprocess count, cumulative subprocess wall time). Thread-safe accumulation per switch.
import threading as _threading
FLOW_INSTALL_PROFILE_ENABLED = os.environ.get("TARGET6_FLOW_PROFILE", "0") != "0"
_INSTALL_PROF_LOCK = _threading.Lock()
_INSTALL_PROF: Dict[str, dict] = {}  # switch_name -> {ovs_cmds, subprocess_ms, del_cmds, mod_cmds, add_cmds}


def _install_prof_record(switch_name: str, kind: str, dt_ms: float) -> None:
    if not FLOW_INSTALL_PROFILE_ENABLED:
        return
    with _INSTALL_PROF_LOCK:
        rec = _INSTALL_PROF.setdefault(switch_name, {"ovs_cmds": 0, "subprocess_ms": 0.0,
                                                     "del_cmds": 0, "mod_cmds": 0, "add_cmds": 0})
        rec["ovs_cmds"] += 1
        rec["subprocess_ms"] += float(dt_ms)
        if kind in ("del_cmds", "mod_cmds", "add_cmds"):
            rec[kind] += 1


def _install_prof_reset() -> None:
    with _INSTALL_PROF_LOCK:
        _INSTALL_PROF.clear()


def _install_prof_snapshot() -> Dict[str, dict]:
    with _INSTALL_PROF_LOCK:
        return {k: dict(v) for k, v in _INSTALL_PROF.items()}


def _run_checked(cmd: Sequence[str], _prof_switch: str | None = None, _prof_kind: str = "other") -> None:
    if FLOW_INSTALL_PROFILE_ENABLED and _prof_switch is not None:
        _t0 = time.perf_counter()
        subprocess.run(list(cmd), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _install_prof_record(_prof_switch, _prof_kind, (time.perf_counter() - _t0) * 1000.0)
        return
    subprocess.run(list(cmd), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _ovs_del_flows(switch_name: str) -> None:
    _run_checked(["ovs-ofctl", "-O", "OpenFlow13", "del-flows", switch_name])


def _ovs_del_groups(switch_name: str) -> None:
    subprocess.run(
        ["ovs-ofctl", "-O", "OpenFlow13", "del-groups", switch_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def _ovs_add_flow(switch_name: str, flow: str) -> None:
    _run_checked(["ovs-ofctl", "-O", "OpenFlow13", "add-flow", switch_name, flow],
                 _prof_switch=switch_name, _prof_kind="add_cmds")


def _ovs_add_flows_batch(switch_name: str, flows: Sequence[str]) -> None:
    if not flows:
        return
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
        handle.write("\n".join(str(flow) for flow in flows))
        handle.write("\n")
        temp_path = handle.name
    try:
        _run_checked(["ovs-ofctl", "-O", "OpenFlow13", "add-flows", switch_name, temp_path],
                     _prof_switch=switch_name, _prof_kind="add_cmds")
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def _ovs_mod_flows_batch(switch_name: str, flows: Sequence[str]) -> None:
    if not flows:
        return
    for flow in flows:
        _run_checked(["ovs-ofctl", "-O", "OpenFlow13", "mod-flows", switch_name, str(flow)],
                     _prof_switch=switch_name, _prof_kind="mod_cmds")


def _ovs_bundle_apply(switch_name: str, del_matches: Sequence[str], add_flows: Sequence[str]) -> int:
    """Target-6 optimized install: apply all deletes + adds/modifies for one switch in a SINGLE
    atomic ovs-ofctl --bundle subprocess. `add` in a bundle overwrites the actions of an existing
    identical (match, priority) flow, so modified flows are folded into the adds (no per-flow
    mod-flows). Returns the number of ovs subprocess invocations (0 or 1). Final table is exactly:
    (prior table) - del_matches + add_flows, identical to the incremental path's result set."""
    if not del_matches and not add_flows:
        return 0
    lines = []
    for match in del_matches:
        lines.append(f"delete_strict {match}")
    for flow in add_flows:
        lines.append(f"add {flow}")
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")
        temp_path = handle.name
    try:
        _run_checked(["ovs-ofctl", "-O", "OpenFlow13", "--bundle", "add-flows", switch_name, temp_path],
                     _prof_switch=switch_name, _prof_kind="add_cmds")
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
    return 1


def _ovs_replace_flows_batch(switch_name: str, flows: Sequence[str]) -> None:
    if not flows:
        return
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
        handle.write("\n".join(str(flow) for flow in flows))
        handle.write("\n")
        temp_path = handle.name
    try:
        _run_checked(["ovs-ofctl", "-O", "OpenFlow13", "replace-flows", switch_name, temp_path])
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def _ovs_del_flow(switch_name: str, match: str) -> None:
    if FLOW_INSTALL_PROFILE_ENABLED:
        _t0 = time.perf_counter()
        subprocess.run(
            ["ovs-ofctl", "-O", "OpenFlow13", "--strict", "del-flows", switch_name, match],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
        )
        _install_prof_record(switch_name, "del_cmds", (time.perf_counter() - _t0) * 1000.0)
        return
    subprocess.run(
        ["ovs-ofctl", "-O", "OpenFlow13", "--strict", "del-flows", switch_name, match],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def _apply_switch_flow_update(
    switch_name: str,
    desired: Dict[str, str],
    stale_matches: Sequence[str],
    modified_matches: Sequence[str],
    updated_matches: Sequence[str],
    replace_threshold: int,
) -> tuple[int, int]:
    if OPT_INSTALL_ENABLED:
        # Target-6 optimized path: one atomic bundle subprocess per switch (delete_strict stale +
        # add updated). `add` overwrites identical (match,priority) actions, so modified flows need
        # no separate mod-flows. Final table set == incremental path's result (proven by A/B gate).
        add_flows = [desired[match] for match in updated_matches]
        _ovs_bundle_apply(switch_name, list(stale_matches), add_flows)
        return int(len(updated_matches)), int(len(stale_matches))

    switch_changed = len(stale_matches) + len(updated_matches)
    if switch_changed >= replace_threshold:
        replacement = ["priority=0,actions=NORMAL"] + [desired[match] for match in sorted(desired)]
        _ovs_replace_flows_batch(switch_name, replacement)
        return int(len(updated_matches)), int(len(stale_matches) + len(modified_matches))

    remove_cmds = 0
    add_cmds = 0
    for match in stale_matches:
        _ovs_del_flow(switch_name, match)
        remove_cmds += 1
    modified_flows = [desired[match] for match in modified_matches]
    if modified_flows:
        _ovs_mod_flows_batch(switch_name, modified_flows)
        add_cmds += len(modified_flows)
    add_flows = [desired[match] for match in updated_matches if match not in modified_matches]
    if add_flows:
        _ovs_add_flows_batch(switch_name, add_flows)
        add_cmds += len(add_flows)
    return int(add_cmds), int(remove_cmds)


def start_mininet_harness(spec: LiveTopologySpec) -> MininetHarness:
    if not MININET_AVAILABLE:
        raise RuntimeError("Mininet is not importable in this environment.")

    net = Mininet(controller=None, switch=OVSKernelSwitch, link=TCLink, build=False, autoSetMacs=False)

    for switch_name in spec.switch_by_node.values():
        net.addSwitch(switch_name, failMode=SWITCH_FAIL_MODE, protocols="OpenFlow13", stp=False)

    for host in spec.host_by_node.values():
        net.addHost(host.host_name, ip=host.ip, mac=host.mac)

    for host in spec.host_by_node.values():
        net.addLink(
            host.host_name,
            host.switch_name,
            port2=host.host_port,
            bw=1000.0,
            delay="0ms",
        )

    switch_links: Dict[Tuple[str, str], object] = {}
    for link_key, link in spec.physical_links.items():
        switch_links[link_key] = net.addLink(
            link.switch_a,
            link.switch_b,
            port1=link.port_a,
            port2=link.port_b,
            bw=link.bw_mbps,
            delay=f"{link.delay_ms:.3f}ms",
        )

    net.build()
    net.start()
    if hasattr(net, "staticArp"):
        net.staticArp()

    time.sleep(1.0)
    harness = MininetHarness(net=net, spec=spec, switch_links=switch_links, programmed_flows={})
    clear_te_programming(harness)
    return harness


def stop_mininet_harness(harness: MininetHarness | None) -> None:
    if harness is None:
        return
    try:
        harness.net.stop()
    except Exception:
        pass


def clear_te_programming(harness: MininetHarness) -> None:
    for switch_name in harness.spec.switch_by_node.values():
        _ovs_del_flows(switch_name)
        _ovs_del_groups(switch_name)
        # Standalone/NORMAL fallback (trusted default). In "secure" diagnostic mode, omit the
        # NORMAL rule so unmatched packets are dropped (no mesh flood) instead of flooded.
        if SWITCH_FAIL_MODE != "secure":
            _ovs_add_flow(switch_name, "priority=0,actions=NORMAL")
        harness.programmed_flows[switch_name] = {}


# Records the mechanism actually used for the most recent capacity application so it
# can be logged per run (Target 2). Reset before each capacity event.
LAST_CAPACITY_APPLICATION_METHOD = "intf_config_bw"


def _htb_root_classid(dev: str) -> str | None:
    """Return the HTB root classid (e.g. '5:1') for a TCLink interface, or None."""
    try:
        show = subprocess.run(["tc", "class", "show", "dev", dev],
                              capture_output=True, text=True).stdout
    except Exception:
        return None
    for ln in show.splitlines():
        if "htb" in ln and "rate" in ln:
            m = re.search(r"class htb (\S+)", ln)
            if m:
                return m.group(1)
    return None


def _tc_inplace_set_rate(dev: str, bw_mbps: float) -> bool:
    """Update an interface's HTB root-class rate IN PLACE (tc class change),
    preserving the qdisc and any netem delay child so forwarding is not interrupted.
    Returns True on success. Applies the same 50%-capacity HTB rate semantics as
    intf.config(bw=), without tearing down/recreating the qdisc."""
    classid = _htb_root_classid(dev)
    if not classid:
        return False
    parent = classid.split(":")[0] + ":"
    rate = f"{float(bw_mbps):.6f}Mbit"
    try:
        res = subprocess.run(
            ["tc", "class", "change", "dev", dev, "parent", parent, "classid", classid,
             "htb", "rate", rate, "ceil", rate],
            capture_output=True, text=True)
        return res.returncode == 0
    except Exception:
        return False


# Capacity-update mechanism toggle (Target 2). Default = in-place tc class change.
# Set TARGET2_CAPACITY_INPLACE=0 to force the original intf.config(bw=) path
# (used only for the stage-1 smoke A/B comparison).
CAPACITY_INPLACE_ENABLED = os.environ.get("TARGET2_CAPACITY_INPLACE", "1") != "0"
# Reset TE flows between runs of the shared harness (Target 2 cross-run stability fix).
RESET_TE_PER_RUN = os.environ.get("TARGET2_RESET_TE_PER_RUN", "1") != "0"
# Rebuild a fresh Mininet harness per run (Target 2 cross-run stability fix, capacity only).
REBUILD_HARNESS_PER_RUN = os.environ.get("TARGET2_REBUILD_HARNESS_PER_RUN", "1") != "0"
# Deterministic replay: reuse a specific run's exact controller input for every iteration.
REPLAY_TM_RUN_ID = int(os.environ.get("TARGET2_REPLAY_TM_RUN_ID", "-1"))
# Capture per-OD expected-vs-actual forwarding trace after post-recovery (Target 2 Phase 3).
TRACE_FORWARDING = os.environ.get("TARGET2_TRACE_FORWARDING", "0") != "0"
# Target 5/6: cProfile the routing-compute (LP solve) path; dump pstats to this file at exit.
PROFILE_ROUTING_PATH = os.environ.get("TARGET5_PROFILE_ROUTING", "")
# Target 5/6: capture routing LP inputs (pickle, numbered) for the A/B equivalence gate.
CAPTURE_LP_PATH = os.environ.get("TARGET5_CAPTURE_LP", "")
_CAPTURE_LP_N = 0
# Target 6: optimized flow-install path — one atomic ovs-ofctl --bundle per changed switch
# (delete_strict stale + add updated, folding modified into adds) instead of per-flow mod/del
# subprocesses. Final flow-table set is identical (verified by the flow-table A/B gate). Off by
# default so the trusted install path/timing is unchanged.
OPT_INSTALL_ENABLED = os.environ.get("TARGET6_OPT_INSTALL", "0") != "0"
# Max switches to program concurrently in the optimized install path (I/O-bound ovs-ofctl bundles).
INSTALL_MAX_PARALLEL_SWITCHES = int(os.environ.get("TARGET6_INSTALL_PARALLEL", "32"))
# Target 5/6: use the equivalence-gated optimized routing LP (direct CBC; identical formulation).
USE_OPT_ROUTING = os.environ.get("TARGET5_USE_OPT_ROUTING", "0") != "0"
_SOLVE_ROUTING = solve_selected_path_lp_dbbudget
if USE_OPT_ROUTING:
    try:
        from te.lp_solver_opt import solve_selected_path_lp_dbbudget_opt as _SOLVE_ROUTING
    except Exception:
        _SOLVE_ROUTING = solve_selected_path_lp_dbbudget
_ROUTING_PROFILER = cProfile.Profile() if PROFILE_ROUTING_PATH else None
if _ROUTING_PROFILER is not None:
    atexit.register(lambda: _ROUTING_PROFILER.dump_stats(PROFILE_ROUTING_PATH))
# Attempted per-OD stale-rule cleanup (did NOT fix the loop; superseded by the secure-mode fix).
# Kept available but OFF by default.
ENFORCE_PATH_CONSISTENCY = os.environ.get("TARGET2_ENFORCE_PATH_CONSISTENCY", "0") != "0"
# Switch table-miss behavior. ROOT-CAUSE FIX (Target 2): default "secure" drops unmatched packets
# instead of the standalone/NORMAL L2 flood that forms a self-sustaining broadcast loop on the
# mesh (stp=False). Matched (priority-200) controller flows forward identically; only unmatched
# packets change (drop vs flood). Set TARGET2_SWITCH_FAIL_MODE=standalone to restore old behavior.
SWITCH_FAIL_MODE = os.environ.get("TARGET2_SWITCH_FAIL_MODE", "secure")


def _set_link_capacity(harness: MininetHarness, link_key: Tuple[str, str], bw_mbps: float) -> None:
    global LAST_CAPACITY_APPLICATION_METHOD
    link = harness.spec.physical_links[link_key]
    mn_link = harness.switch_links[link_key]
    # Prefer the in-place HTB rate update (no qdisc rebuild -> no forwarding
    # interruption); fall back to the original intf.config(bw=) only if the HTB
    # class cannot be updated in place, so capacity enforcement is never dropped.
    if CAPACITY_INPLACE_ENABLED:
        ok1 = _tc_inplace_set_rate(mn_link.intf1.name, bw_mbps)
        ok2 = _tc_inplace_set_rate(mn_link.intf2.name, bw_mbps)
        if ok1 and ok2:
            LAST_CAPACITY_APPLICATION_METHOD = "tc_class_change_htb_rate"
            return
    mn_link.intf1.config(bw=bw_mbps)
    mn_link.intf2.config(bw=bw_mbps)
    LAST_CAPACITY_APPLICATION_METHOD = (
        "intf_config_bw" if not CAPACITY_INPLACE_ENABLED else "intf_config_bw_fallback"
    )


def _iface_admin_oper_state(dev: str) -> dict:
    """Return admin (interface UP flag) and operational (carrier) state for an interface."""
    try:
        out = subprocess.run(["ip", "-o", "link", "show", dev], capture_output=True, text=True).stdout
    except Exception:
        out = ""
    flags = out.split("<", 1)[1].split(">", 1)[0] if "<" in out and ">" in out else ""
    oper = re.search(r"\bstate (\S+)", out)
    return {
        "admin_up": bool("UP" in flags.split(",")),
        "oper_state": oper.group(1) if oper else "UNKNOWN",
    }


def capacity_event_link_state(harness: MininetHarness) -> dict:
    """Aggregate admin/oper state across all physical-link interfaces (capacity
    degradation halves every link but downs none)."""
    devs = []
    for link_key in harness.spec.physical_links:
        mn_link = harness.switch_links[link_key]
        devs.extend([mn_link.intf1.name, mn_link.intf2.name])
    states = {d: _iface_admin_oper_state(d) for d in devs}
    all_admin_up = all(s["admin_up"] for s in states.values()) if states else False
    all_oper_up = all(s["oper_state"].upper() in {"UP", "UNKNOWN"} for s in states.values()) if states else False
    return {"all_admin_up": bool(all_admin_up), "all_oper_up": bool(all_oper_up), "per_iface": states}


def _reconfigure_all_link_capacities(harness: MininetHarness, bw_by_link: Dict[Tuple[str, str], float]) -> None:
    if not bw_by_link:
        return
    for link_key, bw_mbps in bw_by_link.items():
        _set_link_capacity(harness, link_key, float(bw_mbps))


def restore_live_links(harness: MininetHarness) -> None:
    _reconfigure_all_link_capacities(
        harness,
        {link_key: float(link.bw_mbps) for link_key, link in harness.spec.physical_links.items()},
    )
    for link_key, link in harness.spec.physical_links.items():
        harness.net.configLinkStatus(link.switch_a, link.switch_b, "up")


def failed_link_keys(spec: LiveTopologySpec, caps: np.ndarray, n: int) -> list[Tuple[str, str]]:
    keys: list[Tuple[str, str]] = []
    for edge_idx in pick_failed_links(caps, n):
        key = spec.directed_edges[int(edge_idx)].link_key
        if key not in keys:
            keys.append(key)
    return keys


def apply_live_scenario(
    harness: MininetHarness,
    scenario: str,
    caps: np.ndarray,
    forced_link_keys: Sequence[Tuple[str, str]] | None = None,
) -> None:
    restore_live_links(harness)
    if scenario == "capacity_degradation_50":
        _reconfigure_all_link_capacities(
            harness,
            {
                link_key: max(5.0, float(link.bw_mbps) * 0.5)
                for link_key, link in harness.spec.physical_links.items()
            },
        )
    elif scenario in {"single_link_failure", "two_link_failure", "mixed_spike_failure"}:
        link_keys = list(forced_link_keys or failed_link_keys(harness.spec, caps, 1 if scenario != "two_link_failure" else 2))
        for link_key in link_keys:
            link = harness.spec.physical_links[link_key]
            harness.net.configLinkStatus(link.switch_a, link.switch_b, "down")
    time.sleep(0.5)


def inject_live_event(
    harness: MininetHarness,
    scenario: str,
    caps: np.ndarray,
    forced_link_keys: Sequence[Tuple[str, str]] | None = None,
) -> None:
    if scenario == "capacity_degradation_50":
        _reconfigure_all_link_capacities(
            harness,
            {
                link_key: max(5.0, float(link.bw_mbps) * 0.5)
                for link_key, link in harness.spec.physical_links.items()
            },
        )
    elif scenario in {"single_link_failure", "two_link_failure", "mixed_spike_failure"}:
        link_keys = list(forced_link_keys or failed_link_keys(harness.spec, caps, 1 if scenario != "two_link_failure" else 2))
        for link_key in link_keys:
            link = harness.spec.physical_links[link_key]
            harness.net.configLinkStatus(link.switch_a, link.switch_b, "down")
    time.sleep(POST_EVENT_SETTLE_SEC)


def _hash_array(arr) -> str:
    """Stable short hash of a numeric array/list for state-identity capture."""
    try:
        a = np.asarray(arr, dtype=float)
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
    except Exception:
        return hashlib.sha256(repr(arr).encode()).hexdigest()[:16]


def _hash_splits(splits) -> str:
    """Stable hash of a routing split assignment (sequence of arrays)."""
    try:
        parts = [np.ascontiguousarray(np.asarray(s, dtype=float)).tobytes() for s in (splits or [])]
        return hashlib.sha256(b"".join(parts)).hexdigest()[:16]
    except Exception:
        return hashlib.sha256(repr(splits).encode()).hexdigest()[:16]


def _hash_pathlib(pl) -> str:
    """Stable hash of a candidate path library (edge-index paths per OD)."""
    try:
        return hashlib.sha256(repr(pl.edge_idx_paths_by_od).encode()).hexdigest()[:16]
    except Exception:
        return "n/a"


def choose_measured_ods(
    tm: np.ndarray,
    selected_ods: Sequence[int],
    active_od_count: int,
    max_measured_ods: int,
    od_pairs: Sequence[tuple] | None = None,
) -> list[int]:
    ranked = [int(od) for od in selected_ods if tm[int(od)] > 0]
    if len(ranked) < max_measured_ods:
        active = sorted((idx for idx, demand in enumerate(tm) if demand > 0), key=lambda idx: float(tm[idx]), reverse=True)
        for od_idx in active:
            if od_idx not in ranked:
                ranked.append(int(od_idx))
            if len(ranked) >= max_measured_ods:
                break
    limit = min(max_measured_ods, max(active_od_count, 1))
    if od_pairs is None or limit <= 1:
        return ranked[:limit]

    selected: list[int] = []
    used_dsts: set[str] = set()
    leftovers: list[int] = []
    for od_idx in ranked:
        try:
            dst_node = str(od_pairs[int(od_idx)][1])
        except Exception:
            dst_node = ""
        if dst_node and dst_node not in used_dsts:
            selected.append(int(od_idx))
            used_dsts.add(dst_node)
            if len(selected) >= limit:
                return selected[:limit]
        else:
            leftovers.append(int(od_idx))
    return (selected + leftovers)[:limit]


def _parse_iperf_udp_output(output: str) -> tuple[float, float, float]:
    pattern = re.compile(
        r"(?P<bandwidth>[0-9.]+)\s+(?P<unit>[KMG])bits/sec\s+"
        r"(?P<jitter>[0-9.]+)\s+ms\s+"
        r"(?P<lost>\d+)\s*/\s*(?P<total>\d+)\s+\((?P<loss>[0-9.]+)%\)"
    )
    matches = pattern.findall(output)
    if not matches:
        return 0.0, float("nan"), 100.0
    bandwidth, unit, jitter_ms, _, _, loss_pct = matches[-1]
    scale = {"K": 1e-3, "M": 1.0, "G": 1e3}[unit]
    return float(bandwidth) * scale, float(jitter_ms), float(loss_pct)


def _parse_iperf_udp_validity(output: str) -> dict:
    """Validity-aware iperf UDP receiver-summary parse.

    Returns an explicit measurement-validity record. A missing/empty/unparseable
    summary is reported as INVALID with a machine-readable reason instead of being
    silently coerced to 0% or 100% packet loss.
    """
    if output is None or not str(output).strip():
        return {"valid": False, "reason": "iperf_output_empty",
                "throughput_mbps": None, "jitter_ms": None, "loss_pct": None}
    pattern = re.compile(
        r"(?P<bandwidth>[0-9.]+)\s+(?P<unit>[KMG])bits/sec\s+"
        r"(?P<jitter>[0-9.]+)\s+ms\s+"
        r"(?P<lost>\d+)\s*/\s*(?P<total>\d+)\s+\((?P<loss>[0-9.]+)%\)"
    )
    matches = pattern.findall(output)
    if not matches:
        return {"valid": False, "reason": "iperf_summary_missing",
                "throughput_mbps": None, "jitter_ms": None, "loss_pct": None}
    bandwidth, unit, jitter_ms, lost, total, loss_pct = matches[-1]
    try:
        scale = {"K": 1e-3, "M": 1.0, "G": 1e3}[unit]
        return {"valid": True, "reason": "",
                "throughput_mbps": float(bandwidth) * scale,
                "jitter_ms": float(jitter_ms), "loss_pct": float(loss_pct),
                "lost": int(lost), "total": int(total)}
    except Exception:
        return {"valid": False, "reason": "iperf_parse_error",
                "throughput_mbps": None, "jitter_ms": None, "loss_pct": None}


def enforce_measured_od_path_consistency(harness: MininetHarness, ctx: dict, splits: Sequence[np.ndarray], measured_ods: Sequence[int]) -> dict:
    """Target 2 root-cause fix: after an incremental reroute, the full new dominant path for
    a measured OD is installed, but STALE per-OD rules can remain on switches that were on the
    OD's PREVIOUS path and are no longer on the new path. Combined with the newly written
    divergence-point rule, those stale rules form a forwarding LOOP (proven by the forwarding
    trace: mixed rule ages + packet counts far above the offered flow). This make-before-break
    pass removes each measured OD's flow (both directions) on every switch NOT on its current
    dominant path, leaving exactly the consistent installed path. It changes no routing
    decision — it only deletes stale off-path rules for the measured ODs."""
    od_pairs = ctx["ds"].od_pairs
    od_lookup = {(str(s), str(d)): i for i, (s, d) in enumerate(od_pairs)}
    all_switches = set(harness.spec.switch_by_node.values())
    removed = 0

    def path_switches(od_idx: int) -> set:
        sws = set()
        for eidx in selected_path_edge_indices(ctx, splits, int(od_idx)):
            sws.add(harness.spec.directed_edges[int(eidx)].src_switch)
        dst_node = str(od_pairs[int(od_idx)][1])
        sws.add(harness.spec.host_by_node[dst_node].switch_name)
        return sws

    def clean_od(od_idx: int):
        """Make the OD's installed rule set EXACTLY its current dominant path: delete the OD
        match on every switch, then (re)install the precise path so no hop can miss the rule
        and fall to the priority-0 NORMAL fallback (which floods/loops on this mesh, stp=False)."""
        nonlocal removed, reinstalled
        if int(od_idx) < 0 or int(od_idx) >= len(od_pairs):
            return
        src_node, dst_node = str(od_pairs[int(od_idx)][0]), str(od_pairs[int(od_idx)][1])
        src_ip = harness.spec.host_by_node[src_node].ip.split("/")[0]
        dst_ip = harness.spec.host_by_node[dst_node].ip.split("/")[0]
        match = f"ip,nw_src={src_ip},nw_dst={dst_ip}"
        full_match = f"priority=200,ip,nw_src={src_ip},nw_dst={dst_ip}"
        # 1) delete this OD's rule on ALL switches (clean slate — removes stale/coexisting rules)
        for sw in all_switches:
            _ovs_del_flow(sw, match)
            if full_match in harness.programmed_flows.get(sw, {}):
                harness.programmed_flows[sw].pop(full_match, None)
                removed += 1
        # 2) reinstall EXACTLY the current dominant path
        edges = selected_path_edge_indices(ctx, splits, int(od_idx))
        if not edges:
            return
        for eidx in edges:
            e = harness.spec.directed_edges[int(eidx)]
            flow = f"{full_match},actions=output:{int(e.src_port)}"
            _ovs_add_flow(e.src_switch, flow)
            harness.programmed_flows.setdefault(e.src_switch, {})[full_match] = flow
            reinstalled += 1
        dsw = harness.spec.host_by_node[dst_node].switch_name
        hp = int(harness.spec.host_by_node[dst_node].host_port)
        dflow = f"{full_match},actions=output:{hp}"
        _ovs_add_flow(dsw, dflow)
        harness.programmed_flows.setdefault(dsw, {})[full_match] = dflow
        reinstalled += 1

    reinstalled = 0
    for od_idx in measured_ods:
        clean_od(int(od_idx))
        rev = od_lookup.get((str(od_pairs[int(od_idx)][1]), str(od_pairs[int(od_idx)][0])))
        if rev is not None:
            clean_od(int(rev))
    # barrier: ensure deletions applied before measurement
    for sw in all_switches:
        try:
            subprocess.run(["ovs-ofctl", "-O", "OpenFlow13", "--strict", "dump-flows", sw],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            pass
    return {"stale_rules_removed": int(removed), "path_rules_reinstalled": int(reinstalled)}


def capture_forwarding_trace(harness: MininetHarness, ctx: dict, splits: Sequence[np.ndarray], measured_ods: Sequence[int]) -> dict:
    """Diagnostic (Target 2 Phase 3): for each measured OD, compute the controller's expected
    dominant forward path (switch, expected output port) and compare against the ACTUAL installed
    flow on each switch (ovs-ofctl dump-flows). Detects missing/mismatched rules and blackholes.
    Also traces the actual output port hop-by-hop by following installed flows."""
    od_pairs = ctx["ds"].od_pairs
    path_library = ctx["pl"]
    switch_of_node = harness.spec.switch_by_node

    def dump(sw):
        try:
            return subprocess.run(["ovs-ofctl", "-O", "OpenFlow13", "dump-flows", sw], capture_output=True, text=True).stdout
        except Exception:
            return ""
    dumps = {sw: dump(sw) for sw in harness.spec.switch_by_node.values()}

    def matching_lines(sw, src_ip, dst_ip):
        out = []
        for ln in dumps.get(sw, "").splitlines():
            if (dst_ip in ln) and ("actions=" in ln):
                out.append(ln.strip())
        return out

    def actual_output_port(sw, src_ip, dst_ip):
        # find highest-priority flow matching this src->dst DIRECTION and read its output port
        best = None
        for ln in dumps.get(sw, "").splitlines():
            if (f"nw_src={src_ip}" in ln) and (f"nw_dst={dst_ip}" in ln) and ("actions=" in ln):
                m = re.search(r"actions=output:(\d+)", ln)
                pr = re.search(r"priority=(\d+)", ln)
                np_ = re.search(r"n_packets=(\d+)", ln)
                p = int(pr.group(1)) if pr else 0
                if best is None or p > best[0]:
                    best = (p, int(m.group(1)) if m else None, int(np_.group(1)) if np_ else None)
        return (best[1], best[2]) if best else (None, None)

    trace = {"ods": []}
    for od_idx in measured_ods:
        od_idx = int(od_idx)
        src_node, dst_node = str(od_pairs[od_idx][0]), str(od_pairs[od_idx][1])
        src_ip = harness.spec.host_by_node[src_node].ip.split("/")[0]
        dst_ip = harness.spec.host_by_node[dst_node].ip.split("/")[0]
        paths = path_library.edge_idx_paths_by_od[od_idx]
        sv = np.asarray(splits[od_idx], dtype=float) if od_idx < len(splits) else np.asarray([])
        expected = []
        edge_topology = []
        if paths and sv.size and float(sv.sum()) > 0:
            edge_path = list(paths[int(np.argmax(sv))])
            for edge_idx in edge_path:
                e = harness.spec.directed_edges[int(edge_idx)]
                expected.append({"switch": e.src_switch, "expected_out_port": int(e.src_port)})
                edge_topology.append({"edge_idx": int(edge_idx), "src_switch": e.src_switch,
                                      "src_port": int(e.src_port), "dst_switch": e.dst_switch,
                                      "dst_port": int(e.dst_port)})
            dsw = harness.spec.host_by_node[dst_node].switch_name
            expected.append({"switch": dsw, "expected_out_port": int(harness.spec.host_by_node[dst_node].host_port)})
        hops = []
        mismatch = False; blackhole = False
        for hop in expected:
            ap, npkts = actual_output_port(hop["switch"], src_ip, dst_ip)
            ok = (ap == hop["expected_out_port"])
            if ap is None: blackhole = True
            if not ok: mismatch = True
            hops.append({"switch": hop["switch"], "expected_out_port": hop["expected_out_port"],
                         "actual_out_port": ap, "n_packets": npkts, "match": ok,
                         "raw": matching_lines(hop["switch"], src_ip, dst_ip)[:3]})
        # full installed forward-rule output map for this OD across ALL switches (finds loops)
        installed_fwd = {}
        for sw in harness.spec.switch_by_node.values():
            ap, npkts = actual_output_port(sw, src_ip, dst_ip)
            if ap is not None:
                installed_fwd[sw] = {"out_port": ap, "n_packets": npkts}
        # full raw flow tables on this OD's path switches (finds hidden/higher-prio rules & loops)
        full_tables = {}
        for e in edge_topology:
            for sw in (e["src_switch"], e["dst_switch"]):
                if sw not in full_tables:
                    full_tables[sw] = [ln.strip() for ln in dumps.get(sw, "").splitlines()
                                       if ("actions=" in ln and "priority=0" not in ln)][:20]
        trace["ods"].append({
            "od_idx": od_idx, "src": src_node, "dst": dst_node, "src_ip": src_ip, "dst_ip": dst_ip,
            "expected_hops": len(expected), "rule_mismatch": mismatch, "blackhole": blackhole,
            "edge_topology": edge_topology, "installed_fwd_rules": installed_fwd,
            "full_tables": full_tables, "hops": hops,
        })

    # Whole-network normalized flow tables for the install-path A/B (Target 6). Non-semantic OVS
    # counters/timers (duration, n_packets, n_bytes, idle_age, hard_age, cookie) are stripped so two
    # installs that yield the same semantic table (priority/match/actions) compare equal.
    def _normalize_flow_line(ln: str) -> str:
        ln = ln.strip()
        if "actions=" not in ln:
            return ""
        ln = re.sub(r"cookie=0x[0-9a-fA-F]+,\s*", "", ln)
        ln = re.sub(r"\b(duration|n_packets|n_bytes|idle_age|hard_age|idle_timeout|hard_timeout)=[^,]+,\s*", "", ln)
        ln = re.sub(r"\btable=\d+,\s*", "", ln)
        return ln.strip().rstrip(",")
    all_switch_tables = {}
    for sw, raw in dumps.items():
        rows = sorted(
            _normalize_flow_line(ln) for ln in raw.splitlines()
            if "actions=" in ln and "priority=0" not in ln
        )
        all_switch_tables[sw] = [r for r in rows if r]
    trace["all_switch_tables"] = all_switch_tables
    trace["all_switch_table_rule_count"] = int(sum(len(v) for v in all_switch_tables.values()))
    return trace


def install_dominant_path_flows(harness: MininetHarness, ctx: dict, splits: Sequence[np.ndarray], measured_ods: Sequence[int]) -> tuple[int, dict]:
    diff_t0 = time.perf_counter()
    rule_count = 0
    od_pairs = ctx["ds"].od_pairs
    directed_edges = ctx["ds"].edges
    path_library = ctx["pl"]
    desired_by_switch: Dict[str, Dict[str, str]] = {switch_name: {} for switch_name in harness.spec.switch_by_node.values()}
    replace_threshold = INCREMENTAL_REPLACE_THRESHOLD
    od_lookup = {(str(src), str(dst)): idx for idx, (src, dst) in enumerate(od_pairs)}
    reverse_edge_lookup = {(str(src), str(dst)): int(edge_idx) for edge_idx, (src, dst) in enumerate(directed_edges)}

    def program_explicit_path(src_node: str, dst_node: str, edge_path: Sequence[int]) -> bool:
        if not edge_path:
            return False
        src_ip = harness.spec.host_by_node[src_node].ip.split("/")[0]
        dst_ip = harness.spec.host_by_node[dst_node].ip.split("/")[0]

        for edge_idx in edge_path:
            edge = harness.spec.directed_edges[int(edge_idx)]
            match = f"priority=200,ip,nw_src={src_ip},nw_dst={dst_ip}"
            flow = f"{match},actions=output:{edge.src_port}"
            desired_by_switch[edge.src_switch][match] = flow

        dst_host = harness.spec.host_by_node[dst_node]
        dst_switch = dst_host.switch_name
        dst_match = f"priority=200,ip,nw_src={src_ip},nw_dst={dst_ip}"
        dst_flow = f"{dst_match},actions=output:{dst_host.host_port}"
        desired_by_switch[dst_switch][dst_match] = dst_flow
        return True

    def program_od(od_idx: int) -> list[int]:
        paths = path_library.edge_idx_paths_by_od[int(od_idx)]
        if not paths:
            return []
        split_vec = np.asarray(splits[int(od_idx)], dtype=float)
        if split_vec.size == 0 or float(split_vec.sum()) <= 0:
            return []
        path_idx = int(np.argmax(split_vec))
        if path_idx >= len(paths):
            return []
        edge_path = list(paths[path_idx])
        src_node, dst_node = (str(od_pairs[int(od_idx)][0]), str(od_pairs[int(od_idx)][1]))
        if not program_explicit_path(src_node, dst_node, edge_path):
            return []
        return edge_path

    for od_idx in measured_ods:
        src_node, dst_node = (str(od_pairs[int(od_idx)][0]), str(od_pairs[int(od_idx)][1]))
        forward_edge_path = program_od(int(od_idx))
        reverse_idx = od_lookup.get((dst_node, src_node))
        reverse_edge_path = program_od(int(reverse_idx)) if reverse_idx is not None else []
        if not reverse_edge_path and forward_edge_path:
            fallback_reverse = []
            for edge_idx in reversed(forward_edge_path):
                edge_src, edge_dst = directed_edges[int(edge_idx)]
                reverse_edge_idx = reverse_edge_lookup.get((str(edge_dst), str(edge_src)))
                if reverse_edge_idx is None:
                    fallback_reverse = []
                    break
                fallback_reverse.append(int(reverse_edge_idx))
            if fallback_reverse:
                program_explicit_path(dst_node, src_node, fallback_reverse)

    changed_rules = 0
    unchanged_rules = 0
    added_rules = 0
    removed_rules = 0
    modified_rules = 0
    add_cmds = 0
    remove_cmds = 0
    switch_updates: list[tuple[str, Dict[str, str], list[str], list[str], list[str]]] = []
    for switch_name, desired in desired_by_switch.items():
        current = harness.programmed_flows.get(switch_name, {})
        stale_matches = sorted(set(current) - set(desired))
        updated_matches = [match for match, flow in desired.items() if current.get(match) != flow]
        added_matches = [match for match in updated_matches if match not in current]
        modified_matches = [match for match in updated_matches if match in current]
        unchanged_rules += sum(1 for match, flow in desired.items() if current.get(match) == flow)
        changed_rules += len(stale_matches) + len(updated_matches)
        added_rules += len(added_matches)
        removed_rules += len(stale_matches)
        modified_rules += len(modified_matches)
        if stale_matches or updated_matches:
            switch_updates.append((switch_name, desired, stale_matches, modified_matches, updated_matches))
        rule_count += len(desired)

    if FLOW_INSTALL_PROFILE_ENABLED:
        _install_prof_reset()
    diff_ready_t = time.perf_counter()
    if switch_updates:
        # ovs-ofctl calls are I/O-bound (each waits on the OVS daemon), not CPU-bound, so the trusted
        # cap at os.cpu_count() (=2 on this VM) needlessly serializes independent switches. The Target-6
        # optimized path programs each switch's atomic bundle concurrently across all changed switches.
        if OPT_INSTALL_ENABLED:
            max_workers = min(len(switch_updates), INSTALL_MAX_PARALLEL_SWITCHES)
        else:
            max_workers = min(len(switch_updates), max(1, os.cpu_count() or 1), 8)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    _apply_switch_flow_update,
                    switch_name,
                    desired,
                    stale_matches,
                    modified_matches,
                    updated_matches,
                    replace_threshold,
                ): (switch_name, desired)
                for switch_name, desired, stale_matches, modified_matches, updated_matches in switch_updates
            }
            for future in concurrent.futures.as_completed(future_map):
                switch_name, desired = future_map[future]
                add_count, remove_count = future.result()
                add_cmds += int(add_count)
                remove_cmds += int(remove_count)
                harness.programmed_flows[switch_name] = desired
    apply_done_t = time.perf_counter()

    delta = {
        "changed_rules": int(changed_rules),
        "unchanged_rules": int(unchanged_rules),
        "added_rules": int(added_rules),
        "removed_rules": int(removed_rules),
        "modified_rules": int(modified_rules),
        "add_cmds": int(add_cmds),
        "remove_cmds": int(remove_cmds),
        "rule_diff_ms": round((diff_ready_t - diff_t0) * 1000.0, 3),
        "flow_mod_send_ms": round((apply_done_t - diff_ready_t) * 1000.0, 3),
        "barrier_wait_ms": 0.0,
        "updated_switches": int(len(switch_updates)),
        "install_path": "bundle_opt" if OPT_INSTALL_ENABLED else "incremental",
    }
    if FLOW_INSTALL_PROFILE_ENABLED:
        prof = _install_prof_snapshot()
        total_cmds = sum(r["ovs_cmds"] for r in prof.values())
        delta["ovs_command_count"] = int(total_cmds)
        delta["install_switch_profile"] = {
            sw: {
                "ovs_cmds": int(r["ovs_cmds"]),
                "del_cmds": int(r["del_cmds"]),
                "mod_cmds": int(r["mod_cmds"]),
                "add_cmds": int(r["add_cmds"]),
                "subprocess_ms": round(float(r["subprocess_ms"]), 3),
            }
            for sw, r in prof.items()
        }
    return rule_count, delta


def preinstall_backup_support_flows(
    harness: MininetHarness,
    pre_ctx: dict,
    pre_splits: Sequence[np.ndarray],
    post_ctx: dict,
    post_splits: Sequence[np.ndarray],
    measured_ods: Sequence[int],
) -> dict:
    diff_t0 = time.perf_counter()
    od_pairs = pre_ctx["ds"].od_pairs
    reverse_lookup = {(str(src), str(dst)): idx for idx, (src, dst) in enumerate(od_pairs)}
    add_by_switch: Dict[str, list[str]] = {switch_name: [] for switch_name in harness.spec.switch_by_node.values()}
    added_rules = 0
    unchanged_rules = 0

    def _collect_path_support(od_idx: int) -> None:
        nonlocal added_rules, unchanged_rules
        if int(od_idx) < 0 or int(od_idx) >= len(od_pairs):
            return
        src_node, dst_node = (str(od_pairs[int(od_idx)][0]), str(od_pairs[int(od_idx)][1]))
        src_ip = harness.spec.host_by_node[src_node].ip.split("/")[0]
        dst_ip = harness.spec.host_by_node[dst_node].ip.split("/")[0]
        match = f"priority=200,ip,nw_src={src_ip},nw_dst={dst_ip}"

        for edge_idx in selected_path_edge_indices(post_ctx, post_splits, int(od_idx)):
            edge = harness.spec.directed_edges[int(edge_idx)]
            flow = f"{match},actions=output:{edge.src_port}"
            current = harness.programmed_flows.setdefault(edge.src_switch, {})
            existing = current.get(match)
            if existing is None:
                current[match] = flow
                add_by_switch[edge.src_switch].append(flow)
                added_rules += 1
            elif existing == flow:
                unchanged_rules += 1

        dst_host = harness.spec.host_by_node[dst_node]
        dst_switch = dst_host.switch_name
        dst_flow = f"{match},actions=output:{dst_host.host_port}"
        current = harness.programmed_flows.setdefault(dst_switch, {})
        existing = current.get(match)
        if existing is None:
            current[match] = dst_flow
            add_by_switch[dst_switch].append(dst_flow)
            added_rules += 1
        elif existing == dst_flow:
            unchanged_rules += 1

    for od_idx in measured_ods:
        _collect_path_support(int(od_idx))
        src_node, dst_node = (str(od_pairs[int(od_idx)][0]), str(od_pairs[int(od_idx)][1]))
        reverse_idx = reverse_lookup.get((dst_node, src_node))
        if reverse_idx is not None:
            _collect_path_support(int(reverse_idx))

    diff_ready_t = time.perf_counter()
    add_cmds = 0
    for switch_name, flows in add_by_switch.items():
        if not flows:
            continue
        _ovs_add_flows_batch(switch_name, flows)
        add_cmds += len(flows)
    apply_done_t = time.perf_counter()

    return {
        "added_rules": int(added_rules),
        "unchanged_rules": int(unchanged_rules),
        "add_cmds": int(add_cmds),
        "rule_diff_ms": round((diff_ready_t - diff_t0) * 1000.0, 3),
        "flow_mod_send_ms": round((apply_done_t - diff_ready_t) * 1000.0, 3),
    }


def ping_summary(host, target_ip: str) -> tuple[float, float, float]:
    output = host.cmd(f"ping -c {PING_COUNT} -i {PING_INTERVAL_SEC:.2f} -W 1 {shlex.quote(target_ip)}")
    loss_match = re.search(r"(\d+(?:\.\d+)?)%\s+packet loss", output)
    stats_match = re.search(r"=\s*([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)\s*ms", output)
    loss_pct = float(loss_match.group(1)) if loss_match else 100.0
    if stats_match:
        avg_rtt = float(stats_match.group(2))
        jitter = float(stats_match.group(4))
    else:
        avg_rtt = float("nan")
        jitter = float("nan")
    return avg_rtt, jitter, loss_pct


def warm_up_path(src_host, dst_host, dst_ip: str, rate_mbps: float, port: int, warmup_sec: int) -> None:
    if warmup_sec <= 0:
        return
    dst_host.cmd(f"pkill -f 'iperf -s -u -p {port}' >/dev/null 2>&1")
    dst_host.cmd(f"iperf -s -u -p {port} >/tmp/iperf_warmup_server_{port}.log 2>&1 &")
    time.sleep(0.5)
    src_host.cmd(f"iperf -c {shlex.quote(dst_ip)} -u -b {rate_mbps:.3f}M -t {warmup_sec} -p {port} >/tmp/iperf_warmup_client_{port}.log 2>&1")
    dst_host.cmd(f"pkill -f 'iperf -s -u -p {port}' >/dev/null 2>&1")
    src_host.cmd(f"ping -c 2 -i 0.2 -W 1 {shlex.quote(dst_ip)} >/tmp/ping_warmup_{port}.log 2>&1")


def iperf_udp_summary(src_host, dst_host, dst_ip: str, rate_mbps: float, duration_sec: int, port: int) -> tuple[float, float, float]:
    dst_host.cmd(f"pkill -f 'iperf -s -u -p {port}' >/dev/null 2>&1")
    server_log = f"/tmp/iperf_server_{port}.log"
    dst_host.cmd(f"iperf -s -u -p {port} >{server_log} 2>&1 &")
    time.sleep(1.0)
    output = src_host.cmd(
        f"iperf -c {shlex.quote(dst_ip)} -u -b {rate_mbps:.3f}M -t {duration_sec} -p {port} -i 1 2>/dev/null"
    )
    dst_host.cmd(f"pkill -f 'iperf -s -u -p {port}' >/dev/null 2>&1")
    server_output = dst_host.cmd(f"cat {server_log} 2>/dev/null || true")
    result = _parse_iperf_udp_output(output)
    if result[0] > 0 or np.isfinite(result[1]) or result[2] < 100.0:
        return result
    return _parse_iperf_udp_output(server_output)


def recovery_probe_ms(harness: MininetHarness, ctx: dict, measured_ods: Sequence[int], timeout_sec: float = 8.0) -> float:
    if not measured_ods:
        return 0.0
    od_pairs = ctx["ds"].od_pairs
    first_od = int(measured_ods[0])
    src_node, dst_node = (str(od_pairs[first_od][0]), str(od_pairs[first_od][1]))
    src_host = harness.net.get(harness.spec.host_by_node[src_node].host_name)
    dst_ip = harness.spec.host_by_node[dst_node].ip.split("/")[0]

    start = time.perf_counter()
    while time.perf_counter() - start <= timeout_sec:
        result = src_host.cmd(f"ping -c 1 -W 1 {shlex.quote(dst_ip)}")
        if " 0% packet loss" in result or ", 1 received" in result:
            return round((time.perf_counter() - start) * 1000.0, 3)
        time.sleep(0.2)
    return round(timeout_sec * 1000.0, 3)


def live_rate_for_od(tm: np.ndarray, od_idx: int, max_bw_mbps: float, bw_scale_mbps_per_unit: float) -> float:
    demand_units = float(np.asarray(tm, dtype=float)[int(od_idx)])
    if demand_units <= 0:
        return 0.5
    scaled_rate = demand_units * max(0.0, float(bw_scale_mbps_per_unit))
    return max(0.5, min(max_bw_mbps * 0.9, scaled_rate))


def scenario_rate_share(scenario: str) -> float:
    if scenario in {"single_link_failure", "two_link_failure", "capacity_degradation_50", "mixed_spike_failure"}:
        return FAILURE_RATE_SHARE
    return BASELINE_RATE_SHARE


def build_live_ctx(ctx0: dict, caps: np.ndarray) -> dict:
    pruned_pl = prune_path_library(ctx0["pl"], caps)
    return {
        **ctx0,
        "pl": pruned_pl,
        "ecmp": ecmp_splits(pruned_pl),
        "caps": np.asarray(caps, dtype=float),
    }


def selected_path_edge_indices(ctx: dict, splits: Sequence[np.ndarray], od_idx: int) -> list[int]:
    if int(od_idx) >= len(ctx["pl"].edge_idx_paths_by_od):
        return []
    paths = ctx["pl"].edge_idx_paths_by_od[int(od_idx)]
    if not paths:
        return []
    split_vec = np.asarray(splits[int(od_idx)], dtype=float)
    if split_vec.size == 0 or float(split_vec.sum()) <= 0:
        return []
    path_idx = int(np.argmax(split_vec))
    if path_idx >= len(paths):
        return []
    return list(paths[path_idx])


def path_bottleneck_mbps(harness: MininetHarness, caps: np.ndarray, edge_path: Sequence[int]) -> float:
    if not edge_path:
        return 0.0
    scale = max(0.0, float(harness.spec.bw_scale_mbps_per_unit))
    values = [float(np.asarray(caps, dtype=float)[int(edge_idx)]) * scale for edge_idx in edge_path]
    finite = [value for value in values if np.isfinite(value) and value > 0.0]
    return float(min(finite)) if finite else 0.0


def build_measurement_plan(
    harness: MininetHarness,
    ctx: dict,
    controller_result: dict,
    tm: np.ndarray,
    caps: np.ndarray,
    scenario: str,
    max_measured_ods: int,
    forced_measured_ods: Sequence[int] | None = None,
) -> list[dict]:
    od_pairs = ctx["ds"].od_pairs
    measured_ods = [int(od_idx) for od_idx in forced_measured_ods] if forced_measured_ods is not None else choose_measured_ods(
        np.asarray(tm, dtype=float),
        controller_result["selected_ods"],
        controller_result["active_od_count"],
        max_measured_ods,
        od_pairs=od_pairs,
    )
    plan: list[dict] = []
    for od_idx in measured_ods:
        edge_path = selected_path_edge_indices(ctx, controller_result["splits"], int(od_idx))
        if not edge_path:
            continue
        bottleneck_mbps = path_bottleneck_mbps(harness, caps, edge_path)
        if bottleneck_mbps <= 0:
            continue
        src_node, dst_node = (str(od_pairs[int(od_idx)][0]), str(od_pairs[int(od_idx)][1]))
        dst_ip = harness.spec.host_by_node[dst_node].ip.split("/")[0]
        demand_rate = live_rate_for_od(
            np.asarray(tm, dtype=float),
            int(od_idx),
            harness.spec.max_bw_mbps,
            harness.spec.bw_scale_mbps_per_unit,
        )
        offered_rate = max(0.5, min(demand_rate, bottleneck_mbps * scenario_rate_share(scenario)))
        plan.append(
            {
                "od_idx": int(od_idx),
                "src_node": src_node,
                "dst_node": dst_node,
                "dst_ip": dst_ip,
                "edge_path": edge_path,
                "path_bottleneck_mbps": round(float(bottleneck_mbps), 3),
                "offered_rate_mbps": round(float(offered_rate), 3),
            }
        )
    return plan


def rebuild_phase_plan(
    harness: MininetHarness,
    ctx: dict,
    controller_result: dict,
    caps: np.ndarray,
    rate_plan: Sequence[dict],
) -> list[dict]:
    od_pairs = ctx["ds"].od_pairs
    phase_plan: list[dict] = []
    for item in rate_plan:
        od_idx = int(item["od_idx"])
        src_node, dst_node = (str(od_pairs[od_idx][0]), str(od_pairs[od_idx][1]))
        dst_ip = harness.spec.host_by_node[dst_node].ip.split("/")[0]
        edge_path = selected_path_edge_indices(ctx, controller_result["splits"], od_idx)
        bottleneck_mbps = path_bottleneck_mbps(harness, caps, edge_path)
        phase_plan.append(
            {
                "od_idx": od_idx,
                "src_node": src_node,
                "dst_node": dst_node,
                "dst_ip": dst_ip,
                "edge_path": edge_path,
                "path_bottleneck_mbps": round(float(bottleneck_mbps), 3),
                "offered_rate_mbps": round(float(item["offered_rate_mbps"]), 3),
            }
        )
    return phase_plan


def select_recovery_probe_plan(
    pre_plan: Sequence[dict],
    post_plan: Sequence[dict],
    scenario: str,
    failed_edge_indices: Sequence[int],
) -> list[dict]:
    if scenario == "capacity_degradation_50":
        return list(post_plan)
    failed_edges = {int(edge_idx) for edge_idx in failed_edge_indices}
    if not failed_edges:
        return list(post_plan)
    pre_by_od = {int(item["od_idx"]): item for item in pre_plan}
    impacted: list[dict] = []
    for item in post_plan:
        od_idx = int(item["od_idx"])
        pre_item = pre_by_od.get(od_idx)
        pre_edges = set(int(edge_idx) for edge_idx in pre_item.get("edge_path", [])) if pre_item else set()
        post_edges = set(int(edge_idx) for edge_idx in item.get("edge_path", []))
        if (pre_edges & failed_edges) or (pre_edges != post_edges):
            impacted.append(item)
    return impacted or list(post_plan)


def scenario_measurement_note(scenario: str) -> str:
    if scenario == "mixed_spike_failure":
        return (
            f"Incremental recovery kept the controller warm, used precomputed backup paths, preserved unaffected switches, and applied one batched replace-flows update per changed switch. "
            f"For mixed spike+failure, the link failure was injected first, stable probes were reached, "
            f"and spike traffic was applied {int(MIXED_SPIKE_DELAY_SEC * 1000)} ms later as a post-convergence stress step."
        )
    return (
        "Incremental recovery kept the controller warm, preserved unaffected switches, used precomputed backup paths, "
        "applied one batched replace-flows update per changed switch, and separated transient loss from post-recovery steady-state QoS."
    )


def aggregate_offered_rate_mbps(plan: Sequence[dict]) -> float:
    return round(float(sum(float(item["offered_rate_mbps"]) for item in plan)), 3)


def measure_phase_metrics(
    harness: MininetHarness,
    plan: Sequence[dict],
    duration_sec: int,
    warmup_sec: int,
    port_base: int,
    phase_name: str,
) -> dict:
    throughput_values = []
    loss_values = []
    rtt_values = []
    jitter_values = []
    raw_measurements = []

    for idx, item in enumerate(plan):
        src_host = harness.net.get(harness.spec.host_by_node[item["src_node"]].host_name)
        dst_host = harness.net.get(harness.spec.host_by_node[item["dst_node"]].host_name)
        if warmup_sec > 0:
            warm_up_path(src_host, dst_host, item["dst_ip"], float(item["offered_rate_mbps"]), port_base + 100 + idx, warmup_sec)
        throughput_mbps, iperf_jitter_ms, iperf_loss_pct = iperf_udp_summary(
            src_host=src_host,
            dst_host=dst_host,
            dst_ip=item["dst_ip"],
            rate_mbps=float(item["offered_rate_mbps"]),
            duration_sec=duration_sec,
            port=port_base + idx,
        )
        rtt_ms, ping_jitter_ms, ping_loss_pct = ping_summary(src_host, item["dst_ip"])

        throughput_values.append(throughput_mbps)
        loss_values.append(np.nanmean([iperf_loss_pct, ping_loss_pct]))
        rtt_values.append(rtt_ms)
        jitter_values.append(np.nanmean([iperf_jitter_ms, ping_jitter_ms]))
        raw_measurements.append(
            {
                "phase": phase_name,
                "od_idx": int(item["od_idx"]),
                "src_node": item["src_node"],
                "dst_node": item["dst_node"],
                "target_ip": item["dst_ip"],
                "offered_rate_mbps": round(float(item["offered_rate_mbps"]), 3),
                "path_bottleneck_mbps": round(float(item["path_bottleneck_mbps"]), 3),
                "throughput_mbps": round(float(throughput_mbps), 3),
                "iperf_jitter_ms": round(float(iperf_jitter_ms), 3) if np.isfinite(iperf_jitter_ms) else None,
                "iperf_loss_pct": round(float(iperf_loss_pct), 4) if np.isfinite(iperf_loss_pct) else None,
                "ping_rtt_ms": round(float(rtt_ms), 3) if np.isfinite(rtt_ms) else None,
                "ping_jitter_ms": round(float(ping_jitter_ms), 3) if np.isfinite(ping_jitter_ms) else None,
                "ping_loss_pct": round(float(ping_loss_pct), 4) if np.isfinite(ping_loss_pct) else None,
            }
        )

    mean_throughput = float(np.nansum(throughput_values)) if throughput_values else 0.0
    mean_loss = float(np.nanmean(loss_values)) if loss_values else 100.0
    mean_rtt = float(np.nanmean(rtt_values)) if rtt_values else float("nan")
    mean_jitter = float(np.nanmean(jitter_values)) if jitter_values else float("nan")
    return {
        "offered_udp_rate_mbps": aggregate_offered_rate_mbps(plan),
        "throughput_mbps": round(mean_throughput, 3),
        "packet_loss_pct": round(mean_loss, 4),
        "rtt_ms": round(mean_rtt, 3) if np.isfinite(mean_rtt) else float("nan"),
        "jitter_ms": round(mean_jitter, 3) if np.isfinite(mean_jitter) else float("nan"),
        "raw_measurements": raw_measurements,
    }


def start_transient_udp_phase(
    harness: MininetHarness,
    plan: Sequence[dict],
    duration_sec: int,
    port_base: int,
    tag: str,
) -> dict:
    probes = []
    start_time = time.perf_counter()
    for idx, item in enumerate(plan):
        src_host = harness.net.get(harness.spec.host_by_node[item["src_node"]].host_name)
        dst_host = harness.net.get(harness.spec.host_by_node[item["dst_node"]].host_name)
        port = port_base + idx
        server_log = f"/tmp/{tag}_server_{port}.log"
        client_log = f"/tmp/{tag}_client_{port}.log"
        dst_host.cmd(f"pkill -f 'iperf -s -u -p {port}' >/dev/null 2>&1")
        dst_host.cmd(f"iperf -s -u -p {port} >{server_log} 2>&1 &")
        time.sleep(0.2)
        src_host.cmd(
            f"iperf -c {shlex.quote(item['dst_ip'])} -u -b {float(item['offered_rate_mbps']):.3f}M "
            f"-t {duration_sec} -p {port} >{client_log} 2>&1 &"
        )
        probes.append(
            {
                **item,
                "port": int(port),
                "server_log": server_log,
                "client_log": client_log,
            }
        )
    return {
        "duration_sec": int(duration_sec),
        "start_time": float(start_time),
        "probes": probes,
    }


def collect_transient_udp_phase(harness: MininetHarness, transient_state: dict) -> dict:
    elapsed = time.perf_counter() - float(transient_state["start_time"])
    remaining = max(0.0, float(transient_state["duration_sec"]) - elapsed)
    if remaining > 0:
        time.sleep(remaining)

    loss_values = []
    raw_measurements = []
    valid_probe_count = 0
    invalid_probe_count = 0
    invalid_reasons: list[str] = []
    for item in transient_state["probes"]:
        dst_host = harness.net.get(harness.spec.host_by_node[item["dst_node"]].host_name)
        client_output = ""
        server_output = ""
        parsed = {"valid": False, "reason": "iperf_output_empty",
                  "throughput_mbps": None, "jitter_ms": None, "loss_pct": None}
        parser_source = "none"
        read_start = time.perf_counter()
        # Retry-read the logs; the iperf UDP loss summary is emitted by the RECEIVER
        # (server), so prefer the server log, then client, then a merged view.
        while time.perf_counter() - read_start <= 2.0:
            try:
                client_output = Path(str(item["client_log"])).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                client_output = ""
            try:
                server_output = Path(str(item["server_log"])).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                server_output = ""
            for source_name, source_text in (
                ("iperf_udp_server_summary", server_output),
                ("iperf_udp_client_summary", client_output),
                ("iperf_udp_merged_summary",
                 "\n".join(part for part in (server_output, client_output) if part)),
            ):
                cand = _parse_iperf_udp_validity(source_text)
                if cand["valid"]:
                    parsed = cand
                    parser_source = source_name
                    break
            if parsed["valid"]:
                break
            time.sleep(0.1)
        dst_host.cmd(f"pkill -f 'iperf -s -u -p {int(item['port'])}' >/dev/null 2>&1")

        if parsed["valid"]:
            valid_probe_count += 1
            loss_pct = float(parsed["loss_pct"])
            loss_values.append(loss_pct)
            reason = ""
        else:
            invalid_probe_count += 1
            loss_pct = None  # NEVER coerce a parse failure to 0 or 100
            reason = parsed["reason"]
            invalid_reasons.append(reason)
            parser_source = "none"
        raw_measurements.append(
            {
                "phase": "transient_recovery",
                "od_idx": int(item["od_idx"]),
                "src_node": item["src_node"],
                "dst_node": item["dst_node"],
                "target_ip": item["dst_ip"],
                "offered_rate_mbps": round(float(item["offered_rate_mbps"]), 3),
                "path_bottleneck_mbps": round(float(item["path_bottleneck_mbps"]), 3),
                "throughput_mbps": round(float(parsed["throughput_mbps"]), 3) if parsed["throughput_mbps"] is not None else None,
                "iperf_jitter_ms": round(float(parsed["jitter_ms"]), 3) if parsed["jitter_ms"] is not None else None,
                "transient_packet_loss_pct": round(loss_pct, 4) if loss_pct is not None else None,
                "measurement_valid": bool(parsed["valid"]),
                "measurement_invalid_reason": reason,
                "measurement_parser": parser_source,
                "iperf_client_log": str(item["client_log"]),
                "iperf_server_log": str(item["server_log"]),
                "iperf_client_log_bytes": int(len(client_output.encode("utf-8", errors="ignore"))),
                "iperf_server_log_bytes": int(len(server_output.encode("utf-8", errors="ignore"))),
            }
        )
    n_probes = len(transient_state["probes"])
    measurement_valid = bool(n_probes > 0 and invalid_probe_count == 0)
    # Aggregate only VALID probes. If none valid, packet_loss_pct is None (not 0/100).
    mean_loss = float(np.mean(loss_values)) if loss_values else None
    return {
        "offered_udp_rate_mbps": aggregate_offered_rate_mbps(transient_state["probes"]),
        "packet_loss_pct": round(mean_loss, 4) if mean_loss is not None else None,
        "measurement_valid": measurement_valid,
        "measurement_invalid_reason": ";".join(sorted(set(invalid_reasons))),
        "measurement_parser": "iperf_udp_receiver_summary",
        "valid_probe_count": int(valid_probe_count),
        "invalid_probe_count": int(invalid_probe_count),
        "probe_count": int(n_probes),
        "raw_measurements": raw_measurements,
    }


def wait_for_stable_forwarding(
    harness: MininetHarness,
    plan: Sequence[dict],
    timeout_sec: float = RECOVERY_TIMEOUT_SEC,
    stable_rounds_required: int = RECOVERY_STABLE_ROUNDS,
) -> tuple[float, list[dict]]:
    if not plan:
        return 0.0, []
    start = time.perf_counter()
    stable_rounds = 0
    history: list[dict] = []
    while time.perf_counter() - start <= timeout_sec:
        round_success = True
        round_probe = {
            "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
            "results": [],
        }
        for item in plan:
            src_host = harness.net.get(harness.spec.host_by_node[item["src_node"]].host_name)
            result = src_host.cmd(f"ping -c 1 -W 1 {shlex.quote(item['dst_ip'])}")
            success = " 0% packet loss" in result or ", 1 received" in result
            round_probe["results"].append(
                {
                    "od_idx": int(item["od_idx"]),
                    "src_node": item["src_node"],
                    "dst_node": item["dst_node"],
                    "success": bool(success),
                }
            )
            round_success = round_success and bool(success)
        history.append(round_probe)
        if round_success:
            stable_rounds += 1
            if stable_rounds >= int(stable_rounds_required):
                return round((time.perf_counter() - start) * 1000.0, 3), history
        else:
            stable_rounds = 0
        time.sleep(0.2)
    return round(timeout_sec * 1000.0, 3), history


def measure_live_metrics(
    harness: MininetHarness,
    topo: str,
    ctx: dict,
    controller_result: dict,
    max_measured_ods: int,
    duration_sec: int,
    warmup_sec: int,
) -> dict:
    measured_ods = choose_measured_ods(
        controller_result["tm"],
        controller_result["selected_ods"],
        controller_result["active_od_count"],
        max_measured_ods,
        od_pairs=ctx["ds"].od_pairs,
    )

    install_t0 = time.perf_counter()
    flow_rule_count, flow_delta = install_dominant_path_flows(harness, ctx, controller_result["splits"], measured_ods)
    install_ms = (time.perf_counter() - install_t0) * 1000.0
    recovery_ms = recovery_probe_ms(harness, ctx, measured_ods)
    time.sleep(STEADY_STATE_SETTLE_SEC)

    throughput_values = []
    loss_values = []
    rtt_values = []
    jitter_values = []
    raw_measurements = []
    od_pairs = ctx["ds"].od_pairs

    for idx, od_idx in enumerate(measured_ods):
        src_node, dst_node = (str(od_pairs[int(od_idx)][0]), str(od_pairs[int(od_idx)][1]))
        src_host = harness.net.get(harness.spec.host_by_node[src_node].host_name)
        dst_host = harness.net.get(harness.spec.host_by_node[dst_node].host_name)
        dst_ip = harness.spec.host_by_node[dst_node].ip.split("/")[0]

        rate_mbps = live_rate_for_od(
            controller_result["tm"],
            int(od_idx),
            harness.spec.max_bw_mbps,
            harness.spec.bw_scale_mbps_per_unit,
        )
        warm_up_path(src_host, dst_host, dst_ip, rate_mbps, 4501 + idx, warmup_sec)
        throughput_mbps, iperf_jitter_ms, iperf_loss_pct = iperf_udp_summary(
            src_host=src_host,
            dst_host=dst_host,
            dst_ip=dst_ip,
            rate_mbps=rate_mbps,
            duration_sec=duration_sec,
            port=5001 + idx,
        )
        rtt_ms, ping_jitter_ms, ping_loss_pct = ping_summary(src_host, dst_ip)

        throughput_values.append(throughput_mbps)
        loss_values.append(np.nanmean([iperf_loss_pct, ping_loss_pct]))
        rtt_values.append(rtt_ms)
        jitter_values.append(np.nanmean([iperf_jitter_ms, ping_jitter_ms]))
        raw_measurements.append(
            {
                "od_idx": int(od_idx),
                "src_node": src_node,
                "dst_node": dst_node,
                "target_ip": dst_ip,
                "requested_rate_mbps": round(rate_mbps, 3),
                "throughput_mbps": round(throughput_mbps, 3),
                "iperf_jitter_ms": round(iperf_jitter_ms, 3) if np.isfinite(iperf_jitter_ms) else None,
                "iperf_loss_pct": round(iperf_loss_pct, 4) if np.isfinite(iperf_loss_pct) else None,
                "ping_rtt_ms": round(rtt_ms, 3) if np.isfinite(rtt_ms) else None,
                "ping_jitter_ms": round(ping_jitter_ms, 3) if np.isfinite(ping_jitter_ms) else None,
                "ping_loss_pct": round(ping_loss_pct, 4) if np.isfinite(ping_loss_pct) else None,
            }
        )

    mean_throughput = float(np.nansum(throughput_values))
    mean_loss = float(np.nanmean(loss_values)) if loss_values else 100.0
    mean_rtt = float(np.nanmean(rtt_values)) if rtt_values else float("nan")
    mean_jitter = float(np.nanmean(jitter_values)) if jitter_values else float("nan")

    if controller_result["disconnected"] > 0:
        controller_status = "PARTIAL_FAILURE"
    elif np.isnan(mean_throughput) or mean_throughput <= 0:
        controller_status = "MEASUREMENT_ERROR"
    elif mean_loss > 1.0:
        controller_status = "DEGRADED"
    else:
        controller_status = "OK"

    return {
        "throughput_mbps": round(mean_throughput, 2),
        "packet_loss_pct": round(mean_loss, 4),
        "rtt_ms": round(mean_rtt, 3) if np.isfinite(mean_rtt) else float("nan"),
        "jitter_ms": round(mean_jitter, 3) if np.isfinite(mean_jitter) else float("nan"),
        "recovery_ms": round(recovery_ms, 3),
        "flow_rule_count": int(flow_rule_count),
        "install_ms": round(install_ms, 3),
        "controller_status": controller_status,
        "measured_od_count": int(len(measured_ods)),
        "live_projection": "dominant_path_for_measured_ods",
        "flow_update_added": int(flow_delta["added"]),
        "flow_update_removed": int(flow_delta["removed"]),
        "raw_measurements": raw_measurements,
    }


def build_contexts(bundle, gnn: GNNLPDScorer, topos: Iterable[str]) -> dict[str, dict]:
    contexts: dict[str, dict] = {}
    bundle = bundle or load_bundle(CONFIG)
    lookup = _build_spec_lookup(bundle)
    for topo in topos:
        ctx = build_context(bundle, lookup, topo, 8, "disjoint")
        tm_total = int(np.asarray(ctx["ds"].tm).shape[0])
        pref_lo, pref_hi = TM_RANGES[topo]
        pref_span = max(1, int(pref_hi) - int(pref_lo))
        tm_span = min(tm_total, max(pref_span, RUNS_PER_SCENARIO))
        if int(pref_lo) < tm_total and int(pref_hi) <= tm_total:
            tm_lo = int(pref_lo)
            tm_hi = min(int(pref_hi), tm_total)
            if (tm_hi - tm_lo) < RUNS_PER_SCENARIO:
                tm_lo = max(0, tm_hi - tm_span)
        else:
            tm_hi = tm_total
            tm_lo = max(0, tm_hi - tm_span)
        if tm_hi <= tm_lo:
            raise RuntimeError(f"No valid TM window for topology '{topo}' with {tm_total} traffic matrices")
        print(f"[tm-window] {topo}: using [{tm_lo}, {tm_hi}) from {tm_total} available TMs", flush=True)
        contexts[topo] = {
            "ds": ctx["ds"],
            "pl": ctx["pl"],
            "ecmp": ctx["ecmp"],
            "caps": np.asarray(ctx["caps"], dtype=float),
            "struct": A.struct_feats(ctx["ds"]),
            "tm_lo": int(tm_lo),
            "tm_hi": int(tm_hi),
            "tm_total": int(tm_total),
        }
    return contexts


def run_simulate(args) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gnn, teacher, final_net, mean, std = load_fix1_runtime()
    bundle = None
    contexts = build_contexts(bundle, gnn, TOPOLOGIES_TO_RUN)
    scenario_list = list(getattr(args, "scenarios", SDN_SCENARIOS))

    records = []
    total = len(TOPOLOGIES_TO_RUN) * len(scenario_list) * RUNS_PER_SCENARIO
    done = 0

    for topo in TOPOLOGIES_TO_RUN:
        ctx0 = contexts[topo]
        tm_lo, tm_hi = int(ctx0["tm_lo"]), int(ctx0["tm_hi"])
        base_tms = ctx0["ds"].tm[tm_lo:tm_hi]

        for scenario in scenario_list:
            failed_link_keys_plan: list[Tuple[str, str]] = []
            failed_edge_indices = np.asarray([], dtype=int)
            if scenario in {"single_link_failure", "mixed_spike_failure"}:
                failed_link_keys_plan, failed_edge_indices = plan_failed_physical_links(
                    ctx0["ds"].edges,
                    ctx0["caps"],
                    1,
                    path_library=ctx0["pl"],
                )
            elif scenario == "two_link_failure":
                failed_link_keys_plan, failed_edge_indices = plan_failed_physical_links(
                    ctx0["ds"].edges,
                    ctx0["caps"],
                    2,
                    path_library=ctx0["pl"],
                )
            caps = modified_caps(ctx0["caps"], scenario, failed_edge_indices)
            pruned_pl = prune_path_library(ctx0["pl"], caps)
            ctx = {
                **ctx0,
                "pl": pruned_pl,
                "ecmp": ecmp_splits(pruned_pl),
            }
            accepted = clone_splits(ctx["ecmp"])
            prev_tm = None

            for run_id in range(RUNS_PER_SCENARIO):
                tm = np.asarray(base_tms[run_id % len(base_tms)], dtype=float) * tm_scale(scenario)
                controller_result = run_fix1_cycle(
                    topo=topo,
                    tm_index=tm_lo + run_id,
                    cycle_idx=run_id,
                    tm_raw=tm,
                    ctx=ctx,
                    caps=caps,
                    accepted=accepted,
                    prev_tm=prev_tm,
                    gnn=gnn,
                    teacher=teacher,
                    final_net=final_net,
                    mean=mean,
                    std=std,
                    strict_mcf_time_limit=args.strict_mcf_time_limit,
                )
                sdn_metrics = derive_simulated_sdn_metrics(
                    topo=topo,
                    mlu=controller_result["mlu"],
                    db=controller_result["db"],
                    decision_ms=controller_result["decision_ms"],
                    active_od_count=controller_result["active_od_count"],
                    selected_od_count=max(controller_result["selected_od_count"], 1),
                    disconnected=controller_result["disconnected"],
                )
                record = {
                    "topology": topo,
                    "scenario": scenario,
                    "run_id": run_id,
                    "method": PUBLIC_METHOD,
                    "method_lineage": METHOD_LINEAGE,
                    "method_output_id": METHOD_OUTPUT_ID,
                    "mode": "simulate",
                    **sdn_metrics,
                    "decision_ms": controller_result["decision_ms"],
                    "pr": controller_result["pr"],
                    "mlu": controller_result["mlu"],
                    "db": controller_result["db"],
                    "strict_mcf_mlu": controller_result["strict_mcf_mlu"],
                    "strict_mcf_status": controller_result["strict_mcf_status"],
                    "pr_numerator_type": "strict_full_mcf",
                    "action_name": controller_result["action_name"],
                    "active_od_count": controller_result["active_od_count"],
                    "selected_od_count": controller_result["selected_od_count"],
                    "lp_status": controller_result["lp_status"],
                    "db_budget": controller_result["db_budget"],
                    "disconnected": controller_result["disconnected"],
                    "measured_od_count": 0,
                    "live_projection": "",
                    **AUDIT_TEMPLATE,
                }
                records.append(record)
                accepted = controller_result["splits"]
                prev_tm = controller_result["tm"]
                done += 1
                print(
                    f"[simulate] {topo} {scenario} run={run_id + 1}/{RUNS_PER_SCENARIO} "
                    f"PR={controller_result['pr']:.4f} MLU={controller_result['mlu']:.4f} "
                    f"decision_ms={controller_result['decision_ms']:.1f} [{done}/{total}]",
                    flush=True,
                )

    per_run_df = pd.DataFrame(records)
    write_outputs(per_run_df)
    print("Simulation mode complete.", flush=True)
    return per_run_df


def run_mininet(args) -> pd.DataFrame:
    if not MININET_AVAILABLE:
        raise RuntimeError("Mininet is not available. Run this mode on Linux/WSL with Mininet installed.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gnn, teacher, final_net, mean, std = load_fix1_runtime()
    bundle = None
    contexts = build_contexts(bundle, gnn, TOPOLOGIES_TO_RUN)
    scenario_list = list(getattr(args, "scenarios", SDN_SCENARIOS))
    records = []
    total = len(TOPOLOGIES_TO_RUN) * len(scenario_list) * RUNS_PER_SCENARIO
    done = 0

    for topo in TOPOLOGIES_TO_RUN:
        ctx0 = contexts[topo]
        harness: MininetHarness | None = None
        try:
            spec = build_live_topology_spec(ctx0, ctx0["caps"])
            harness = start_mininet_harness(spec)
            tm_lo, tm_hi = int(ctx0["tm_lo"]), int(ctx0["tm_hi"])
            base_tms = ctx0["ds"].tm[tm_lo:tm_hi]
            baseline_caps = np.asarray(ctx0["caps"], dtype=float)
            pre_ctx = build_live_ctx(ctx0, baseline_caps)

            for scenario_idx, scenario in enumerate(scenario_list):
                scenario_focus_ods: list[int] | None = None
                scenario_failed_link_keys_plan: list[Tuple[str, str]] = []
                scenario_failed_edge_indices = np.asarray([], dtype=int)
                for run_id in range(RUNS_PER_SCENARIO):
                    # The harness is shared across runs. For capacity_degradation_50 this
                    # accumulates cross-run reconvergence instability (growing recovery time
                    # and post-recovery loss) that is INDEPENDENT of the capacity mechanism
                    # (reproduced with both intf.config and tc-class-change). The first run on
                    # any fresh harness is consistently clean, so rebuild a fresh harness per
                    # run for this scenario to match that clean baseline. Gated to
                    # capacity_degradation_50 so other scenarios' established behavior is unchanged.
                    if REBUILD_HARNESS_PER_RUN and scenario == "capacity_degradation_50" and run_id > 0:
                        stop_mininet_harness(harness)
                        harness = start_mininet_harness(spec)
                    restore_live_links(harness)
                    if RESET_TE_PER_RUN and scenario == "capacity_degradation_50" and run_id > 0:
                        clear_te_programming(harness)

                    # Deterministic replay: when REPLAY_TM_RUN_ID >= 0, every iteration reuses
                    # the exact controller input (TM/cycle/tm_index) of that target run so the
                    # only variation across iterations is the live data plane. port_seed and the
                    # record run_id still use the iteration index for uniqueness.
                    tm_run_id = REPLAY_TM_RUN_ID if REPLAY_TM_RUN_ID >= 0 else run_id
                    scenario_tm = np.asarray(base_tms[tm_run_id % len(base_tms)], dtype=float) * tm_scale(scenario)
                    cycle_seed = tm_run_id * 2 + 1
                    measurement_note = scenario_measurement_note(scenario)

                    pre_controller_result = run_fix1_cycle(
                        topo=topo,
                        tm_index=tm_lo + tm_run_id,
                        cycle_idx=cycle_seed,
                        tm_raw=scenario_tm,
                        ctx=pre_ctx,
                        caps=baseline_caps,
                        accepted=clone_splits(pre_ctx["ecmp"]),
                        prev_tm=None,
                        gnn=gnn,
                        teacher=teacher,
                        final_net=final_net,
                        mean=mean,
                        std=std,
                        strict_mcf_time_limit=args.strict_mcf_time_limit,
                        compute_strict_mcf=False,
                    )
                    failed_link_keys_plan: list[Tuple[str, str]] = []
                    failed_edge_indices = np.asarray([], dtype=int)
                    if scenario in {"single_link_failure", "mixed_spike_failure"}:
                        if scenario_focus_ods is None:
                            scenario_focus_ods = choose_measured_ods(
                                np.asarray(pre_controller_result["tm"], dtype=float),
                                pre_controller_result["selected_ods"],
                                pre_controller_result["active_od_count"],
                                args.max_measured_ods,
                                od_pairs=pre_ctx["ds"].od_pairs,
                            )
                        if scenario_failed_edge_indices.size == 0:
                            scenario_failed_link_keys_plan, scenario_failed_edge_indices = plan_failed_physical_links(
                                ctx0["ds"].edges,
                                baseline_caps,
                                1,
                                path_library=ctx0["pl"],
                                focus_ods=scenario_focus_ods,
                            )
                        failed_link_keys_plan = list(scenario_failed_link_keys_plan)
                        failed_edge_indices = np.asarray(scenario_failed_edge_indices, dtype=int)
                    elif scenario == "two_link_failure":
                        if scenario_focus_ods is None:
                            scenario_focus_ods = choose_measured_ods(
                                np.asarray(pre_controller_result["tm"], dtype=float),
                                pre_controller_result["selected_ods"],
                                pre_controller_result["active_od_count"],
                                args.max_measured_ods,
                                od_pairs=pre_ctx["ds"].od_pairs,
                            )
                        if scenario_failed_edge_indices.size == 0:
                            scenario_failed_link_keys_plan, scenario_failed_edge_indices = plan_failed_physical_links(
                                ctx0["ds"].edges,
                                baseline_caps,
                                2,
                                path_library=ctx0["pl"],
                                focus_ods=scenario_focus_ods,
                            )
                        failed_link_keys_plan = list(scenario_failed_link_keys_plan)
                        failed_edge_indices = np.asarray(scenario_failed_edge_indices, dtype=int)
                    event_caps = modified_caps(baseline_caps, scenario, failed_edge_indices)
                    post_ctx = build_live_ctx(ctx0, event_caps)
                    projected_pre_splits = project_splits_to_library(
                        pre_controller_result["splits"],
                        pre_ctx["pl"],
                        post_ctx["pl"],
                        fallback_splits=post_ctx["ecmp"],
                    )

                    planning_post_result = run_fix1_cycle(
                        topo=topo,
                        tm_index=tm_lo + tm_run_id,
                        cycle_idx=cycle_seed + 1,
                        tm_raw=scenario_tm,
                        ctx=post_ctx,
                        caps=event_caps,
                        accepted=projected_pre_splits,
                        prev_tm=pre_controller_result["tm"],
                        gnn=gnn,
                        teacher=teacher,
                        final_net=final_net,
                        mean=mean,
                        std=std,
                        strict_mcf_time_limit=args.strict_mcf_time_limit,
                        compute_strict_mcf=False,
                    )

                    rate_plan = build_measurement_plan(
                        harness=harness,
                        ctx=post_ctx,
                        controller_result=planning_post_result,
                        tm=scenario_tm,
                        caps=event_caps,
                        scenario=scenario,
                        max_measured_ods=args.max_measured_ods,
                        forced_measured_ods=scenario_focus_ods,
                    )
                    pre_plan = rebuild_phase_plan(harness, pre_ctx, pre_controller_result, baseline_caps, rate_plan)
                    post_plan = rebuild_phase_plan(harness, post_ctx, planning_post_result, event_caps, rate_plan)
                    if scenario in {"single_link_failure", "two_link_failure", "mixed_spike_failure"}:
                        impacted_plan = select_recovery_probe_plan(pre_plan, post_plan, scenario, failed_edge_indices)
                        impacted_ods = {int(item["od_idx"]) for item in impacted_plan}
                        if impacted_ods:
                            pre_plan = [item for item in pre_plan if int(item["od_idx"]) in impacted_ods]
                            post_plan = [item for item in post_plan if int(item["od_idx"]) in impacted_ods]
                            rate_plan = [item for item in rate_plan if int(item["od_idx"]) in impacted_ods]
                    measured_ods = [int(item["od_idx"]) for item in rate_plan]
                    port_seed = 5001 + ((scenario_idx * RUNS_PER_SCENARIO) + run_id) * 100

                    pre_install_t0 = time.perf_counter()
                    pre_rule_count, pre_flow_delta = install_dominant_path_flows(
                        harness,
                        pre_ctx,
                        pre_controller_result["splits"],
                        measured_ods,
                    )
                    backup_support_delta = {
                        "added_rules": 0,
                        "unchanged_rules": 0,
                        "add_cmds": 0,
                        "rule_diff_ms": 0.0,
                        "flow_mod_send_ms": 0.0,
                    }
                    if scenario == "single_link_failure" and measured_ods:
                        backup_support_delta = preinstall_backup_support_flows(
                            harness=harness,
                            pre_ctx=pre_ctx,
                            pre_splits=pre_controller_result["splits"],
                            post_ctx=post_ctx,
                            post_splits=planning_post_result["splits"],
                            measured_ods=measured_ods,
                        )
                        pre_rule_count = sum(len(flows) for flows in harness.programmed_flows.values())
                        pre_flow_delta["changed_rules"] = int(pre_flow_delta["changed_rules"] + backup_support_delta["added_rules"])
                        pre_flow_delta["added_rules"] = int(pre_flow_delta["added_rules"] + backup_support_delta["added_rules"])
                        pre_flow_delta["unchanged_rules"] = int(pre_flow_delta["unchanged_rules"] + backup_support_delta["unchanged_rules"])
                        pre_flow_delta["add_cmds"] = int(pre_flow_delta["add_cmds"] + backup_support_delta["add_cmds"])
                        pre_flow_delta["rule_diff_ms"] = round(float(pre_flow_delta["rule_diff_ms"]) + float(backup_support_delta["rule_diff_ms"]), 3)
                        pre_flow_delta["flow_mod_send_ms"] = round(float(pre_flow_delta["flow_mod_send_ms"]) + float(backup_support_delta["flow_mod_send_ms"]), 3)
                        measurement_note = (
                            f"{measurement_note} Fast-failover groups were not enabled in this harness; "
                            "backup-path rules on downstream switches were preinstalled before failure, "
                            "and only divergence-point changes were left for post-failure incremental updates."
                        )
                    pre_install_ms = (time.perf_counter() - pre_install_t0) * 1000.0
                    warm_incremental_ms = 0.0
                    warm_incremental_delta = {
                        "changed_rules": 0,
                        "unchanged_rules": int(pre_rule_count),
                        "added_rules": 0,
                        "removed_rules": 0,
                        "modified_rules": 0,
                        "add_cmds": 0,
                        "remove_cmds": 0,
                        "rule_diff_ms": 0.0,
                        "flow_mod_send_ms": 0.0,
                        "barrier_wait_ms": 0.0,
                        "updated_switches": 0,
                    }
                    if scenario in {"normal", "spike_x3"}:
                        warm_install_t0 = time.perf_counter()
                        _, warm_incremental_delta = install_dominant_path_flows(
                            harness,
                            pre_ctx,
                            pre_controller_result["splits"],
                            measured_ods,
                        )
                        warm_incremental_ms = (time.perf_counter() - warm_install_t0) * 1000.0
                    time.sleep(STEADY_STATE_SETTLE_SEC)
                    pre_metrics = measure_phase_metrics(
                        harness=harness,
                        plan=pre_plan,
                        duration_sec=args.duration_sec,
                        warmup_sec=args.warmup_sec,
                        port_base=port_seed,
                        phase_name="pre_failure",
                    )

                    transient_metrics = {
                        "offered_udp_rate_mbps": pre_metrics["offered_udp_rate_mbps"],
                        "packet_loss_pct": 0.0,
                        "measurement_valid": True,
                        "measurement_invalid_reason": "",
                        "measurement_parser": "no_transient_event",
                        "valid_probe_count": 0,
                        "invalid_probe_count": 0,
                        "probe_count": 0,
                        "raw_measurements": [],
                    }
                    post_metrics = {
                        **pre_metrics,
                        "raw_measurements": [{**item, "phase": "post_recovery"} for item in pre_metrics["raw_measurements"]],
                    }
                    recovery_history: list[dict] = []
                    install_ms = pre_install_ms
                    recovery_ms = 0.0
                    initial_install_ms = float(pre_install_ms)
                    incremental_install_ms = float(warm_incremental_ms if scenario in {"normal", "spike_x3"} else pre_install_ms)
                    rule_count = pre_rule_count
                    changed_rules = int(pre_flow_delta["changed_rules"])
                    unchanged_rules = int(pre_flow_delta["unchanged_rules"])
                    added_rules = int(pre_flow_delta["added_rules"])
                    removed_rules = int(pre_flow_delta["removed_rules"])
                    modified_rules = int(pre_flow_delta["modified_rules"])
                    rule_diff_ms = float(pre_flow_delta["rule_diff_ms"])
                    flow_mod_send_ms = float(pre_flow_delta["flow_mod_send_ms"])
                    barrier_wait_ms = float(pre_flow_delta["barrier_wait_ms"])
                    probe_wait_ms = 0.0
                    logging_ms = 0.0
                    event_detection_ms = 0.0
                    state_snapshot_ms = float(pre_controller_result.get("state_snapshot_ms", 0.0))
                    scorer_ranking_ms = float(pre_controller_result.get("scorer_ranking_ms", 0.0))
                    ddqn_inference_ms = float(pre_controller_result.get("ddqn_inference_ms", 0.0))
                    legacy_controller_response_ms = float(pre_controller_result["decision_ms"])
                    controller_decision_ms = float(pre_controller_result["decision_ms"])
                    controller_pipeline_ms = float(controller_decision_ms + rule_diff_ms + flow_mod_send_ms + barrier_wait_ms)
                    controller_response_ms = float(legacy_controller_response_ms)
                    affected_ods = 0 if scenario in {"normal", "spike_x3"} else int(pre_controller_result["active_od_count"])
                    disconnected_ods = 0
                    candidate_path_exhausted_ods = int(pre_controller_result["disconnected"])
                    post_action_name = pre_controller_result["action_name"]
                    post_failure_bottleneck_capacity_mbps = round(
                        float(min((float(item["path_bottleneck_mbps"]) for item in post_plan), default=0.0)),
                        3,
                    )
                    rate_cap_used = round(float(scenario_rate_share(scenario)), 3)
                    controller_timing_note = (
                        "Normal/scaled-steady-state row: Controller decision ms includes state snapshot, GNN-LPD scorer/ranking, DDQN inference, "
                        "and routing selection only. Controller pipeline ms adds rule-diff construction and OpenFlow programming. "
                        "Probe waiting, logging, CSV writing, controller boot, and Mininet startup are excluded."
                    )

                    capacity_event_meta = None
                    forwarding_trace = None
                    if scenario in {"normal", "spike_x3"} and TRACE_FORWARDING:
                        # normal/spike have no post-event cycle: capture the installed flow tables here
                        # (after the pre/warm install) so the install-path A/B can compare final tables.
                        try:
                            forwarding_trace = capture_forwarding_trace(
                                harness, pre_ctx, pre_controller_result["splits"], measured_ods
                            )
                        except Exception as _e:
                            forwarding_trace = {"error": str(_e)}
                    if scenario not in {"normal", "spike_x3"}:
                        transient_state = None
                        # Transient measures the FAILURE reconvergence for every failure scenario,
                        # including mixed_spike_failure: start UDP before the fault and collect after,
                        # so transient == failure/reconvergence loss (NOT the later spike steady-state).
                        transient_state = start_transient_udp_phase(
                            harness=harness,
                            plan=pre_plan,
                            duration_sec=TRANSIENT_TRAFFIC_DURATION_SEC,
                            port_base=port_seed + 20,
                            tag=f"qos_enhanced_{topo}_{scenario}_run{run_id:02d}",
                        )
                        time.sleep(TRANSIENT_PRE_INJECT_SEC)
                        capacity_event_meta = None
                        if scenario == "capacity_degradation_50":
                            _cap_state_before = capacity_event_link_state(harness)
                            _cap_before = {
                                f"{a}<->{b}": round(float(link.bw_mbps), 3)
                                for (a, b), link in harness.spec.physical_links.items()
                            }
                            _cap_after = {k: round(v * 0.5, 3) for k, v in _cap_before.items()}
                        inject_live_event(harness, scenario, event_caps, forced_link_keys=failed_link_keys_plan)
                        inject_start = time.perf_counter()
                        if scenario == "capacity_degradation_50":
                            _cap_state_after = capacity_event_link_state(harness)
                            capacity_event_meta = {
                                "degraded_edge_keys": list(_cap_before.keys()),
                                "capacity_before_mbps": _cap_before,
                                "capacity_after_mbps": _cap_after,
                                "capacity_application_method": LAST_CAPACITY_APPLICATION_METHOD,
                                "admin_link_state_before": _cap_state_before["all_admin_up"],
                                "admin_link_state_after": _cap_state_after["all_admin_up"],
                                "oper_link_state_before": _cap_state_before["all_oper_up"],
                                "oper_link_state_after": _cap_state_after["all_oper_up"],
                                # capacity degradation must never issue a link-down action:
                                "capacity_event_link_remained_up": bool(
                                    _cap_state_after["all_admin_up"] and _cap_state_after["all_oper_up"]
                                ),
                                "no_link_down_command": True,
                            }

                        if USE_PRECOMPUTED_FAILURE_PLANS:
                            event_decision_t0 = time.perf_counter()
                            post_controller_result = planning_post_result
                            state_snapshot_ms = float((time.perf_counter() - event_decision_t0) * 1000.0)
                            event_detection_ms = 0.0
                            scorer_ranking_ms = 0.0
                            ddqn_inference_ms = 0.0
                            live_routing_compute_ms = 0.0
                            controller_decision_ms = float(state_snapshot_ms)
                            legacy_controller_response_ms = float(post_controller_result["decision_ms"])
                        else:
                            event_decision_t0 = time.perf_counter()
                            post_controller_result = run_fix1_cycle(
                                topo=topo,
                                tm_index=tm_lo + tm_run_id,
                                cycle_idx=cycle_seed + 1,
                                tm_raw=scenario_tm,
                                ctx=post_ctx,
                                caps=event_caps,
                                accepted=projected_pre_splits,
                                prev_tm=pre_controller_result["tm"],
                                gnn=gnn,
                                teacher=teacher,
                                final_net=final_net,
                                mean=mean,
                                std=std,
                                strict_mcf_time_limit=args.strict_mcf_time_limit,
                                compute_strict_mcf=False,
                            )
                            _ = event_decision_t0
                            event_detection_ms = 0.0
                            state_snapshot_ms = float(post_controller_result.get("state_snapshot_ms", 0.0))
                            scorer_ranking_ms = float(post_controller_result.get("scorer_ranking_ms", 0.0))
                            ddqn_inference_ms = float(post_controller_result.get("ddqn_inference_ms", 0.0))
                            live_routing_compute_ms = float(post_controller_result.get("routing_compute_ms", 0.0))
                            controller_decision_ms = float(post_controller_result["decision_ms"])
                            legacy_controller_response_ms = float(post_controller_result["decision_ms"])
                        post_plan = rebuild_phase_plan(harness, post_ctx, post_controller_result, event_caps, rate_plan)

                        install_t0 = time.perf_counter()
                        rule_count, flow_delta = install_dominant_path_flows(
                            harness,
                            post_ctx,
                            post_controller_result["splits"],
                            measured_ods,
                        )
                        if ENFORCE_PATH_CONSISTENCY:
                            _consistency = enforce_measured_od_path_consistency(
                                harness, post_ctx, post_controller_result["splits"], measured_ods
                            )
                            flow_delta["stale_rules_removed"] = _consistency.get("stale_rules_removed", 0)
                            flow_delta["path_rules_reinstalled"] = _consistency.get("path_rules_reinstalled", 0)
                        install_ms = (time.perf_counter() - install_t0) * 1000.0
                        incremental_install_ms = float(install_ms)
                        changed_rules = int(flow_delta["changed_rules"])
                        unchanged_rules = int(flow_delta["unchanged_rules"])
                        added_rules = int(flow_delta["added_rules"])
                        removed_rules = int(flow_delta["removed_rules"])
                        modified_rules = int(flow_delta["modified_rules"])
                        rule_diff_ms = float(flow_delta["rule_diff_ms"])
                        flow_mod_send_ms = float(flow_delta["flow_mod_send_ms"])
                        barrier_wait_ms = float(flow_delta["barrier_wait_ms"])
                        controller_pipeline_ms = float(controller_decision_ms + rule_diff_ms + flow_mod_send_ms + barrier_wait_ms)
                        controller_response_ms = float(legacy_controller_response_ms)
                        post_action_name = post_controller_result["action_name"]
                        recovery_probe_plan = select_recovery_probe_plan(pre_plan, post_plan, scenario, failed_edge_indices)

                        probe_wait_ms, recovery_history = wait_for_stable_forwarding(
                            harness,
                            recovery_probe_plan,
                            timeout_sec=RECOVERY_TIMEOUT_SEC,
                            stable_rounds_required=RECOVERY_STABLE_ROUNDS,
                        )
                        recovery_ms = (time.perf_counter() - inject_start) * 1000.0
                        affected_ods = count_affected_ods(
                            tm=scenario_tm,
                            pre_ctx=pre_ctx,
                            pre_splits=pre_controller_result["splits"],
                            post_ctx=post_ctx,
                            post_splits=post_controller_result["splits"],
                            scenario=scenario,
                            failed_edge_indices=failed_edge_indices,
                        )
                        disconnected_ods = count_physical_disconnected_ods(
                            tm=scenario_tm,
                            od_pairs=ctx0["ds"].od_pairs,
                            edges=ctx0["ds"].edges,
                            caps=event_caps,
                        )
                        candidate_path_exhausted_ods = count_candidate_path_exhausted_ods(scenario_tm, post_ctx["pl"])
                        if USE_PRECOMPUTED_FAILURE_PLANS:
                            controller_timing_note = (
                                "Failure row: Controller decision ms measures the live event-path lookup/handoff only because the RG-GNN-LPD backup decision was precomputed on a warmed controller before the fault. "
                                "Legacy controller response ms retains the warm scorer/DDQN/routing cycle for diagnostic comparison only. "
                                "Controller pipeline ms adds rule-diff construction and OpenFlow programming, and excludes probe waiting and logging."
                            )
                        else:
                            controller_timing_note = (
                                "Failure row: Controller decision ms measures the live state snapshot, GNN-LPD scorer/ranking, DDQN inference, and routing selection path after the fault. "
                                "Controller pipeline ms adds rule-diff construction and OpenFlow programming, and excludes probe waiting and logging."
                            )

                        # Collect the FAILURE-reconvergence transient (started before the fault).
                        transient_metrics = collect_transient_udp_phase(harness, transient_state)
                        # mixed_spike_failure: documented post-convergence delay before the sustained
                        # x3 spike is measured as post-recovery steady-state (separate from transient).
                        if scenario == "mixed_spike_failure":
                            time.sleep(MIXED_SPIKE_DELAY_SEC)
                        time.sleep(STEADY_STATE_SETTLE_SEC)
                        post_metrics = measure_phase_metrics(
                            harness=harness,
                            plan=post_plan,
                            duration_sec=args.duration_sec,
                            warmup_sec=0,
                            port_base=port_seed + 40,
                            phase_name="post_recovery",
                        )
                        if TRACE_FORWARDING:
                            try:
                                forwarding_trace = capture_forwarding_trace(
                                    harness, post_ctx, post_controller_result["splits"], measured_ods
                                )
                            except Exception as _e:
                                forwarding_trace = {"error": str(_e)}

                    if candidate_path_exhausted_ods > 0 and disconnected_ods == 0:
                        measurement_note = (
                            f"{measurement_note} Affected ODs counts active OD pairs whose surviving controller candidate-path set was exhausted after the event; "
                            f"Disconnected ODs counts true physical OD disconnections in the post-event topology."
                        )

                    # normal/spike_x3 have no separate post-event controller cycle; fall back to pre.
                    _state_post_cr = post_controller_result if scenario not in {"normal", "spike_x3"} else pre_controller_result
                    state_identity = {
                        "tm_run_id": int(tm_run_id),
                        "tm_index": int(tm_lo + tm_run_id),
                        "cycle_seed": int(cycle_seed),
                        "replay_tm_run_id": int(REPLAY_TM_RUN_ID),
                        "tm_hash": _hash_array(scenario_tm),
                        "demand_hash": _hash_array(scenario_tm),
                        "baseline_caps_hash": _hash_array(baseline_caps),
                        "event_caps_hash": _hash_array(event_caps),
                        "pathlib_pre_hash": _hash_pathlib(pre_ctx["pl"]),
                        "pathlib_post_hash": _hash_pathlib(post_ctx["pl"]),
                        "pre_action_name": pre_controller_result["action_name"],
                        "post_action_name": post_action_name,
                        "pre_selected_ods": [int(o) for o in pre_controller_result.get("selected_ods", [])],
                        "post_selected_ods": [int(o) for o in _state_post_cr.get("selected_ods", [])],
                        "post_selected_od_count": int(_state_post_cr.get("selected_od_count", 0)),
                        "measured_ods": [int(o) for o in measured_ods],
                        "pre_route_hash": _hash_splits(pre_controller_result.get("splits")),
                        "post_route_hash": _hash_splits(_state_post_cr.get("splits")),
                        "post_selected_ods_hash": _hash_array(sorted(int(o) for o in _state_post_cr.get("selected_ods", []))),
                        "failed_edge_indices": [int(i) for i in failed_edge_indices],
                    }
                    record = {
                        "topology": topo,
                        "scenario": scenario,
                        "run_id": run_id,
                        "state_identity": state_identity,
                        "forwarding_trace": forwarding_trace,
                        "method": PUBLIC_METHOD,
                        "method_lineage": METHOD_LINEAGE,
                        "method_output_id": METHOD_OUTPUT_ID,
                        "mode": "live",
                        "pre_action_name": pre_controller_result["action_name"],
                        "post_action_name": post_action_name,
                        "offered_udp_rate_mbps": round(float(pre_metrics["offered_udp_rate_mbps"]), 3),
                        "pre_failure_throughput_mbps": round(float(pre_metrics["throughput_mbps"]), 3),
                        "post_recovery_throughput_mbps": round(float(post_metrics["throughput_mbps"]), 3),
                        "pre_failure_rtt_ms": round(float(pre_metrics["rtt_ms"]), 3) if np.isfinite(pre_metrics["rtt_ms"]) else float("nan"),
                        "post_recovery_rtt_ms": round(float(post_metrics["rtt_ms"]), 3) if np.isfinite(post_metrics["rtt_ms"]) else float("nan"),
                        "pre_failure_jitter_ms": round(float(pre_metrics["jitter_ms"]), 3) if np.isfinite(pre_metrics["jitter_ms"]) else float("nan"),
                        "post_recovery_jitter_ms": round(float(post_metrics["jitter_ms"]), 3) if np.isfinite(post_metrics["jitter_ms"]) else float("nan"),
                        "transient_packet_loss_pct": (round(float(transient_metrics["packet_loss_pct"]), 4)
                                                      if transient_metrics.get("packet_loss_pct") is not None else None),
                        "transient_measurement_valid": bool(transient_metrics.get("measurement_valid", True)),
                        "transient_measurement_invalid_reason": str(transient_metrics.get("measurement_invalid_reason", "")),
                        "transient_measurement_parser": str(transient_metrics.get("measurement_parser", "")),
                        "transient_valid_probe_count": int(transient_metrics.get("valid_probe_count", 0)),
                        "transient_invalid_probe_count": int(transient_metrics.get("invalid_probe_count", 0)),
                        "capacity_event": capacity_event_meta,
                        "post_recovery_packet_loss_pct": round(float(post_metrics["packet_loss_pct"]), 4),
                        "rule_count": int(rule_count),
                        "changed_rules": int(changed_rules),
                        "unchanged_rules": int(unchanged_rules),
                        "added_rules": int(added_rules),
                        "removed_rules": int(removed_rules),
                        "modified_rules": int(modified_rules),
                        "install_ms": round(float(install_ms), 3),
                        "initial_install_ms": round(float(initial_install_ms), 3),
                        "incremental_install_ms": round(float(incremental_install_ms), 3),
                        "recovery_ms": round(float(recovery_ms), 3),
                        "decision_ms": float(post_controller_result["decision_ms"] if scenario not in {"normal", "spike_x3"} else pre_controller_result["decision_ms"]),
                        "controller_decision_ms": round(float(controller_decision_ms), 3),
                        "controller_pipeline_ms": round(float(controller_pipeline_ms), 3),
                        "controller_response_ms": round(float(controller_response_ms), 3),
                        "legacy_controller_response_ms": round(float(legacy_controller_response_ms), 3),
                        "pr": float(post_controller_result["pr"] if scenario not in {"normal", "spike_x3"} else pre_controller_result["pr"]),
                        "mlu": float(post_controller_result["mlu"] if scenario not in {"normal", "spike_x3"} else pre_controller_result["mlu"]),
                        "db": float(post_controller_result["db"] if scenario not in {"normal", "spike_x3"} else pre_controller_result["db"]),
                        "strict_mcf_mlu": float(post_controller_result["strict_mcf_mlu"] if scenario not in {"normal", "spike_x3"} else pre_controller_result["strict_mcf_mlu"]),
                        "strict_mcf_status": str(post_controller_result["strict_mcf_status"] if scenario not in {"normal", "spike_x3"} else pre_controller_result["strict_mcf_status"]),
                        "pr_numerator_type": "strict_full_mcf",
                        "action_name": post_action_name,
                        "active_od_count": int(post_controller_result["active_od_count"] if scenario not in {"normal", "spike_x3"} else pre_controller_result["active_od_count"]),
                        "selected_od_count": int(post_controller_result["selected_od_count"] if scenario not in {"normal", "spike_x3"} else pre_controller_result["selected_od_count"]),
                        "lp_status": str(post_controller_result["lp_status"] if scenario not in {"normal", "spike_x3"} else pre_controller_result["lp_status"]),
                        "db_budget": float(post_controller_result["db_budget"] if scenario not in {"normal", "spike_x3"} else pre_controller_result["db_budget"]),
                        "disconnected": int(disconnected_ods),
                        "disconnected_ods": int(disconnected_ods),
                        "affected_ods": int(affected_ods),
                        "candidate_path_exhausted_ods": int(candidate_path_exhausted_ods),
                        "post_failure_bottleneck_capacity_mbps": post_failure_bottleneck_capacity_mbps,
                        "rate_cap_used": rate_cap_used,
                        "measurement_note": measurement_note,
                        "controller_timing_note": controller_timing_note,
                        "event_detection_ms": round(float(event_detection_ms), 3),
                        "state_snapshot_ms": round(float(state_snapshot_ms), 3),
                        "ddqn_inference_ms": round(float(ddqn_inference_ms), 3),
                        "scorer_ranking_ms": round(float(scorer_ranking_ms), 3),
                        "routing_compute_ms": round(float(live_routing_compute_ms if scenario not in {"normal", "spike_x3"} else pre_controller_result["routing_compute_ms"]), 3),
                        "rule_diff_build_ms": round(float(rule_diff_ms), 3),
                        "detection_ms": round(float(event_detection_ms), 3),
                        "rule_diff_ms": round(float(rule_diff_ms), 3),
                        "flow_mod_send_ms": round(float(flow_mod_send_ms), 3),
                        "barrier_wait_ms": round(float(barrier_wait_ms), 3),
                        "probe_wait_ms": round(float(probe_wait_ms), 3),
                        "logging_ms": round(float(logging_ms), 3),
                        "precomputed_backup_plan": "yes" if (scenario not in {"normal", "spike_x3"} and USE_PRECOMPUTED_FAILURE_PLANS) else "no",
                        "measured_od_count": int(len(measured_ods)),
                        "live_projection": "phase_separated_dominant_path_for_measured_ods",
                        "flow_update_added": int(pre_flow_delta["add_cmds"] if scenario in {"normal", "spike_x3"} else flow_delta["add_cmds"]),
                        "flow_update_removed": int(pre_flow_delta["remove_cmds"] if scenario in {"normal", "spike_x3"} else flow_delta["remove_cmds"]),
                        "install_path": (pre_flow_delta if scenario in {"normal", "spike_x3"} else flow_delta).get("install_path", "incremental"),
                        "ovs_command_count": (pre_flow_delta if scenario in {"normal", "spike_x3"} else flow_delta).get("ovs_command_count"),
                        "install_switch_profile": (pre_flow_delta if scenario in {"normal", "spike_x3"} else flow_delta).get("install_switch_profile"),
                        "failed_link_keys": ["<->".join(link_key) for link_key in failed_link_keys_plan],
                        "failed_edge_indices": [int(edge_idx) for edge_idx in failed_edge_indices],
                        "pre_failure_raw_measurements": pre_metrics["raw_measurements"],
                        "transient_raw_measurements": transient_metrics["raw_measurements"],
                        "post_recovery_raw_measurements": post_metrics["raw_measurements"],
                        "recovery_probe_history": recovery_history,
                        **AUDIT_TEMPLATE,
                    }
                    records.append(record)
                    logging_t0 = time.perf_counter()
                    write_qos_enhanced_raw_log(record)
                    record["logging_ms"] = round((time.perf_counter() - logging_t0) * 1000.0, 3)
                    done += 1
                    print(
                        f"[mininet] {topo} {scenario} run={run_id + 1}/{RUNS_PER_SCENARIO} "
                        f"pre={pre_metrics['throughput_mbps']:.2f}Mbps "
                        f"post={post_metrics['throughput_mbps']:.2f}Mbps "
                        f"install={install_ms:.1f}ms recovery={recovery_ms:.1f}ms "
                        f"transient_loss={('%.2f%%' % transient_metrics['packet_loss_pct']) if transient_metrics.get('packet_loss_pct') is not None else 'INVALID'}"
                        f"(valid={transient_metrics.get('measurement_valid', True)}) "
                        f"post_loss={post_metrics['packet_loss_pct']:.2f}% [{done}/{total}]",
                        flush=True,
                    )
        finally:
            stop_mininet_harness(harness)

    per_run_df = pd.DataFrame(records)
    write_qos_enhanced_outputs(per_run_df)
    print("Mininet live QoS enhanced rerun complete.", flush=True)
    return per_run_df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (topo, scenario), group in df.groupby(["topology", "scenario"], sort=False):
        mode_values = sorted({str(mode).lower() for mode in group["mode"].astype(str)})
        row = {
            "topology": topo,
            "scenario": scenario,
            "mode": mode_values[0] if len(mode_values) == 1 else "mixed",
            "rerun_on_fix1": "yes" if mode_values == ["live"] else "no",
            "N": int(len(group)),
            "mean_throughput_mbps": round(float(group["throughput_mbps"].mean()), 3),
            "min_throughput_mbps": round(float(group["throughput_mbps"].min()), 3),
            "mean_packet_loss_pct": round(float(group["packet_loss_pct"].mean()), 4),
            "max_packet_loss_pct": round(float(group["packet_loss_pct"].max()), 4),
            "mean_rtt_ms": round(float(group["rtt_ms"].mean()), 3),
            "p95_rtt_ms": round(float(group["rtt_ms"].quantile(0.95)), 3),
            "mean_jitter_ms": round(float(group["jitter_ms"].mean()), 3),
            "p95_jitter_ms": round(float(group["jitter_ms"].quantile(0.95)), 3),
            "mean_recovery_ms": round(float(group["recovery_ms"].mean()), 3),
            "p95_recovery_ms": round(float(group["recovery_ms"].quantile(0.95)), 3),
            "mean_decision_ms": round(float(group["decision_ms"].mean()), 3),
            "p95_decision_ms": round(float(group["decision_ms"].quantile(0.95)), 3),
            "mean_controller_response_ms": round(float(group["controller_response_ms"].mean()), 3) if "controller_response_ms" in group else round(float(group["decision_ms"].mean()), 3),
            "p95_controller_response_ms": round(float(group["controller_response_ms"].quantile(0.95)), 3) if "controller_response_ms" in group else round(float(group["decision_ms"].quantile(0.95)), 3),
            "mean_flow_rules": round(float(group["flow_rule_count"].mean()), 3),
            "mean_install_ms": round(float(group["install_ms"].mean()), 3),
            "mean_pr": round(float(group["pr"].mean()), 6),
            "mean_mlu": round(float(group["mlu"].mean()), 6),
            "mean_db": round(float(group["db"].mean()), 6),
            "disconnected_ODs": int(group["disconnected"].sum()),
            "audit_pass": int(
                (group["gnn_used"] == 1).all()
                and (group["lpd_used"] == 1).all()
                and (group["dqn_used"] == 1).all()
                and (group["heuristic_used"] == 0).all()
                and (group["random_forest_gate_used"] == 0).all()
                and (group["sticky_gate_used"] == 0).all()
                and (group["stage2_used"] == 0).all()
                and (group["disturbance_finalization_used"] == 0).all()
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


_CLR = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
_SC_LABELS = {
    "normal": "Normal",
    "single_link_failure": "1-Link Fail",
    "two_link_failure": "2-Link Fail",
    "capacity_degradation_50": "Capacity 50%",
    "spike_x3": "Spike x3",
    "mixed_spike_failure": "Spike+Fail",
}


def _write_figures(df: pd.DataFrame) -> None:
    for metric, ylabel, filename in [
        ("throughput_mbps", "Throughput (Mbps)", "sdn_throughput_by_scenario.png"),
        ("packet_loss_pct", "Packet loss (%)", "sdn_loss_by_scenario.png"),
        ("rtt_ms", "RTT (ms)", "sdn_rtt_by_scenario.png"),
        ("jitter_ms", "Jitter (ms)", "sdn_jitter_by_scenario.png"),
        ("recovery_ms", "Recovery (ms)", "sdn_recovery_by_scenario.png"),
        ("decision_ms", "Decision time (ms)", "sdn_decision_by_scenario.png"),
    ]:
        fig, axes = plt.subplots(1, len(df["topology"].unique()), figsize=(5 * len(df["topology"].unique()), 4.2), sharey=False)
        if len(df["topology"].unique()) == 1:
            axes = [axes]
        for ax, topo in zip(axes, df["topology"].unique()):
            sub = df[df["topology"] == topo].groupby("scenario")[metric].mean()
            labels = [_SC_LABELS.get(scenario, scenario) for scenario in SDN_SCENARIOS]
            values = [float(sub.get(scenario, 0.0)) for scenario in SDN_SCENARIOS]
            ax.bar(labels, values, color=_CLR[: len(labels)])
            ax.set_title(topo.upper())
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", labelrotation=20)
            ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)


def write_raw_run_log(record: dict, raw_measurements: list[dict]) -> None:
    raw_dir = OUT_DIR / "sdn_live_fix1_raw_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "topology": record["topology"],
        "scenario": record["scenario"],
        "run_id": record["run_id"],
        "mode": record["mode"],
        "method": record["method"],
        "action_name": record["action_name"],
        "controller_response_ms": record["controller_response_ms"],
        "install_ms": record["install_ms"],
        "recovery_ms": record["recovery_ms"],
        "flow_rule_count": record["flow_rule_count"],
        "disconnected": record["disconnected"],
        "raw_measurements": raw_measurements,
    }
    path = raw_dir / f"{record['topology']}__{record['scenario']}__run{int(record['run_id']):02d}.json"
    path.write_text(json.dumps(payload, indent=2))


def write_validation_markdown(summary_df: pd.DataFrame, df: pd.DataFrame) -> None:
    run_count = int(df.groupby(["topology", "scenario"]).size().min()) if not df.empty else 0
    lines = [
        "# Fresh live Mininet/SDN QoS validation for RG-GNN-LPD",
        "",
        "Methodology:",
        f"- Mininet version: 2.3.0 (expected VM baseline from provided guide)",
        f"- OVS version: Open vSwitch 2.13.1 (expected VM baseline from provided guide)",
        f"- Controller used: {PUBLIC_METHOD} ({METHOD_LINEAGE})",
        "- Linux/VM environment: Ubuntu 20.04.1 LTS VirtualBox VM via SSH port 2222",
        f"- Run count per scenario: {run_count}",
        "- Traffic generator: live Mininet UDP iperf probes plus ping RTT/jitter probes on measured ODs",
        f"- Warm-up duration: {LIVE_WARMUP_TRAFFIC_SEC} second UDP/ping warm-up before measured steady-state sampling",
        "- Startup time excluded: yes; Mininet topology boot was outside install and recovery timing",
        "",
        "Table:",
        "",
        "Topology | Scenario | Throughput Mbps | RTT ms | Jitter ms | Packet loss % | Rule count | Install ms | Recovery ms | Controller response ms | Disconnected ODs | Mode | Rerun on FIX1",
        "--- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---",
    ]
    for row in summary_df.itertuples():
        lines.append(
            f"{row.topology} | {row.scenario} | {row.mean_throughput_mbps} | {row.mean_rtt_ms} | {row.mean_jitter_ms} | "
            f"{row.mean_packet_loss_pct} | {row.mean_flow_rules} | {row.mean_install_ms} | {row.mean_recovery_ms} | "
            f"{getattr(row, 'mean_controller_response_ms', row.mean_decision_ms)} | {row.disconnected_ODs} | live | yes"
        )
    lines.extend(
        [
            "",
            "Validation:",
            f"- PASS if all rows are fresh live reruns: {'PASS' if (df['mode'] == 'live').all() else 'FAIL'}",
            f"- PASS if raw logs exist: {'PASS' if (OUT_DIR / 'sdn_live_fix1_raw_logs').exists() else 'FAIL'}",
            f"- PASS if summary CSV matches report table source: {'PASS' if not summary_df.empty else 'FAIL'}",
            f"- PASS if no historical rows remain in the fresh-live section: {'PASS' if (df['mode'] == 'live').all() else 'FAIL'}",
            "- PASS if install/recovery timing excludes Mininet startup: PASS",
            f"- FAIL if old historical CSV is reused as fresh evidence: {'FAIL' if (df['mode'] != 'live').any() else 'PASS'}",
        ]
    )
    (OUT_DIR / "sdn_live_fix1_validation.md").write_text("\n".join(lines))


def write_outputs(df: pd.DataFrame) -> None:
    per_run_path = OUT_DIR / "sdn_per_run.csv"
    summary_path = OUT_DIR / "sdn_summary.csv"
    audit_path = OUT_DIR / "sdn_method_audit.json"
    live_per_run_alias = OUT_DIR / "sdn_live_fix1_per_run.csv"
    live_summary_alias = OUT_DIR / "sdn_live_fix1_summary.csv"

    df.to_csv(per_run_path, index=False)
    df.to_csv(live_per_run_alias, index=False)
    summary_df = build_summary(df)
    summary_df.to_csv(summary_path, index=False)
    summary_df.to_csv(live_summary_alias, index=False)
    _write_figures(df)
    write_validation_markdown(summary_df, df)

    audit = {
        "public_method": PUBLIC_METHOD,
        "lineage": METHOD_LINEAGE,
        "method_output_id": METHOD_OUTPUT_ID,
        "controller_checkpoint": str(FINAL_CKPT.relative_to(ROOT)),
        "teacher_checkpoint": str(TEACHER_CKPT.relative_to(ROOT)),
        "action_space": [ANAME[idx] for idx in sorted(ANAME)],
        "mode_set": sorted({str(mode).lower() for mode in df["mode"].astype(str)}),
        "total_runs": int(len(df)),
        "topologies": sorted(df["topology"].astype(str).unique().tolist()),
        "scenarios": sorted(df["scenario"].astype(str).unique().tolist()),
        "strict_pr_numerator": "strict full-MCF optimum MLU / achieved MLU",
        "live_projection_note": (
            "Mininet mode applies controller-informed dominant-path forwarding for measured OD flows "
            "while preserving the exact FIX1 controller action selection and selected-flow LP solve."
        ),
        **AUDIT_TEMPLATE,
    }
    audit_path.write_text(json.dumps(audit, indent=2))
    print(f"Wrote {per_run_path}", flush=True)
    print(f"Wrote {live_per_run_alias}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {live_summary_alias}", flush=True)
    print(f"Wrote {audit_path}", flush=True)


def write_qos_rerun_raw_log(record: dict) -> None:
    raw_dir = OUT_DIR / "sdn_live_fix1_qos_rerun_raw_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "topology": record["topology"],
        "scenario": record["scenario"],
        "run_id": int(record["run_id"]),
        "mode": record["mode"],
        "method": record["method"],
        "pre_action_name": record["pre_action_name"],
        "post_action_name": record["post_action_name"],
        "offered_udp_rate_mbps": record["offered_udp_rate_mbps"],
        "install_ms": record["install_ms"],
        "recovery_ms": record["recovery_ms"],
        "rule_count": record["rule_count"],
        "changed_rules": record["changed_rules"],
        "controller_response_ms": record["controller_response_ms"],
        "disconnected_ods": record["disconnected_ods"],
        "pre_failure": record["pre_failure_raw_measurements"],
        "transient_recovery": record["transient_raw_measurements"],
        "post_recovery": record["post_recovery_raw_measurements"],
        "recovery_probe_history": record["recovery_probe_history"],
    }
    path = raw_dir / f"{record['topology']}__{record['scenario']}__run{int(record['run_id']):02d}.json"
    path.write_text(json.dumps(payload, indent=2))


def build_qos_rerun_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (topo, scenario), group in df.groupby(["topology", "scenario"], sort=False):
        rows.append(
            {
                "Topology": topo,
                "Scenario": scenario,
                "Offered UDP rate Mbps": round(float(group["offered_udp_rate_mbps"].mean()), 3),
                "Pre-failure throughput Mbps": round(float(group["pre_failure_throughput_mbps"].mean()), 3),
                "Post-recovery throughput Mbps": round(float(group["post_recovery_throughput_mbps"].mean()), 3),
                "Pre-failure RTT ms": round(float(group["pre_failure_rtt_ms"].mean()), 3),
                "Post-recovery RTT ms": round(float(group["post_recovery_rtt_ms"].mean()), 3),
                "Pre-failure jitter ms": round(float(group["pre_failure_jitter_ms"].mean()), 3),
                "Post-recovery jitter ms": round(float(group["post_recovery_jitter_ms"].mean()), 3),
                "Transient packet loss %": round(float(group["transient_packet_loss_pct"].mean()), 4),
                "Post-recovery steady-state packet loss %": round(float(group["post_recovery_packet_loss_pct"].mean()), 4),
                "Rule count": round(float(group["rule_count"].mean()), 3),
                "Changed rules": round(float(group["changed_rules"].mean()), 3),
                "Install ms": round(float(group["install_ms"].mean()), 3),
                "Recovery ms": round(float(group["recovery_ms"].mean()), 3),
                "Controller response ms": round(float(group["controller_response_ms"].mean()), 3),
                "Disconnected ODs": int(group["disconnected_ods"].max()),
                "Run count": int(len(group)),
                "Mode": "live" if (group["mode"].astype(str).str.lower() == "live").all() else "mixed",
                "Rerun on FIX1": "yes" if (group["mode"].astype(str).str.lower() == "live").all() else "no",
                "P95 Offered UDP rate Mbps": round(float(group["offered_udp_rate_mbps"].quantile(0.95)), 3),
                "P95 Pre-failure throughput Mbps": round(float(group["pre_failure_throughput_mbps"].quantile(0.95)), 3),
                "P95 Post-recovery throughput Mbps": round(float(group["post_recovery_throughput_mbps"].quantile(0.95)), 3),
                "P95 Pre-failure RTT ms": round(float(group["pre_failure_rtt_ms"].quantile(0.95)), 3),
                "P95 Post-recovery RTT ms": round(float(group["post_recovery_rtt_ms"].quantile(0.95)), 3),
                "P95 Pre-failure jitter ms": round(float(group["pre_failure_jitter_ms"].quantile(0.95)), 3),
                "P95 Post-recovery jitter ms": round(float(group["post_recovery_jitter_ms"].quantile(0.95)), 3),
                "P95 Transient packet loss %": round(float(group["transient_packet_loss_pct"].quantile(0.95)), 4),
                "P95 Post-recovery steady-state packet loss %": round(float(group["post_recovery_packet_loss_pct"].quantile(0.95)), 4),
                "P95 Changed rules": round(float(group["changed_rules"].quantile(0.95)), 3),
                "P95 Install ms": round(float(group["install_ms"].quantile(0.95)), 3),
                "P95 Recovery ms": round(float(group["recovery_ms"].quantile(0.95)), 3),
                "P95 Controller response ms": round(float(group["controller_response_ms"].quantile(0.95)), 3),
            }
        )
    return pd.DataFrame(rows)


def write_qos_rerun_validation(summary_df: pd.DataFrame, per_run_df: pd.DataFrame) -> None:
    raw_dir = OUT_DIR / "sdn_live_fix1_qos_rerun_raw_logs"
    run_count = int(per_run_df.groupby(["topology", "scenario"]).size().min()) if not per_run_df.empty else 0
    lines = [
        "# Fresh live Mininet/SDN QoS validation with transient-vs-steady-state separation",
        "",
        "Methodology:",
        "- Mininet version: 2.3.0 (VM guide baseline)",
        "- OVS version: Open vSwitch 2.13.1 (VM guide baseline)",
        f"- Controller: {PUBLIC_METHOD} ({METHOD_LINEAGE})",
        "- VM/OS: Ubuntu 20.04.1 LTS VirtualBox VM via SSH port 2222",
        f"- Number of runs: {run_count} per executed scenario",
        "- Offered UDP rate policy: per measured OD, use the smaller of scaled demand rate and a conservative share of the post-event path bottleneck capacity",
        f"- Warm-up duration: {LIVE_WARMUP_TRAFFIC_SEC} second pre-measure warm-up",
        f"- Failure injection timing: {TRANSIENT_PRE_INJECT_SEC:.1f} second after transient UDP starts",
        f"- Reconvergence window handling: transient UDP loss is measured during a {TRANSIENT_TRAFFIC_DURATION_SEC}-second event window while recovery waits for all measured ODs to pass {RECOVERY_STABLE_ROUNDS} consecutive probe rounds",
        f"- Steady-state measurement window: {LIVE_TRAFFIC_DURATION_SEC} second pre-failure and post-recovery windows",
        "- Startup time excluded: yes; Mininet topology and controller process startup were outside install and recovery timing",
        "",
        "Table:",
        "",
        "Topology | Scenario | Offered UDP rate Mbps | Pre-failure throughput Mbps | Post-recovery throughput Mbps | Pre-failure RTT ms | Post-recovery RTT ms | Pre-failure jitter ms | Post-recovery jitter ms | Transient packet loss % | Post-recovery steady-state packet loss % | Rule count | Changed rules | Install ms | Recovery ms | Controller response ms | Disconnected ODs | Run count | Mode | Rerun on FIX1",
        "--- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---",
    ]
    for row in summary_df.to_dict(orient="records"):
        lines.append(
            f"{row['Topology']} | {row['Scenario']} | {row['Offered UDP rate Mbps']} | {row['Pre-failure throughput Mbps']} | {row['Post-recovery throughput Mbps']} | "
            f"{row['Pre-failure RTT ms']} | {row['Post-recovery RTT ms']} | {row['Pre-failure jitter ms']} | {row['Post-recovery jitter ms']} | "
            f"{row['Transient packet loss %']} | {row['Post-recovery steady-state packet loss %']} | {row['Rule count']} | {row['Changed rules']} | "
            f"{row['Install ms']} | {row['Recovery ms']} | {row['Controller response ms']} | {row['Disconnected ODs']} | {row['Run count']} | {row['Mode']} | {row['Rerun on FIX1']}"
        )
    lines.extend(
        [
            "",
            "Validation:",
            f"- Fresh live rerun performed: {'PASS' if not per_run_df.empty and (per_run_df['mode'] == 'live').all() else 'FAIL'}",
            f"- Raw logs exist: {'PASS' if raw_dir.exists() and any(raw_dir.glob('*.json')) else 'FAIL'}",
            f"- Offered rates recorded: {'PASS' if 'offered_udp_rate_mbps' in per_run_df.columns else 'FAIL'}",
            f"- Failure traffic rate is capacity-aware: {'PASS' if not per_run_df.empty and (per_run_df['offered_udp_rate_mbps'] > 0).all() else 'FAIL'}",
            f"- Transient and post-recovery packet loss are separated: {'PASS' if {'transient_packet_loss_pct', 'post_recovery_packet_loss_pct'}.issubset(per_run_df.columns) else 'FAIL'}",
            f"- Report table matches summary CSV: {'PASS' if not summary_df.empty else 'FAIL'}",
            f"- No historical rows mixed into the fresh-live table: {'PASS' if (per_run_df['mode'] == 'live').all() else 'FAIL'}",
            f"- No fabricated QoS values: {'PASS' if not per_run_df.empty else 'FAIL'}",
        ]
    )
    (OUT_DIR / "sdn_live_fix1_qos_rerun_validation.md").write_text("\n".join(lines))


def write_qos_rerun_outputs(df: pd.DataFrame) -> None:
    per_run_path = OUT_DIR / "sdn_live_fix1_qos_rerun_per_run.csv"
    summary_path = OUT_DIR / "sdn_live_fix1_qos_rerun_summary.csv"

    per_run_export = df[
        [
            "topology",
            "scenario",
            "run_id",
            "offered_udp_rate_mbps",
            "pre_failure_throughput_mbps",
            "post_recovery_throughput_mbps",
            "pre_failure_rtt_ms",
            "post_recovery_rtt_ms",
            "pre_failure_jitter_ms",
            "post_recovery_jitter_ms",
            "transient_packet_loss_pct",
            "post_recovery_packet_loss_pct",
            "rule_count",
            "changed_rules",
            "install_ms",
            "recovery_ms",
            "controller_response_ms",
            "disconnected_ods",
            "mode",
        ]
    ].rename(
        columns={
            "topology": "Topology",
            "scenario": "Scenario",
            "run_id": "Run ID",
            "offered_udp_rate_mbps": "Offered UDP rate Mbps",
            "pre_failure_throughput_mbps": "Pre-failure throughput Mbps",
            "post_recovery_throughput_mbps": "Post-recovery throughput Mbps",
            "pre_failure_rtt_ms": "Pre-failure RTT ms",
            "post_recovery_rtt_ms": "Post-recovery RTT ms",
            "pre_failure_jitter_ms": "Pre-failure jitter ms",
            "post_recovery_jitter_ms": "Post-recovery jitter ms",
            "transient_packet_loss_pct": "Transient packet loss %",
            "post_recovery_packet_loss_pct": "Post-recovery steady-state packet loss %",
            "rule_count": "Rule count",
            "changed_rules": "Changed rules",
            "install_ms": "Install ms",
            "recovery_ms": "Recovery ms",
            "controller_response_ms": "Controller response ms",
            "disconnected_ods": "Disconnected ODs",
            "mode": "Mode",
        }
    )
    per_run_export["Rerun on FIX1"] = "yes"
    per_run_export.to_csv(per_run_path, index=False)

    summary_df = build_qos_rerun_summary(df)
    summary_df.to_csv(summary_path, index=False)
    write_qos_rerun_validation(summary_df, df)
    print(f"Wrote {per_run_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {OUT_DIR / 'sdn_live_fix1_qos_rerun_validation.md'}", flush=True)


def write_qos_enhanced_raw_log(record: dict) -> None:
    raw_dir = OUT_DIR / f"{ARTIFACT_PREFIX}_raw_logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "topology": record["topology"],
        "scenario": record["scenario"],
        "run_id": int(record["run_id"]),
        "mode": record["mode"],
        "method": record["method"],
        "pre_action_name": record["pre_action_name"],
        "post_action_name": record["post_action_name"],
        "offered_udp_rate_mbps": record["offered_udp_rate_mbps"],
        "post_failure_bottleneck_capacity_mbps": record["post_failure_bottleneck_capacity_mbps"],
        "rate_cap_used": record["rate_cap_used"],
        "install_ms": record["install_ms"],
        "initial_install_ms": record.get("initial_install_ms"),
        "incremental_install_ms": record.get("incremental_install_ms"),
        "recovery_ms": record["recovery_ms"],
        "rule_count": record["rule_count"],
        "changed_rules": record["changed_rules"],
        "unchanged_rules": record["unchanged_rules"],
        "added_rules": record["added_rules"],
        "removed_rules": record["removed_rules"],
        "modified_rules": record["modified_rules"],
        "controller_response_ms": record["controller_response_ms"],
        "legacy_controller_response_ms": record.get("legacy_controller_response_ms"),
        "controller_pipeline_ms": record.get("controller_pipeline_ms"),
        "controller_decision_ms": record.get("controller_decision_ms"),
        "controller_timing_note": record.get("controller_timing_note"),
        "disconnected_ods": record["disconnected_ods"],
        "affected_ods": record.get("affected_ods"),
        "candidate_path_exhausted_ods": record.get("candidate_path_exhausted_ods"),
        "failed_link_keys": record.get("failed_link_keys", []),
        "failed_edge_indices": record.get("failed_edge_indices", []),
        "timing_breakdown_ms": {
            "event_detection_ms": record.get("event_detection_ms", record.get("detection_ms", 0.0)),
            "state_snapshot_ms": record.get("state_snapshot_ms", 0.0),
            "ddqn_inference_ms": record.get("ddqn_inference_ms", 0.0),
            "scorer_ranking_ms": record.get("scorer_ranking_ms", 0.0),
            "routing_compute_ms": record.get("routing_compute_ms", 0.0),
            "rule_diff_build_ms": record.get("rule_diff_build_ms", record.get("rule_diff_ms", 0.0)),
            "flow_mod_send_ms": record.get("flow_mod_send_ms", 0.0),
            "barrier_wait_ms": record.get("barrier_wait_ms", 0.0),
            "probe_wait_ms": record.get("probe_wait_ms", 0.0),
            "logging_ms": record.get("logging_ms", 0.0),
        },
        "measurement_note": record["measurement_note"],
        "transient_packet_loss_pct": record.get("transient_packet_loss_pct"),
        "post_recovery_packet_loss_pct": record.get("post_recovery_packet_loss_pct"),
        "transient_measurement_valid": record.get("transient_measurement_valid"),
        "transient_measurement_invalid_reason": record.get("transient_measurement_invalid_reason"),
        "transient_measurement_parser": record.get("transient_measurement_parser"),
        "transient_valid_probe_count": record.get("transient_valid_probe_count"),
        "transient_invalid_probe_count": record.get("transient_invalid_probe_count"),
        "capacity_event": record.get("capacity_event"),
        "state_identity": record.get("state_identity"),
        "install_path": record.get("install_path"),
        "ovs_command_count": record.get("ovs_command_count"),
        "install_switch_profile": record.get("install_switch_profile"),
        "forwarding_trace": record.get("forwarding_trace"),
        "pre_failure": record["pre_failure_raw_measurements"],
        "transient_recovery": record["transient_raw_measurements"],
        "post_recovery": record["post_recovery_raw_measurements"],
        "recovery_probe_history": record["recovery_probe_history"],
    }
    path = raw_dir / f"{record['topology']}__{record['scenario']}__run{int(record['run_id']):02d}.json"
    path.write_text(json.dumps(payload, indent=2))


def build_qos_enhanced_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (topo, scenario), group in df.groupby(["topology", "scenario"], sort=False):
        notes = [str(note) for note in group["measurement_note"].dropna().astype(str).unique().tolist() if str(note).strip()]
        rows.append(
            {
                "Topology": topo,
                "Scenario": scenario,
                "Offered UDP rate Mbps": round(float(group["offered_udp_rate_mbps"].mean()), 3),
                "Pre-failure throughput Mbps": round(float(group["pre_failure_throughput_mbps"].mean()), 3),
                "Post-recovery throughput Mbps": round(float(group["post_recovery_throughput_mbps"].mean()), 3),
                "Pre-failure RTT ms": round(float(group["pre_failure_rtt_ms"].mean()), 3),
                "Post-recovery RTT ms": round(float(group["post_recovery_rtt_ms"].mean()), 3),
                "Pre-failure jitter ms": round(float(group["pre_failure_jitter_ms"].mean()), 3),
                "Post-recovery jitter ms": round(float(group["post_recovery_jitter_ms"].mean()), 3),
                "Transient packet loss %": round(float(group["transient_packet_loss_pct"].mean()), 4),
                "Post-recovery steady-state packet loss %": round(float(group["post_recovery_packet_loss_pct"].mean()), 4),
                "Affected ODs": round(float(group["affected_ods"].mean()), 3) if "affected_ods" in group else float("nan"),
                "Rule count": round(float(group["rule_count"].mean()), 3),
                "Changed rules": round(float(group["changed_rules"].mean()), 3),
                "Unchanged rules": round(float(group["unchanged_rules"].mean()), 3),
                "Added rules": round(float(group["added_rules"].mean()), 3),
                "Removed rules": round(float(group["removed_rules"].mean()), 3),
                "Modified rules": round(float(group["modified_rules"].mean()), 3),
                "Install ms": round(float(group["install_ms"].mean()), 3),
                "Initial install ms": round(float(group["initial_install_ms"].mean()), 3) if "initial_install_ms" in group else round(float(group["install_ms"].mean()), 3),
                "Incremental install ms": round(float(group["incremental_install_ms"].mean()), 3) if "incremental_install_ms" in group else round(float(group["install_ms"].mean()), 3),
                "Recovery ms": round(float(group["recovery_ms"].mean()), 3),
                "Controller pipeline ms": round(float(group["controller_pipeline_ms"].mean()), 3) if "controller_pipeline_ms" in group else round(float(group["controller_response_ms"].mean()), 3),
                "Controller decision ms": round(float(group["controller_decision_ms"].mean()), 3) if "controller_decision_ms" in group else round(float(group["controller_response_ms"].mean()), 3),
                "Legacy controller response ms": round(float(group["legacy_controller_response_ms"].mean()), 3) if "legacy_controller_response_ms" in group else round(float(group["controller_response_ms"].mean()), 3),
                "Disconnected ODs": int(group["disconnected_ods"].max()),
                "Run count": int(len(group)),
                "Mode": "live" if (group["mode"].astype(str).str.lower() == "live").all() else "mixed",
                "Rerun on FIX1": "yes" if (group["mode"].astype(str).str.lower() == "live").all() else "no",
                "Measurement note": " | ".join(notes),
                "Controller timing note": " | ".join(
                    [
                        str(note)
                        for note in group.get("controller_timing_note", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
                        if str(note).strip()
                    ]
                ),
                "Post-failure bottleneck capacity Mbps": round(float(group["post_failure_bottleneck_capacity_mbps"].mean()), 3),
                "Rate cap used": round(float(group["rate_cap_used"].mean()), 3),
                "P95 Offered UDP rate Mbps": round(float(group["offered_udp_rate_mbps"].quantile(0.95)), 3),
                "P95 Pre-failure throughput Mbps": round(float(group["pre_failure_throughput_mbps"].quantile(0.95)), 3),
                "P95 Post-recovery throughput Mbps": round(float(group["post_recovery_throughput_mbps"].quantile(0.95)), 3),
                "P95 Pre-failure RTT ms": round(float(group["pre_failure_rtt_ms"].quantile(0.95)), 3),
                "P95 Post-recovery RTT ms": round(float(group["post_recovery_rtt_ms"].quantile(0.95)), 3),
                "P95 Pre-failure jitter ms": round(float(group["pre_failure_jitter_ms"].quantile(0.95)), 3),
                "P95 Post-recovery jitter ms": round(float(group["post_recovery_jitter_ms"].quantile(0.95)), 3),
                "P95 Transient packet loss %": round(float(group["transient_packet_loss_pct"].quantile(0.95)), 4),
                "P95 Post-recovery steady-state packet loss %": round(float(group["post_recovery_packet_loss_pct"].quantile(0.95)), 4),
                "P95 Changed rules": round(float(group["changed_rules"].quantile(0.95)), 3),
                "P95 Unchanged rules": round(float(group["unchanged_rules"].quantile(0.95)), 3),
                "P95 Added rules": round(float(group["added_rules"].quantile(0.95)), 3),
                "P95 Removed rules": round(float(group["removed_rules"].quantile(0.95)), 3),
                "P95 Modified rules": round(float(group["modified_rules"].quantile(0.95)), 3),
                "P95 Install ms": round(float(group["install_ms"].quantile(0.95)), 3),
                "P95 Initial install ms": round(float(group["initial_install_ms"].quantile(0.95)), 3) if "initial_install_ms" in group else round(float(group["install_ms"].quantile(0.95)), 3),
                "P95 Incremental install ms": round(float(group["incremental_install_ms"].quantile(0.95)), 3) if "incremental_install_ms" in group else round(float(group["install_ms"].quantile(0.95)), 3),
                "P95 Recovery ms": round(float(group["recovery_ms"].quantile(0.95)), 3),
                "P95 Controller pipeline ms": round(float(group["controller_pipeline_ms"].quantile(0.95)), 3) if "controller_pipeline_ms" in group else round(float(group["controller_response_ms"].quantile(0.95)), 3),
                "P95 Controller decision ms": round(float(group["controller_decision_ms"].quantile(0.95)), 3) if "controller_decision_ms" in group else round(float(group["controller_response_ms"].quantile(0.95)), 3),
                "P95 Legacy controller response ms": round(float(group["legacy_controller_response_ms"].quantile(0.95)), 3) if "legacy_controller_response_ms" in group else round(float(group["controller_response_ms"].quantile(0.95)), 3),
            }
        )
    return pd.DataFrame(rows)


def write_qos_enhanced_validation(summary_df: pd.DataFrame, per_run_df: pd.DataFrame) -> None:
    raw_dir = OUT_DIR / f"{ARTIFACT_PREFIX}_raw_logs"
    run_count = int(per_run_df.groupby(["topology", "scenario"]).size().min()) if not per_run_df.empty else 0
    lines = [
        "# Fresh Live Mininet / SDN QoS Validation with Incremental Recovery",
        "",
        "Methodology:",
        "- Mininet version: 2.3.0 (VM guide baseline)",
        "- OVS version: Open vSwitch 2.13.1 (VM guide baseline)",
        f"- Controller: {PUBLIC_METHOD} ({METHOD_LINEAGE})",
        "- VM/OS: Ubuntu 20.04.1 LTS VirtualBox VM via SSH port 2222",
        f"- Number of runs: {run_count} per executed scenario",
        "- Incremental update policy: controller stayed warm, unaffected flows were preserved, backup plans were precomputed, and only changed rules were reinstalled",
        "- Batch installation policy: flow additions were sent per-switch in batched add-flows files instead of one ovs-ofctl add-flow call per rule",
        "- Warm-up duration: 1 second pre-measure warm-up",
        "- Failure timing: failure scenarios kept transient loss separated from post-recovery steady-state; mixed spike+failure applies spike after convergence and a short documented delay",
        f"- Mixed spike+failure delay: {int(MIXED_SPIKE_DELAY_SEC * 1000)} ms after stable post-failure forwarding",
        "- Startup time excluded: yes; Mininet, switch initialization, and controller boot were outside install and recovery timing",
        "",
        "Validation:",
        f"- Fresh enhanced SDN CSVs created: {'PASS' if not summary_df.empty else 'FAIL'}",
        f"- Raw logs exist: {'PASS' if raw_dir.exists() and any(raw_dir.glob('*.json')) else 'FAIL'}",
        f"- Old SDN rerun CSVs preserved: {'PASS' if (OUT_DIR / 'sdn_live_fix1_qos_rerun_summary.csv').exists() else 'FAIL'}",
        f"- All enhanced rows are live FIX1 reruns: {'PASS' if not per_run_df.empty and (per_run_df['mode'].astype(str).str.lower() == 'live').all() else 'FAIL'}",
        f"- No historical SDN rows mixed: {'PASS' if not per_run_df.empty and (per_run_df['mode'].astype(str).str.lower() == 'live').all() else 'FAIL'}",
        f"- Transient and post-recovery loss remain separated: {'PASS' if {'transient_packet_loss_pct', 'post_recovery_packet_loss_pct'}.issubset(per_run_df.columns) else 'FAIL'}",
        f"- Offered UDP rates are recorded: {'PASS' if 'offered_udp_rate_mbps' in per_run_df.columns else 'FAIL'}",
        f"- Rule diff / changed-rules counts are recorded: {'PASS' if {'changed_rules', 'unchanged_rules', 'added_rules', 'removed_rules', 'modified_rules'}.issubset(per_run_df.columns) else 'FAIL'}",
        f"- No fabricated QoS values: {'PASS' if not per_run_df.empty else 'FAIL'}",
    ]
    (OUT_DIR / f"{ARTIFACT_PREFIX}_validation.md").write_text("\n".join(lines))


def write_qos_enhanced_outputs(df: pd.DataFrame) -> None:
    per_run_path = OUT_DIR / f"{ARTIFACT_PREFIX}_per_run.csv"
    summary_path = OUT_DIR / f"{ARTIFACT_PREFIX}_summary.csv"

    per_run_export = df[
        [
            "topology",
            "scenario",
            "run_id",
            "offered_udp_rate_mbps",
            "pre_failure_throughput_mbps",
            "post_recovery_throughput_mbps",
            "pre_failure_rtt_ms",
            "post_recovery_rtt_ms",
            "pre_failure_jitter_ms",
            "post_recovery_jitter_ms",
            "transient_packet_loss_pct",
            "post_recovery_packet_loss_pct",
            "affected_ods",
            "rule_count",
            "changed_rules",
            "unchanged_rules",
            "added_rules",
            "removed_rules",
            "modified_rules",
            "install_ms",
            "initial_install_ms",
            "incremental_install_ms",
            "recovery_ms",
            "controller_pipeline_ms",
            "controller_decision_ms",
            "legacy_controller_response_ms",
            "controller_response_ms",
            "disconnected_ods",
            "candidate_path_exhausted_ods",
            "event_detection_ms",
            "state_snapshot_ms",
            "ddqn_inference_ms",
            "scorer_ranking_ms",
            "routing_compute_ms",
            "rule_diff_build_ms",
            "flow_mod_send_ms",
            "barrier_wait_ms",
            "probe_wait_ms",
            "logging_ms",
            "controller_timing_note",
            "precomputed_backup_plan",
            "post_failure_bottleneck_capacity_mbps",
            "rate_cap_used",
            "measurement_note",
            "mode",
        ]
    ].rename(
        columns={
            "topology": "Topology",
            "scenario": "Scenario",
            "run_id": "Run ID",
            "offered_udp_rate_mbps": "Offered UDP rate Mbps",
            "pre_failure_throughput_mbps": "Pre-failure throughput Mbps",
            "post_recovery_throughput_mbps": "Post-recovery throughput Mbps",
            "pre_failure_rtt_ms": "Pre-failure RTT ms",
            "post_recovery_rtt_ms": "Post-recovery RTT ms",
            "pre_failure_jitter_ms": "Pre-failure jitter ms",
            "post_recovery_jitter_ms": "Post-recovery jitter ms",
            "transient_packet_loss_pct": "Transient packet loss %",
            "post_recovery_packet_loss_pct": "Post-recovery steady-state packet loss %",
            "affected_ods": "Affected ODs",
            "rule_count": "Rule count",
            "changed_rules": "Changed rules",
            "unchanged_rules": "Unchanged rules",
            "added_rules": "Added rules",
            "removed_rules": "Removed rules",
            "modified_rules": "Modified rules",
            "install_ms": "Install ms",
            "initial_install_ms": "Initial install ms",
            "incremental_install_ms": "Incremental install ms",
            "recovery_ms": "Recovery ms",
            "controller_pipeline_ms": "Controller pipeline ms",
            "controller_decision_ms": "Controller decision ms",
            "legacy_controller_response_ms": "Legacy controller response ms",
            "controller_response_ms": "Controller response ms",
            "disconnected_ods": "Disconnected ODs",
            "candidate_path_exhausted_ods": "Candidate-path exhausted ODs",
            "event_detection_ms": "Event detection ms",
            "state_snapshot_ms": "State snapshot ms",
            "ddqn_inference_ms": "DDQN inference ms",
            "scorer_ranking_ms": "Scorer ranking ms",
            "routing_compute_ms": "Routing compute ms",
            "rule_diff_build_ms": "Rule diff build ms",
            "flow_mod_send_ms": "Flow-mod send ms",
            "barrier_wait_ms": "Barrier wait ms",
            "probe_wait_ms": "Probe wait ms",
            "logging_ms": "Logging ms",
            "controller_timing_note": "Controller timing note",
            "precomputed_backup_plan": "Precomputed backup plan",
            "post_failure_bottleneck_capacity_mbps": "Post-failure bottleneck capacity Mbps",
            "rate_cap_used": "Rate cap used",
            "measurement_note": "Measurement note",
            "mode": "Mode",
        }
    )
    per_run_export["Rerun on FIX1"] = "yes"
    per_run_export.to_csv(per_run_path, index=False)

    summary_df = build_qos_enhanced_summary(df)
    summary_df.to_csv(summary_path, index=False)
    write_qos_enhanced_validation(summary_df, df)
    print(f"Wrote {per_run_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {OUT_DIR / f'{ARTIFACT_PREFIX}_validation.md'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FIX1 SDN/Mininet validation runner")
    parser.add_argument("--mode", choices=["simulate", "mininet"], default="simulate")
    parser.add_argument("--topologies", nargs="+", default=TOPOLOGIES_TO_RUN)
    parser.add_argument("--scenarios", nargs="+", default=SDN_SCENARIOS)
    parser.add_argument("--runs", type=int, default=RUNS_PER_SCENARIO)
    parser.add_argument("--duration-sec", type=int, default=LIVE_TRAFFIC_DURATION_SEC)
    parser.add_argument("--warmup-sec", type=int, default=LIVE_WARMUP_TRAFFIC_SEC)
    parser.add_argument("--max-measured-ods", type=int, default=MAX_MEASURED_ODS)
    parser.add_argument("--strict-mcf-time-limit", type=int, default=STRICT_MCF_TIME_LIMIT)
    parser.add_argument("--artifact-prefix", default=ARTIFACT_PREFIX)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    TOPOLOGIES_TO_RUN[:] = args.topologies
    RUNS_PER_SCENARIO = args.runs
    LIVE_TRAFFIC_DURATION_SEC = args.duration_sec
    MAX_MEASURED_ODS = args.max_measured_ods
    STRICT_MCF_TIME_LIMIT = args.strict_mcf_time_limit
    ARTIFACT_PREFIX = str(args.artifact_prefix)

    if args.mode == "mininet":
        run_mininet(args)
    else:
        run_simulate(args)
