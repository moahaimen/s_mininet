"""LP solvers for hybrid path-based TE and full MCF reference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pulp

from te.baselines import clone_splits
from te.paths import PathLibrary
from te.simulator import RoutingResult, apply_routing

EPS = 1e-12


def _get_solver(msg: bool = False, time_limit_sec: int = 60, seed_override: int | None = None):
    """Return the best available LP solver, preferring HiGHS over CBC.

    Seed selection (in order of precedence):
      1. seed_override argument (used by multi-restart loop)
      2. CBC_SEED env var (used by single-solve and as base for multi-restart)
      3. default 42

    L2b — pinning seed makes CBC/HiGHS tie-break deterministic across platforms.
    """
    if seed_override is not None:
        _seed = str(int(seed_override))
    else:
        _seed = os.environ.get("CBC_SEED", "42")
        try:
            int(_seed)
        except ValueError:
            _seed = "42"
    try:
        _highs = pulp.getSolver(
            "HiGHS", msg=msg, timeLimit=int(time_limit_sec),
            options=[f"random_seed={_seed}"],
        )
        if _highs.available():
            return _highs
    except Exception:
        pass
    return pulp.PULP_CBC_CMD(
        msg=msg, timeLimit=int(time_limit_sec), threads=1,
        options=["randomCbcSeed", _seed],
    )


@dataclass
class HybridLPResult:
    splits: List[np.ndarray]
    routing: RoutingResult
    status: str


@dataclass
class FullMCFResult:
    mlu: float
    link_loads: np.ndarray
    status: str
    edge_flows_by_od: List[Dict[int, float]]


@dataclass
class PathLPResult:
    """Result of the candidate-path all-OD LP.

    Distinct from FullMCFResult: this LP is constrained to the precomputed
    candidate-path library, so its MLU is an UPPER bound on the true LP
    optimum.  Used for the PR_path_opt metric on topologies where the full
    MCF LP cannot be solved.
    """
    mlu: float
    link_loads: np.ndarray
    status: str
    splits: List[np.ndarray]  # per-OD K-path split fractions
    edge_flows_by_od: List[Dict[int, float]]


def solve_all_od_path_lp(
    tm_vector: np.ndarray,
    path_library: PathLibrary,
    capacities: np.ndarray,
    time_limit_sec: int = 60,
    solver_msg: bool = False,
) -> PathLPResult:
    """Minimise MLU subject to all OD demand routed over candidate paths only.

    Variables:  f[od, p] in [0, 1]  -- fraction of demand[od] routed via path p
                U                    -- max link utilisation

    Objective:  minimise U

    Constraints:
        sum_p f[od, p] = 1                     for every OD with demand > 0
        sum_{od, p: e in path}
            demand[od] * f[od, p] <= U * cap[e] for every edge e

    Returns PathLPResult.  Always within the candidate-path library, so
    PR_path_opt = (this MLU) / (method MLU) <= 1.0001 must hold for any
    method that uses the SAME path library.
    """
    num_edges = int(len(capacities))
    num_od = int(len(tm_vector))

    prob = pulp.LpProblem("all_od_path_lp", pulp.LpMinimize)
    U = pulp.LpVariable("U", lowBound=0)

    # f[od][p]
    f: List[List[pulp.LpVariable]] = []
    for od in range(num_od):
        paths = path_library.edge_idx_paths_by_od[od]
        if not paths:
            f.append([])
            continue
        row = []
        for p_idx in range(len(paths)):
            row.append(pulp.LpVariable(f"f_{od}_{p_idx}", lowBound=0, upBound=1))
        f.append(row)

    # Edge load expressions
    edge_load: Dict[int, List[Tuple[float, pulp.LpVariable]]] = {
        e: [] for e in range(num_edges)
    }

    # Demand-conservation constraints
    for od in range(num_od):
        demand = float(tm_vector[od])
        if demand <= 0:
            continue
        paths = path_library.edge_idx_paths_by_od[od]
        if not paths:
            continue
        prob += pulp.lpSum(f[od]) == 1, f"demand_{od}"
        for p_idx, edge_path in enumerate(paths):
            for e in edge_path:
                edge_load[e].append((demand, f[od][p_idx]))

    for e in range(num_edges):
        cap = float(capacities[e])
        if cap <= 0:
            continue
        terms = edge_load[e]
        if not terms:
            continue
        prob += (
            pulp.lpSum(coef * var for coef, var in terms) <= U * cap
        ), f"cap_{e}"

    prob += U

    solver = _get_solver(msg=solver_msg, time_limit_sec=time_limit_sec)
    status = prob.solve(solver)
    status_str = pulp.LpStatus[status]

    splits: List[np.ndarray] = []
    edge_flows_by_od: List[Dict[int, float]] = [{} for _ in range(num_od)]
    link_loads = np.zeros(num_edges, dtype=float)

    for od in range(num_od):
        demand = float(tm_vector[od])
        paths = path_library.edge_idx_paths_by_od[od]
        if not paths or not f[od]:
            splits.append(np.array([], dtype=float))
            continue
        sp = np.zeros(len(paths), dtype=float)
        for p_idx, var in enumerate(f[od]):
            v = var.value()
            sp[p_idx] = float(v) if v is not None else 0.0
        s = sp.sum()
        if s > EPS and abs(s - 1.0) > 1e-6 and demand > 0:
            sp = sp / s
        splits.append(sp)

        if demand > 0:
            for p_idx, edge_path in enumerate(paths):
                flow = demand * float(sp[p_idx])
                if flow <= 0:
                    continue
                for e in edge_path:
                    link_loads[e] += flow
                    edge_flows_by_od[od][e] = (
                        edge_flows_by_od[od].get(e, 0.0) + flow
                    )

    util = link_loads / np.maximum(capacities, EPS)
    mlu_value = float(np.max(util)) if num_edges else 0.0

    u_val = U.value()
    if u_val is not None and status_str.lower() == "optimal":
        if abs(float(u_val) - mlu_value) > 1e-3:
            status_str = "PrimalMismatch"

    return PathLPResult(
        mlu=mlu_value,
        link_loads=link_loads,
        status=status_str,
        splits=splits,
        edge_flows_by_od=edge_flows_by_od,
    )


def _build_background_load(
    tm_vector: np.ndarray,
    base_splits: Sequence[np.ndarray],
    path_library: PathLibrary,
    selected_set: set[int],
    num_edges: int,
) -> np.ndarray:
    load = np.zeros(num_edges, dtype=float)
    for od_idx, demand in enumerate(tm_vector):
        if demand <= 0 or od_idx in selected_set:
            continue

        paths = path_library.edge_idx_paths_by_od[od_idx]
        if not paths:
            continue

        splits = np.asarray(base_splits[od_idx], dtype=float)
        if splits.size == 0:
            continue

        split_sum = float(np.sum(splits))
        if split_sum <= EPS:
            continue

        splits = splits / split_sum
        for path_idx, frac in enumerate(splits):
            if frac <= 0:
                continue
            flow = float(demand) * float(frac)
            for edge_idx in paths[path_idx]:
                load[edge_idx] += flow

    return load


def _extract_splits_from_lp(
    tm_vector: np.ndarray,
    selected_set: set,
    base_splits: Sequence[np.ndarray],
    path_library: PathLibrary,
    flow_vars: Dict[Tuple[int, int], pulp.LpVariable],
) -> List[np.ndarray]:
    """Read the LP solver's variable values back into a list of split vectors."""
    splits = clone_splits(base_splits)
    for od_idx in sorted(selected_set):
        demand = float(tm_vector[od_idx])
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if demand <= 0 or not paths:
            continue
        vec = np.zeros(len(paths), dtype=float)
        for path_idx in range(len(paths)):
            var = flow_vars.get((od_idx, path_idx))
            if var is None:
                continue
            vec[path_idx] = max(float(var.value() or 0.0), 0.0)
        if demand > EPS:
            vec /= demand
        vec_sum = float(np.sum(vec))
        if vec_sum > EPS:
            vec /= vec_sum
            splits[od_idx] = vec
    return splits


def _db_l1_demand_weighted(
    prev_splits: Sequence[np.ndarray] | None,
    new_splits: Sequence[np.ndarray],
    tm_vector: np.ndarray,
) -> float:
    if prev_splits is None:
        return 0.0
    total = 0.0
    for od_idx, demand in enumerate(tm_vector):
        d = float(demand)
        if d <= 0:
            continue
        if od_idx >= len(prev_splits) or od_idx >= len(new_splits):
            continue
        prev = np.asarray(prev_splits[od_idx], dtype=float)
        new = np.asarray(new_splits[od_idx], dtype=float)
        if prev.size != new.size:
            total += d * 2.0
            continue
        total += d * float(np.sum(np.abs(prev - new)))
    norm = float(np.sum(np.maximum(tm_vector, 0.0)))
    return total / max(norm, EPS)


def solve_selected_path_lp(
    tm_vector: np.ndarray,
    selected_ods: Sequence[int],
    base_splits: Sequence[np.ndarray],
    path_library: PathLibrary,
    capacities: np.ndarray,
    time_limit_sec: int = 20,
    solver_msg: bool = False,
    prev_splits: Sequence[np.ndarray] | None = None,
    n_restarts: int | None = None,
) -> HybridLPResult:
    num_edges = int(capacities.size)
    selected_set = {
        int(od_idx)
        for od_idx in selected_ods
        if 0 <= int(od_idx) < len(tm_vector) and tm_vector[int(od_idx)] > 0
    }

    if not selected_set:
        splits = clone_splits(base_splits)
        routing = apply_routing(tm_vector, splits, path_library, capacities)
        return HybridLPResult(splits=splits, routing=routing, status="NoSelection")

    if n_restarts is None:
        try:
            n_restarts = int(os.environ.get("LP_RESTARTS", "1"))
        except ValueError:
            n_restarts = 1
    multi_restart_active = (n_restarts > 1) and (prev_splits is not None)

    background = _build_background_load(tm_vector, base_splits, path_library, selected_set, num_edges)

    model = pulp.LpProblem("hybrid_te_selected_lp", pulp.LpMinimize)
    U = pulp.LpVariable("U", lowBound=0.0)

    flow_vars: Dict[Tuple[int, int], pulp.LpVariable] = {}
    incidence: List[List[pulp.LpVariable]] = [[] for _ in range(num_edges)]

    for od_idx in sorted(selected_set):
        demand = float(tm_vector[od_idx])
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if demand <= 0 or not paths:
            continue

        per_od_vars = []
        for path_idx, edge_path in enumerate(paths):
            var = pulp.LpVariable(f"f_{od_idx}_{path_idx}", lowBound=0.0)
            flow_vars[(od_idx, path_idx)] = var
            per_od_vars.append(var)
            for edge_idx in edge_path:
                incidence[edge_idx].append(var)

        model += pulp.lpSum(per_od_vars) == demand, f"demand_{od_idx}"

    for edge_idx in range(num_edges):
        model += (
            background[edge_idx] + pulp.lpSum(incidence[edge_idx]) <= U * float(capacities[edge_idx]),
            f"cap_{edge_idx}",
        )

    model += U

    if not multi_restart_active:
        solver = _get_solver(msg=solver_msg, time_limit_sec=int(time_limit_sec))
        status_code = model.solve(solver)
        status = pulp.LpStatus.get(status_code, "Unknown")

        if status not in {"Optimal", "Not Solved", "Undefined"}:
            splits = clone_splits(base_splits)
            routing = apply_routing(tm_vector, splits, path_library, capacities)
            return HybridLPResult(splits=splits, routing=routing, status=status)

        splits = _extract_splits_from_lp(tm_vector, selected_set, base_splits, path_library, flow_vars)
        routing = apply_routing(tm_vector, splits, path_library, capacities)
        return HybridLPResult(splits=splits, routing=routing, status=status)

    try:
        base_seed = int(os.environ.get("CBC_SEED", "42"))
    except ValueError:
        base_seed = 42

    best_splits = None
    best_db = float("inf")
    best_status = "Unknown"
    per_restart_time = max(2, int(time_limit_sec) // n_restarts) if n_restarts > 0 else int(time_limit_sec)
    for restart_idx in range(n_restarts):
        seed = base_seed + 7 * restart_idx
        solver = _get_solver(msg=solver_msg,
                             time_limit_sec=per_restart_time,
                             seed_override=seed)
        try:
            status_code = model.solve(solver)
            status = pulp.LpStatus.get(status_code, "Unknown")
        except Exception:
            continue
        if status not in {"Optimal", "Not Solved", "Undefined"}:
            continue
        candidate_splits = _extract_splits_from_lp(
            tm_vector, selected_set, base_splits, path_library, flow_vars,
        )
        db = _db_l1_demand_weighted(prev_splits, candidate_splits, tm_vector)
        if db < best_db:
            best_db = db
            best_splits = candidate_splits
            best_status = status

    if best_splits is None:
        splits = clone_splits(base_splits)
        routing = apply_routing(tm_vector, splits, path_library, capacities)
        return HybridLPResult(splits=splits, routing=routing, status="AllRestartsFailed")

    routing = apply_routing(tm_vector, best_splits, path_library, capacities)
    return HybridLPResult(splits=best_splits, routing=routing, status=best_status)


def solve_selected_path_lp_min_db(
    tm_vector: np.ndarray,
    selected_ods: Sequence[int],
    base_splits: Sequence[np.ndarray],
    path_library: PathLibrary,
    capacities: np.ndarray,
    prev_splits: Sequence[np.ndarray] | None,
    U_star: float,
    epsilon: float = 0.02,
    time_limit_sec: int = 20,
    solver_msg: bool = False,
) -> HybridLPResult:
    num_edges = int(capacities.size)
    selected_set = {
        int(od_idx)
        for od_idx in selected_ods
        if 0 <= int(od_idx) < len(tm_vector) and tm_vector[int(od_idx)] > 0
    }

    if not selected_set or prev_splits is None:
        splits = clone_splits(base_splits)
        routing = apply_routing(tm_vector, splits, path_library, capacities)
        return HybridLPResult(splits=splits, routing=routing, status="NoSelection")

    background = _build_background_load(tm_vector, base_splits, path_library, selected_set, num_edges)
    cap_bound = (1.0 + float(epsilon)) * float(U_star)

    model = pulp.LpProblem("min_db_selected_lp", pulp.LpMinimize)
    flow_vars: Dict[Tuple[int, int], pulp.LpVariable] = {}
    aux_vars: List[pulp.LpVariable] = []
    incidence: List[List[pulp.LpVariable]] = [[] for _ in range(num_edges)]

    for od_idx in sorted(selected_set):
        demand = float(tm_vector[od_idx])
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if demand <= 0 or not paths:
            continue
        prev_vec = np.asarray(prev_splits[od_idx], dtype=float) if od_idx < len(prev_splits) else np.zeros(len(paths))
        prev_pad = np.zeros(len(paths), dtype=float)
        prev_pad[: min(prev_vec.size, len(paths))] = prev_vec[: min(prev_vec.size, len(paths))]
        prev_f = prev_pad * demand

        per_od_vars = []
        for path_idx, edge_path in enumerate(paths):
            f = pulp.LpVariable(f"f_{od_idx}_{path_idx}", lowBound=0.0)
            flow_vars[(od_idx, path_idx)] = f
            per_od_vars.append(f)
            for edge_idx in edge_path:
                incidence[edge_idx].append(f)
            y = pulp.LpVariable(f"y_{od_idx}_{path_idx}", lowBound=0.0)
            aux_vars.append(y)
            model += y >= f - float(prev_f[path_idx]), f"abs_pos_{od_idx}_{path_idx}"
            model += y >= float(prev_f[path_idx]) - f, f"abs_neg_{od_idx}_{path_idx}"

        model += pulp.lpSum(per_od_vars) == demand, f"demand_{od_idx}"

    for edge_idx in range(num_edges):
        model += (
            background[edge_idx] + pulp.lpSum(incidence[edge_idx]) <= cap_bound * float(capacities[edge_idx]),
            f"cap_{edge_idx}",
        )

    model += pulp.lpSum(aux_vars)

    solver = _get_solver(msg=solver_msg, time_limit_sec=int(time_limit_sec))
    status_code = model.solve(solver)
    status = pulp.LpStatus.get(status_code, "Unknown")

    if status not in {"Optimal", "Not Solved", "Undefined"}:
        splits = clone_splits(base_splits)
        routing = apply_routing(tm_vector, splits, path_library, capacities)
        return HybridLPResult(splits=splits, routing=routing, status=f"MinDBFailed:{status}")

    splits = _extract_splits_from_lp(tm_vector, selected_set, base_splits, path_library, flow_vars)
    routing = apply_routing(tm_vector, splits, path_library, capacities)
    return HybridLPResult(splits=splits, routing=routing, status=status)


def solve_selected_path_lp_dbbudget(
    tm_vector: np.ndarray,
    selected_ods: Sequence[int],
    base_splits: Sequence[np.ndarray],
    path_library: PathLibrary,
    capacities: np.ndarray,
    prev_splits: Sequence[np.ndarray] | None,
    db_budget: float,
    db_weight: float = 1e-3,
    time_limit_sec: int = 20,
    solver_msg: bool = False,
) -> HybridLPResult:
    num_edges = int(capacities.size)
    selected_set = {
        int(od_idx)
        for od_idx in selected_ods
        if 0 <= int(od_idx) < len(tm_vector) and tm_vector[int(od_idx)] > 0
    }

    if not selected_set:
        splits = clone_splits(base_splits)
        routing = apply_routing(tm_vector, splits, path_library, capacities)
        return HybridLPResult(splits=splits, routing=routing, status="NoSelection")

    total_demand = float(np.sum(np.maximum(tm_vector, 0.0)))
    background = _build_background_load(tm_vector, base_splits, path_library, selected_set, num_edges)

    budget_active = prev_splits is not None and total_demand > EPS

    fixed_sum = 0.0
    rhs = float("inf")
    if budget_active:
        for od_idx, demand in enumerate(tm_vector):
            d = float(demand)
            if d <= 0 or od_idx in selected_set:
                continue
            paths = path_library.edge_idx_paths_by_od[od_idx]
            if not paths:
                continue
            dim = len(paths)
            prev_vec = np.asarray(prev_splits[od_idx], dtype=float) if od_idx < len(prev_splits) else np.zeros(dim)
            base_vec = np.asarray(base_splits[od_idx], dtype=float) if od_idx < len(base_splits) else np.zeros(dim)
            prev_pad = np.zeros(dim, dtype=float); prev_pad[: min(prev_vec.size, dim)] = prev_vec[: min(prev_vec.size, dim)]
            base_pad = np.zeros(dim, dtype=float); base_pad[: min(base_vec.size, dim)] = base_vec[: min(base_vec.size, dim)]
            fixed_sum += d * float(np.sum(np.abs(prev_pad - base_pad)))
        rhs = 2.0 * total_demand * float(db_budget) - fixed_sum
        if rhs < 0.0:
            splits = clone_splits(base_splits)
            routing = apply_routing(tm_vector, splits, path_library, capacities)
            return HybridLPResult(splits=splits, routing=routing, status="BudgetInfeasible")

    model = pulp.LpProblem("dbbudget_selected_lp", pulp.LpMinimize)
    U = pulp.LpVariable("U", lowBound=0.0)
    flow_vars: Dict[Tuple[int, int], pulp.LpVariable] = {}
    aux_vars: List[pulp.LpVariable] = []
    incidence: List[List[pulp.LpVariable]] = [[] for _ in range(num_edges)]

    for od_idx in sorted(selected_set):
        demand = float(tm_vector[od_idx])
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if demand <= 0 or not paths:
            continue

        if budget_active:
            prev_vec = np.asarray(prev_splits[od_idx], dtype=float) if od_idx < len(prev_splits) else np.zeros(len(paths))
            prev_pad = np.zeros(len(paths), dtype=float)
            prev_pad[: min(prev_vec.size, len(paths))] = prev_vec[: min(prev_vec.size, len(paths))]
            prev_f = prev_pad * demand
        else:
            prev_f = None

        per_od_vars = []
        for path_idx, edge_path in enumerate(paths):
            f = pulp.LpVariable(f"f_{od_idx}_{path_idx}", lowBound=0.0)
            flow_vars[(od_idx, path_idx)] = f
            per_od_vars.append(f)
            for edge_idx in edge_path:
                incidence[edge_idx].append(f)
            if budget_active:
                y = pulp.LpVariable(f"y_{od_idx}_{path_idx}", lowBound=0.0)
                aux_vars.append(y)
                model += y >= f - float(prev_f[path_idx]), f"abs_pos_{od_idx}_{path_idx}"
                model += y >= float(prev_f[path_idx]) - f, f"abs_neg_{od_idx}_{path_idx}"

        model += pulp.lpSum(per_od_vars) == demand, f"demand_{od_idx}"

    for edge_idx in range(num_edges):
        model += (
            background[edge_idx] + pulp.lpSum(incidence[edge_idx]) <= U * float(capacities[edge_idx]),
            f"cap_{edge_idx}",
        )

    if budget_active and aux_vars:
        model += pulp.lpSum(aux_vars) <= rhs, "db_budget"

    if budget_active and aux_vars and float(db_weight) > 0.0:
        model += U + float(db_weight) * pulp.lpSum(aux_vars)
    else:
        model += U

    solver = _get_solver(msg=solver_msg, time_limit_sec=int(time_limit_sec))
    status_code = model.solve(solver)
    status = pulp.LpStatus.get(status_code, "Unknown")

    if status not in {"Optimal", "Not Solved", "Undefined"}:
        splits = clone_splits(base_splits)
        routing = apply_routing(tm_vector, splits, path_library, capacities)
        return HybridLPResult(splits=splits, routing=routing, status=f"DBBudgetFailed:{status}")

    splits = _extract_splits_from_lp(tm_vector, selected_set, base_splits, path_library, flow_vars)
    routing = apply_routing(tm_vector, splits, path_library, capacities)
    return HybridLPResult(splits=splits, routing=routing, status=status)


def solve_full_mcf_min_mlu(
    tm_vector: np.ndarray,
    od_pairs: Sequence[Tuple[str, str]],
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str]],
    capacities: np.ndarray,
    time_limit_sec: int = 60,
    solver_msg: bool = False,
) -> FullMCFResult:
    active_ods = [idx for idx, demand in enumerate(tm_vector) if demand > 0]
    num_edges = len(edges)

    if not active_ods:
        return FullMCFResult(
            mlu=0.0,
            link_loads=np.zeros(num_edges, dtype=float),
            status="NoDemand",
            edge_flows_by_od=[{} for _ in od_pairs],
        )

    node_to_out: Dict[str, List[int]] = {node: [] for node in nodes}
    node_to_in: Dict[str, List[int]] = {node: [] for node in nodes}
    for edge_idx, (src, dst) in enumerate(edges):
        node_to_out[src].append(edge_idx)
        node_to_in[dst].append(edge_idx)

    model = pulp.LpProblem("full_mcf_min_mlu", pulp.LpMinimize)
    U = pulp.LpVariable("U", lowBound=0.0)

    x: Dict[Tuple[int, int], pulp.LpVariable] = {}
    edge_to_vars: List[List[pulp.LpVariable]] = [[] for _ in range(num_edges)]

    for od_idx in active_ods:
        for edge_idx in range(num_edges):
            var = pulp.LpVariable(f"x_{od_idx}_{edge_idx}", lowBound=0.0)
            x[(od_idx, edge_idx)] = var
            edge_to_vars[edge_idx].append(var)

    for edge_idx in range(num_edges):
        model += pulp.lpSum(edge_to_vars[edge_idx]) <= U * float(capacities[edge_idx]), f"cap_{edge_idx}"

    for od_idx in active_ods:
        src, dst = od_pairs[od_idx]
        demand = float(tm_vector[od_idx])

        for node in nodes:
            out_flow = pulp.lpSum(x[(od_idx, e_idx)] for e_idx in node_to_out[node])
            in_flow = pulp.lpSum(x[(od_idx, e_idx)] for e_idx in node_to_in[node])

            rhs = 0.0
            if node == src:
                rhs = demand
            elif node == dst:
                rhs = -demand

            model += out_flow - in_flow == rhs, f"flow_{od_idx}_{node}"

    model += U

    solver = _get_solver(msg=solver_msg, time_limit_sec=int(time_limit_sec))
    status_code = model.solve(solver)
    status = pulp.LpStatus.get(status_code, "Unknown")

    link_loads = np.zeros(num_edges, dtype=float)
    edge_flows_by_od: List[Dict[int, float]] = [{} for _ in od_pairs]

    for od_idx in active_ods:
        od_map: Dict[int, float] = {}
        for edge_idx in range(num_edges):
            var = x.get((od_idx, edge_idx))
            if var is None:
                continue
            value = max(float(var.value() or 0.0), 0.0)
            if value > EPS:
                od_map[edge_idx] = value
                link_loads[edge_idx] += value
        edge_flows_by_od[od_idx] = od_map

    util = link_loads / np.maximum(capacities, EPS)
    actual_mlu = float(np.max(util)) if util.size else 0.0
    u_val = float(U.value()) if U.value() is not None else float("inf")

    if status == "Optimal":
        sol_stat = getattr(model, "sol_status", None)
        if sol_stat is not None and sol_stat != 1:
            status = "TimeLimit"
            mlu = float("inf")
        elif u_val < float("inf") and actual_mlu > 0:
            rel_err = abs(actual_mlu - u_val) / max(u_val, 1e-9)
            if rel_err > 0.05:
                status = "PrimalMismatch"
                mlu = float("inf")
            else:
                mlu = u_val
        else:
            mlu = actual_mlu
    elif status in {"Not Solved", "Undefined"}:
        mlu = float("inf")
    else:
        mlu = float("inf")

    return FullMCFResult(
        mlu=mlu,
        link_loads=link_loads,
        status=status,
        edge_flows_by_od=edge_flows_by_od,
    )

import scipy.sparse as sp
from scipy.optimize import linprog
import numpy as np

def solve_selected_path_lp_dbbudget_scipy(
    tm_vector,
    selected_ods,
    base_splits,
    path_library,
    capacities,
    prev_splits,
    db_budget,
    db_weight = 1e-3,
    time_limit_sec = 20,
    solver_msg = False,
):
    EPS = 1e-12
    from te.simulator import apply_routing
    from te.baselines import clone_splits
    from te.lp_solver import _build_background_load

    num_edges = int(capacities.size)
    selected_set = {
        int(od_idx)
        for od_idx in selected_ods
        if 0 <= int(od_idx) < len(tm_vector) and tm_vector[int(od_idx)] > 0
    }

    if not selected_set:
        splits = clone_splits(base_splits)
        routing = apply_routing(tm_vector, splits, path_library, capacities)
        return HybridLPResult(splits=splits, routing=routing, status="NoSelection")

    total_demand = float(np.sum(np.maximum(tm_vector, 0.0)))
    background = _build_background_load(tm_vector, base_splits, path_library, selected_set, num_edges)

    budget_active = prev_splits is not None and total_demand > EPS

    fixed_sum = 0.0
    rhs = float("inf")
    if budget_active:
        for od_idx, demand in enumerate(tm_vector):
            d = float(demand)
            if d <= 0 or od_idx in selected_set:
                continue
            paths = path_library.edge_idx_paths_by_od[od_idx]
            if not paths:
                continue
            dim = len(paths)
            prev_vec = np.asarray(prev_splits[od_idx], dtype=float) if od_idx < len(prev_splits) else np.zeros(dim)
            base_vec = np.asarray(base_splits[od_idx], dtype=float) if od_idx < len(base_splits) else np.zeros(dim)
            prev_pad = np.zeros(dim, dtype=float); prev_pad[: min(prev_vec.size, dim)] = prev_vec[: min(prev_vec.size, dim)]
            base_pad = np.zeros(dim, dtype=float); base_pad[: min(base_vec.size, dim)] = base_vec[: min(base_vec.size, dim)]
            fixed_sum += d * float(np.sum(np.abs(prev_pad - base_pad)))
        rhs = 2.0 * total_demand * float(db_budget) - fixed_sum
        if rhs < 0.0:
            splits = clone_splits(base_splits)
            routing = apply_routing(tm_vector, splits, path_library, capacities)
            return HybridLPResult(splits=splits, routing=routing, status="BudgetInfeasible")

    num_f = 0
    od_path_map = []
    for od_idx in sorted(selected_set):
        paths = path_library.edge_idx_paths_by_od[od_idx]
        od_path_map.append((od_idx, len(paths)))
        num_f += len(paths)

    num_vars = 1 + num_f + (num_f if budget_active else 0)

    c = np.zeros(num_vars, dtype=float)
    c[0] = 1.0
    if budget_active and float(db_weight) > 0.0:
        c[1 + num_f :] = float(db_weight)

    bounds = [(0.0, None)] * num_vars

    A_eq = []
    b_eq = []
    row_idx = 0
    col_offset = 1
    I_eq, J_eq, V_eq = [], [], []
    for od_idx, num_p in od_path_map:
        demand = float(tm_vector[od_idx])
        for p in range(num_p):
            I_eq.append(row_idx)
            J_eq.append(col_offset + p)
            V_eq.append(1.0)
        b_eq.append(demand)
        row_idx += 1
        col_offset += num_p

    A_eq = sp.coo_matrix((V_eq, (I_eq, J_eq)), shape=(row_idx, num_vars))

    I_ub, J_ub, V_ub = [], [], []
    b_ub = []
    row_idx = 0
    col_offset = 1
    for od_idx, num_p in od_path_map:
        paths = path_library.edge_idx_paths_by_od[od_idx]
        for p, edge_path in enumerate(paths):
            for e in edge_path:
                I_ub.append(e)
                J_ub.append(col_offset + p)
                V_ub.append(1.0)
        col_offset += num_p

    for e in range(num_edges):
        I_ub.append(e)
        J_ub.append(0)
        V_ub.append(-float(capacities[e]))
        b_ub.append(-float(background[e]))

    row_idx = num_edges

    if budget_active:
        col_offset = 1
        y_offset = 1 + num_f
        for od_idx, num_p in od_path_map:
            demand = float(tm_vector[od_idx])
            prev_vec = np.asarray(prev_splits[od_idx], dtype=float) if od_idx < len(prev_splits) else np.zeros(num_p)
            prev_pad = np.zeros(num_p, dtype=float)
            prev_pad[: min(prev_vec.size, num_p)] = prev_vec[: min(prev_vec.size, num_p)]
            prev_f = prev_pad * demand

            for p in range(num_p):
                I_ub.append(row_idx)
                J_ub.append(col_offset + p)
                V_ub.append(1.0)
                I_ub.append(row_idx)
                J_ub.append(y_offset + p)
                V_ub.append(-1.0)
                b_ub.append(float(prev_f[p]))
                row_idx += 1

                I_ub.append(row_idx)
                J_ub.append(col_offset + p)
                V_ub.append(-1.0)
                I_ub.append(row_idx)
                J_ub.append(y_offset + p)
                V_ub.append(-1.0)
                b_ub.append(-float(prev_f[p]))
                row_idx += 1
            col_offset += num_p
            y_offset += num_p

        y_offset = 1 + num_f
        for p in range(num_f):
            I_ub.append(row_idx)
            J_ub.append(y_offset + p)
            V_ub.append(1.0)
        b_ub.append(rhs)
        row_idx += 1

    A_ub = sp.coo_matrix((V_ub, (I_ub, J_ub)), shape=(row_idx, num_vars))
    options = {"time_limit": float(time_limit_sec), "disp": solver_msg, "presolve": True}
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs", options=options)

    if not res.success:
        splits = clone_splits(base_splits)
        routing = apply_routing(tm_vector, splits, path_library, capacities)
        return HybridLPResult(splits=splits, routing=routing, status=f"DBBudgetFailed:{res.message}")

    x = res.x
    splits = clone_splits(base_splits)
    col_offset = 1
    for od_idx, num_p in od_path_map:
        demand = float(tm_vector[od_idx])
        f_vals = x[col_offset : col_offset + num_p]
        if demand > EPS:
            sp_vals = f_vals / demand
            sp_vals = np.maximum(sp_vals, 0.0)
            s = np.sum(sp_vals)
            if s > 0:
                sp_vals /= s
            splits[od_idx] = sp_vals
        col_offset += num_p

    routing = apply_routing(tm_vector, splits, path_library, capacities)
    return HybridLPResult(splits=splits, routing=routing, status="Optimal")
