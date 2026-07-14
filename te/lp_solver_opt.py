"""Independent routing-LP optimization for Target 5/6 (GEANT routing_compute).

Bypasses PuLP's slow Python object model by writing the CBC LP file directly and invoking the
SAME CBC binary/seed PuLP uses. The LP formulation is byte-for-byte the trusted one
(te.lp_solver.solve_selected_path_lp_dbbudget): minimize U (+ db_weight*sum(y) under the DB budget),
subject to per-OD demand, per-edge capacity (background + selected flow <= U*cap), and the
disturbance-budget/abs constraints. Non-LP logic (background load, rhs, split extraction, routing
evaluation) is reused from te.lp_solver so it is identical.

This is developed independently from the trusted branch (NOT Gemini's SciPy/HiGHS/_TEMPLATE_CACHE
code) and must pass the A/B equivalence gate (scripts/phase1_5/target5_ab_gate.py) before use.
"""
from __future__ import annotations
import os, subprocess, tempfile
from typing import Sequence, List
import numpy as np

from te.lp_solver import (
    _build_background_load, apply_routing, clone_splits, HybridLPResult, EPS,
)
from te.paths import PathLibrary

_CBC_PATH = None


def _cbc_path() -> str:
    global _CBC_PATH
    if _CBC_PATH is None:
        import pulp
        _CBC_PATH = pulp.PULP_CBC_CMD(msg=False).path
    return _CBC_PATH


def _fmt(x: float) -> str:
    return repr(float(x))


def solve_selected_path_lp_dbbudget_opt(
    tm_vector: np.ndarray,
    selected_ods: Sequence[int],
    base_splits: Sequence[np.ndarray],
    path_library: "PathLibrary",
    capacities: np.ndarray,
    prev_splits: Sequence[np.ndarray] = None,
    db_budget: float = 0.05,
    db_weight: float = 1e-3,
    time_limit_sec: int = 20,
    solver_msg: bool = False,
) -> HybridLPResult:
    num_edges = int(capacities.size)
    selected_set = {
        int(od) for od in selected_ods
        if 0 <= int(od) < len(tm_vector) and tm_vector[int(od)] > 0
    }
    if not selected_set:
        splits = clone_splits(base_splits)
        return HybridLPResult(splits=splits, routing=apply_routing(tm_vector, splits, path_library, capacities), status="NoSelection")

    total_demand = float(np.sum(np.maximum(tm_vector, 0.0)))
    background = _build_background_load(tm_vector, base_splits, path_library, selected_set, num_edges)
    budget_active = prev_splits is not None and total_demand > EPS

    # DB budget rhs (identical to trusted): fixed disturbance from non-selected ODs.
    rhs = float("inf")
    if budget_active:
        fixed_sum = 0.0
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
            prev_pad = np.zeros(dim); prev_pad[: min(prev_vec.size, dim)] = prev_vec[: min(prev_vec.size, dim)]
            base_pad = np.zeros(dim); base_pad[: min(base_vec.size, dim)] = base_vec[: min(base_vec.size, dim)]
            fixed_sum += d * float(np.sum(np.abs(prev_pad - base_pad)))
        rhs = 2.0 * total_demand * float(db_budget) - fixed_sum
        if rhs < 0.0:
            splits = clone_splits(base_splits)
            return HybridLPResult(splits=splits, routing=apply_routing(tm_vector, splits, path_library, capacities), status="BudgetInfeasible")

    ordered = sorted(selected_set)
    # Build the constraint/coefficient model. ROW order MUST match the trusted PuLP
    # construction order (per od: abs_pos/abs_neg per path, then demand; then cap per edge;
    # then db_budget) and COLUMN order MUST be lexicographic by variable name (PuLP's
    # model.variables() order) so CBC selects the identical vertex on degenerate LPs.
    row_order = []          # list of (rowname, sense)  sense in {'E','L','G'}
    rhs_map = {}            # rowname -> rhs value
    col_coef = {}           # varname -> {rowname: coef}
    obj_coef = {}           # varname -> objective coefficient

    def add_row(name, sense, r):
        row_order.append((name, sense)); rhs_map[name] = float(r)

    def add_coef(var, row, c):
        col_coef.setdefault(var, {})[row] = col_coef.get(var, {}).get(row, 0.0) + float(c)

    obj_coef["U"] = 1.0
    edge_fs = [[] for _ in range(num_edges)]  # edge -> f var names (incidence, built once)
    for od_idx in ordered:
        demand = float(tm_vector[od_idx])
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if demand <= 0 or not paths:
            continue
        prev_f = None
        if budget_active:
            prev_vec = np.asarray(prev_splits[od_idx], dtype=float) if od_idx < len(prev_splits) else np.zeros(len(paths))
            prev_pad = np.zeros(len(paths)); prev_pad[: min(prev_vec.size, len(paths))] = prev_vec[: min(prev_vec.size, len(paths))]
            prev_f = prev_pad * demand
        for pi, edge_path in enumerate(paths):
            fname = f"f_{od_idx}_{pi}"
            for e in edge_path:
                edge_fs[e].append(fname)
            if budget_active:
                yname = f"y_{od_idx}_{pi}"
                obj_coef[yname] = float(db_weight)
                pf = float(prev_f[pi])
                # y - f >= -pf ; y + f >= pf
                rp = f"abs_pos_{od_idx}_{pi}"; add_row(rp, "G", -pf); add_coef(yname, rp, 1.0); add_coef(fname, rp, -1.0)
                rn = f"abs_neg_{od_idx}_{pi}"; add_row(rn, "G", pf); add_coef(yname, rn, 1.0); add_coef(fname, rn, 1.0)
        dname = f"demand_{od_idx}"; add_row(dname, "E", demand)
        for pi in range(len(paths)):
            add_coef(f"f_{od_idx}_{pi}", dname, 1.0)
    for e in range(num_edges):
        cname = f"cap_{e}"; add_row(cname, "L", -float(background[e]))
        # background + sum(f on e) <= U*cap  ->  sum(f) - cap*U <= -background
        for fname in edge_fs[e]:
            add_coef(fname, cname, 1.0)
        add_coef("U", cname, -float(capacities[e]))
    if budget_active:
        y_all = [v for v in col_coef if v.startswith("y_")]
        if y_all:
            bname = "db_budget"; add_row(bname, "L", rhs)
            for od_idx in ordered:
                paths = path_library.edge_idx_paths_by_od[od_idx]
                for pi in range(len(paths)):
                    yn = f"y_{od_idx}_{pi}"
                    if yn in col_coef:
                        add_coef(yn, bname, 1.0)

    # column order = lexicographic by variable name (PuLP model.variables() order)
    all_vars = sorted(set(list(col_coef.keys()) + list(obj_coef.keys())))
    # row index order for writing each column's entries (obj first, then row_order)
    row_index = {"obj": 0}
    for i, (rn, _s) in enumerate(row_order):
        row_index[rn] = i + 1

    mps = ["*SENSE:Minimize", "NAME          T5MODEL", "ROWS", " N  obj"]
    for rn, sense in row_order:
        mps.append(f" {sense}  {rn}")
    mps.append("COLUMNS")
    for v in all_vars:
        entries = []
        if v in obj_coef and obj_coef[v] != 0.0:
            entries.append(("obj", obj_coef[v]))
        for rn, c in col_coef.get(v, {}).items():
            entries.append((rn, c))
        entries.sort(key=lambda kv: row_index.get(kv[0], 1 << 30))
        for rn, c in entries:
            mps.append(f"    {v}  {rn}  {_fmt(c)}")
    mps.append("RHS")
    for rn, _s in row_order:
        mps.append(f"    RHS  {rn}  {_fmt(rhs_map[rn])}")
    mps.append("BOUNDS")
    mps.append("ENDATA")
    mps_text = "\n".join(mps) + "\n"

    seed = os.environ.get("CBC_SEED", "42")
    try:
        int(seed)
    except ValueError:
        seed = "42"
    tmpd = tempfile.mkdtemp(prefix="t5lp_")
    lp_file = os.path.join(tmpd, "m.mps"); sol_file = os.path.join(tmpd, "m.sol")
    with open(lp_file, "w") as f:
        f.write(mps_text)
    cmd = [_cbc_path(), lp_file, "-sec", str(int(time_limit_sec)), "-threads", "1",
           "-randomCbcSeed", str(seed), "solve", "solu", sol_file]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        values = {}
        status = "Optimal"
        with open(sol_file) as f:
            first = f.readline()
            low = first.lower()
            if "infeasible" in low:
                status = "Infeasible"
            elif "unbounded" in low:
                status = "Unbounded"
            for ln in f:
                parts = ln.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    values[parts[1]] = float(parts[2])
                elif len(parts) >= 2:
                    try:
                        values[parts[0]] = float(parts[1])
                    except ValueError:
                        pass
    except Exception as ex:
        splits = clone_splits(base_splits)
        return HybridLPResult(splits=splits, routing=apply_routing(tm_vector, splits, path_library, capacities), status=f"OptSolveError:{ex}")
    finally:
        for p in (lp_file, sol_file):
            try: os.remove(p)
            except OSError: pass
        try: os.rmdir(tmpd)
        except OSError: pass

    if status not in {"Optimal"}:
        splits = clone_splits(base_splits)
        return HybridLPResult(splits=splits, routing=apply_routing(tm_vector, splits, path_library, capacities), status=f"DBBudgetFailed:{status}")

    # extract splits (identical normalization to _extract_splits_from_lp)
    splits = clone_splits(base_splits)
    for od_idx in ordered:
        demand = float(tm_vector[od_idx])
        paths = path_library.edge_idx_paths_by_od[od_idx]
        if demand <= 0 or not paths:
            continue
        vec = np.zeros(len(paths), dtype=float)
        for pi in range(len(paths)):
            vec[pi] = max(values.get(f"f_{od_idx}_{pi}", 0.0), 0.0)
        if demand > EPS:
            vec /= demand
        s = float(np.sum(vec))
        if s > EPS:
            vec /= s
            splits[od_idx] = vec
    routing = apply_routing(tm_vector, splits, path_library, capacities)
    return HybridLPResult(splits=splits, routing=routing, status="Optimal")
