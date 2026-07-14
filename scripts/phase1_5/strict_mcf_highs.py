#!/usr/bin/env python3
"""Exact unrestricted full-MCF min-MLU solved via HiGHS through a directly-built sparse matrix (no pulp).

Same mathematical LP as te.lp_solver.solve_full_mcf_min_mlu (identical feasible set, objective, commodities):
  minimize U
  s.t.  sum_k x[k,e] <= U * cap_e          for every surviving edge e   (capacity)
        sum_{e:src=n} x[k,e] - sum_{e:dst=n} x[k,e] = b(k,n)            (flow conservation)
        x >= 0, U >= 0
Exact-preserving reductions: drop zero-capacity failed edges (flow on them is 0 by the cap=0 constraint) and
drop physically disconnected commodities (no s->t path -> unroutable, demand 0). No candidate paths / K / selected
ODs / heuristics. The matrix is emitted directly to HiGHS (CSC) so a 1.6M-variable LP builds in seconds.

Returns mlu, status, and full audit diagnostics.
"""
from collections import deque
import numpy as np, scipy.sparse as sp, time, highspy
EPS = 1e-9; INF = 1e30

def solve_exact_mcf_highs(tm_vector, od_pairs, nodes, edges, capacities, time_limit_sec=600, output=False,
                          method="choose", presolve=True, threads=0):
    caps = np.asarray(capacities, float)
    nidx = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
    surv = [e for e in range(len(edges)) if caps[e] > EPS]
    E = len(surv); zero_cap = len(edges) - E
    caps_s = caps[surv]
    src_l = np.array([nidx[edges[e][0]] for e in surv]); dst_l = np.array([nidx[edges[e][1]] for e in surv])
    # reachability (drop disconnected commodities) -- forward BFS from each node
    fadj = {n: [] for n in nodes}
    for e in surv: fadj[edges[e][0]].append(edges[e][1])
    def fwd(s):
        seen = {s}; q = deque([s])
        while q:
            u = q.popleft()
            for v in fadj[u]:
                if v not in seen: seen.add(v); q.append(v)
        return seen
    reach_cache = {}
    active = [k for k in range(len(od_pairs)) if float(tm_vector[k]) > 0]
    reachable, disconnected = [], []
    for k in active:
        s, t = od_pairs[k]
        if s == t: reachable.append(k); continue
        if s not in reach_cache: reach_cache[s] = fwd(s)
        (reachable if t in reach_cache[s] else disconnected).append(k)

    t0 = time.perf_counter()
    Keff = [k for k in reachable if od_pairs[k][0] != od_pairs[k][1]]
    K = len(Keff)
    src_k = np.array([nidx[od_pairs[k][0]] for k in Keff]); dst_k = np.array([nidx[od_pairs[k][1]] for k in Keff])
    dem_orig = np.array([float(tm_vector[k]) for k in Keff])
    # exact-preserving numerical rescaling: normalize caps and demand to O(1) (pure change of units).
    cap_ref = float(np.median(caps_s)) if E and np.median(caps_s) > 0 else 1.0
    dem_ref = float(np.median(dem_orig)) if K and np.median(dem_orig) > 0 else 1.0
    caps_scaled = caps_s / cap_ref
    dem = dem_orig / dem_ref
    ncol = 1 + K * E
    nrow = E + K * N
    # ---- column-index grids for x[kk,le] ----
    kk = np.repeat(np.arange(K), E); le = np.tile(np.arange(E), K)
    xcol = 1 + kk * E + le
    # capacity nonzeros: x (+1) and U (-cap)
    cap_rows_x = le; cap_cols_x = xcol; cap_val_x = np.ones(K * E)
    cap_rows_U = np.arange(E); cap_cols_U = np.zeros(E, int); cap_val_U = -caps_scaled
    # conservation nonzeros: +1 at src row, -1 at dst row
    con_src_row = E + kk * N + src_l[le]; con_dst_row = E + kk * N + dst_l[le]
    rows = np.concatenate([cap_rows_x, cap_rows_U, con_src_row, con_dst_row])
    cols = np.concatenate([cap_cols_x, cap_cols_U, xcol, xcol])
    vals = np.concatenate([cap_val_x, cap_val_U, np.ones(K * E), -np.ones(K * E)])
    A = sp.csc_matrix((vals, (rows, cols)), shape=(nrow, ncol))
    # row bounds
    row_lower = np.empty(nrow); row_upper = np.empty(nrow)
    row_lower[:E] = -INF; row_upper[:E] = 0.0                          # capacity <= 0
    bb = np.zeros(K * N)
    np.add.at(bb, np.arange(K) * N + src_k, dem)
    np.add.at(bb, np.arange(K) * N + dst_k, -dem)
    row_lower[E:] = bb; row_upper[E:] = bb                              # conservation ==
    col_cost = np.zeros(ncol); col_cost[0] = 1.0
    col_lower = np.zeros(ncol); col_upper = np.full(ncol, INF)
    build_s = time.perf_counter() - t0

    h = highspy.Highs()
    h.setOptionValue("output_flag", bool(output))
    h.setOptionValue("time_limit", float(time_limit_sec))
    h.setOptionValue("solver", method)                 # "choose" (dual simplex for LP) | "simplex" | "ipm"
    h.setOptionValue("presolve", "on" if presolve else "off")
    if threads: h.setOptionValue("threads", int(threads))
    if method == "ipm": h.setOptionValue("run_crossover", "off")
    lp = highspy.HighsLp()
    lp.num_col_ = ncol; lp.num_row_ = nrow
    lp.col_cost_ = col_cost; lp.col_lower_ = col_lower; lp.col_upper_ = col_upper
    lp.row_lower_ = row_lower; lp.row_upper_ = row_upper
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
    lp.sense_ = highspy.ObjSense.kMinimize
    s0 = time.perf_counter(); h.passModel(lp); h.run(); solve_s = time.perf_counter() - s0
    ms = h.getModelStatus(); status = h.modelStatusToString(ms)
    sol = h.getSolution(); info = h.getInfo()
    cv = np.array(sol.col_value) if len(sol.col_value) else np.zeros(ncol)
    # recover ORIGINAL units: scaled flows -> original flows (x_orig = x_scaled * dem_ref); U_orig = U_scaled*dem_ref/cap_ref
    u_val = float(cv[0]) * dem_ref / cap_ref if cv.size else float("inf")
    xval = (cv[1:].reshape(K, E) * dem_ref) if cv.size == ncol else np.zeros((K, E))
    load_s = xval.sum(axis=0)                       # original-unit link loads on surviving edges
    link_load = np.zeros(len(edges)); link_load[surv] = load_s
    util = link_load / np.maximum(caps, EPS); mlu = float(util.max()) if util.size else 0.0
    cap_viol = float(np.max(load_s - u_val * caps_s)) if E else 0.0
    # flow-conservation residual in ORIGINAL units: recover scaled residual then * dem_ref
    resid_vec = (A[E:, :].dot(cv) - bb) * dem_ref
    resid = float(np.max(np.abs(resid_vec))) if resid_vec.size else 0.0
    ok = status.lower().startswith("optimal")
    return dict(nodes=N, directed_edges=len(edges), surviving_edges=E, zero_cap_edges=zero_cap,
                reachable_commodities=K, disconnected_commodities=len(disconnected),
                flow_vars=K * E, total_vars=ncol, cap_constraints=E, flow_conservation_constraints=K * N,
                nonzeros=int(A.nnz), build_s=round(build_s, 2), solve_s=round(solve_s, 2),
                simplex_iters=int(getattr(info, "simplex_iteration_count", -1)),
                ipm_iters=int(getattr(info, "ipm_iteration_count", -1)),
                status=status, mlu=(mlu if ok else float("inf")), u_val=u_val,
                max_cap_violation=cap_viol, max_flow_conservation_residual=resid, solver="HiGHS-ipm")
