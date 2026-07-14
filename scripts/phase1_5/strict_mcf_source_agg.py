#!/usr/bin/env python3
"""EXACT unrestricted full-MCF min-MLU via SOURCE-AGGREGATED formulation (HiGHS, direct sparse build).

For min-MLU the objective depends only on total link load, so all OD demands sharing a SOURCE node can be
carried by ONE multi-sink commodity without changing link loads or feasibility. This is mathematically exact
(identical optimum to the per-OD formulation) but uses <=N commodities instead of #OD-pairs -> ~100x smaller
for dense topologies (VtlWavenet: 92 source-commodities x 192 edges ~= 17.6k vars vs 1.6M).

LP (identical feasible link-load set to the per-OD full-MCF):
  min U
  s.t.  sum_s x[s,e] <= U * cap_e                    (every surviving edge)
        for each source s, each node n:  out(n) - in(n) = b[s,n]
          b[s,s]  = sum_d demand(s->d) over routable dests ;  b[s,d] = -demand(s->d) ;  else 0
        x >= 0, U >= 0
Exact-preserving: drop zero-cap failed edges; drop physically-disconnected (s,d) demands (no s->t path on the
surviving graph) and count them; per-source directed-reachability trimming; O(1) numerical rescaling.
No candidate paths / K / selected ODs / heuristics. Returns mlu, status, diagnostics."""
from collections import deque
import numpy as np, scipy.sparse as sp, time, highspy
EPS = 1e-9; INF = 1e30

def solve_source_mcf(tm_vector, od_pairs, nodes, edges, capacities, time_limit_sec=300, output=False,
                     method="choose", presolve=True):
    caps = np.asarray(capacities, float)
    nidx = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
    surv = [e for e in range(len(edges)) if caps[e] > EPS]; E = len(surv)
    zero_cap = len(edges) - E; caps_s = caps[surv]
    src_l = np.array([nidx[edges[e][0]] for e in surv], int); dst_l = np.array([nidx[edges[e][1]] for e in surv], int)
    # forward reachability per source node (surviving graph)
    fadj = {n: [] for n in nodes}
    for e in surv: fadj[edges[e][0]].append(edges[e][1])
    def fwd(s):
        seen = {s}; q = deque([s])
        while q:
            u = q.popleft()
            for v in fadj[u]:
                if v not in seen: seen.add(v); q.append(v)
        return seen
    FWD = {}
    # aggregate demand by source; drop physically disconnected (s,d)
    src_dem = {}; disc = 0; n_od_routable = 0
    for k in range(len(od_pairs)):
        d = float(tm_vector[k])
        if d <= 0: continue
        s, t = od_pairs[k]
        if s == t: continue
        if s not in FWD: FWD[s] = fwd(s)
        if t not in FWD[s]: disc += 1; continue
        src_dem.setdefault(s, {}).setdefault(t, 0.0)
        src_dem[s][t] += d; n_od_routable += 1
    sources = sorted(src_dem.keys(), key=lambda s: nidx[s])
    Ks = len(sources)
    if Ks == 0:
        return dict(status="NoDemand", mlu=0.0, nodes=N, surviving_edges=E, source_commodities=0,
                    disconnected_od=disc, total_vars=1, flow_vars=0, build_s=0.0, solve_s=0.0,
                    max_cap_violation=0.0, max_flow_conservation_residual=0.0, solver="HiGHS")
    # numerical rescaling
    tot_out = np.array([sum(src_dem[s].values()) for s in sources])
    cap_ref = float(np.median(caps_s)) if E and np.median(caps_s) > 0 else 1.0
    dem_ref = float(np.median(tot_out)) if np.median(tot_out) > 0 else 1.0
    caps_scaled = caps_s / cap_ref

    t0 = time.perf_counter()
    ncol = 1 + Ks * E; nrow = E + Ks * N
    kk = np.repeat(np.arange(Ks), E); le = np.tile(np.arange(E), Ks); xcol = 1 + kk * E + le
    cap_rows_x = le; cap_cols_x = xcol; cap_val_x = np.ones(Ks * E)
    cap_rows_U = np.arange(E); cap_cols_U = np.zeros(E, int); cap_val_U = -caps_scaled
    con_src_row = E + kk * N + src_l[le]; con_dst_row = E + kk * N + dst_l[le]
    rows = np.concatenate([cap_rows_x, cap_rows_U, con_src_row, con_dst_row])
    cols = np.concatenate([cap_cols_x, cap_cols_U, xcol, xcol])
    vals = np.concatenate([cap_val_x, cap_val_U, np.ones(Ks * E), -np.ones(Ks * E)])
    A = sp.csc_matrix((vals, (rows, cols)), shape=(nrow, ncol))
    # conservation RHS (scaled): b[si,node]
    bb = np.zeros(Ks * N)
    for si, s in enumerate(sources):
        base = si * N; tot = 0.0
        for d, dem in src_dem[s].items():
            bb[base + nidx[d]] -= dem / dem_ref; tot += dem / dem_ref
        bb[base + nidx[s]] += tot
    row_lower = np.empty(nrow); row_upper = np.empty(nrow)
    row_lower[:E] = -INF; row_upper[:E] = 0.0
    row_lower[E:] = bb; row_upper[E:] = bb
    col_cost = np.zeros(ncol); col_cost[0] = 1.0
    col_lower = np.zeros(ncol); col_upper = np.full(ncol, INF)
    build_s = time.perf_counter() - t0

    h = highspy.Highs(); h.setOptionValue("output_flag", bool(output))
    h.setOptionValue("time_limit", float(time_limit_sec)); h.setOptionValue("solver", method)
    h.setOptionValue("presolve", "on" if presolve else "off")
    lp = highspy.HighsLp(); lp.num_col_ = ncol; lp.num_row_ = nrow
    lp.col_cost_ = col_cost; lp.col_lower_ = col_lower; lp.col_upper_ = col_upper
    lp.row_lower_ = row_lower; lp.row_upper_ = row_upper
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
    lp.sense_ = highspy.ObjSense.kMinimize
    s0 = time.perf_counter(); h.passModel(lp); h.run(); solve_s = time.perf_counter() - s0
    status = h.modelStatusToString(h.getModelStatus())
    sol = h.getSolution(); cv = np.array(sol.col_value) if len(sol.col_value) else np.zeros(ncol)
    xval = (cv[1:].reshape(Ks, E) * dem_ref) if cv.size == ncol else np.zeros((Ks, E))
    load_s = xval.sum(axis=0); link_load = np.zeros(len(edges)); link_load[surv] = load_s
    util = link_load / np.maximum(caps, EPS); mlu = float(util.max()) if util.size else 0.0
    u_val = float(cv[0]) * dem_ref / cap_ref if cv.size else float("inf")
    cap_viol = float(np.max(load_s - u_val * caps_s)) if E else 0.0
    resid = float(np.max(np.abs((A[E:, :].dot(cv) - bb) * dem_ref))) if nrow > E else 0.0
    ok = status.lower().startswith("optimal")
    return dict(status=status, mlu=(mlu if ok else float("inf")), u_val=u_val, nodes=N, surviving_edges=E,
                source_commodities=Ks, disconnected_od=disc, routable_od=n_od_routable,
                flow_vars=Ks * E, total_vars=ncol, nonzeros=int(A.nnz), build_s=round(build_s, 3),
                solve_s=round(solve_s, 3), max_cap_violation=cap_viol, max_flow_conservation_residual=resid,
                solver=f"HiGHS-{method}-sourceagg")
