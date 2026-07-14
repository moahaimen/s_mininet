#!/usr/bin/env python3
"""Exact-equivalent unrestricted full-MCF min-MLU solver (same feasible set as te.lp_solver.solve_full_mcf_min_mlu),
optimized so large failure-state LPs are tractable. Preserves the EXACT mathematical problem:
  minimize U  s.t.  sum_k x[k,e] <= U * cap_e  (all edges);  flow conservation per commodity;  x >= 0.

Exact-preserving reductions ONLY (no candidate paths, no K, no selected ODs, no heuristics):
  (R1) drop zero-capacity (failed) edges  -> flow on them is forced to 0 anyway (cap=0 constraint);
  (R2) drop physically disconnected commodities (no s->t path on surviving graph) -> demand 0 (unroutable);
  (R3) per-commodity directed-reachability trimming: create x[k,(u,v)] only if u is reachable from src_k AND
       dst_k is reachable from v. Flow on any other edge cannot be part of an s->t flow, so it is 0 in every
       feasible solution -> removing the variable does not change the feasible set or the optimum.
Solver is pluggable (HiGHS or CBC); the LP is identical, so the optimum is solver-independent (validated).
Returns mlu, status, and audit diagnostics (var/constraint counts, residuals)."""
from collections import deque
import numpy as np, pulp, time

EPS = 1e-9

def _reach_sets(nodes, adj):
    """forward-reachable set from each node (BFS)."""
    out = {}
    for s in nodes:
        seen = {s}; q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in seen: seen.add(v); q.append(v)
        out[s] = seen
    return out

def solve_exact_mcf(tm_vector, od_pairs, nodes, edges, capacities, solver="highs",
                    time_limit_sec=300, msg=False, return_flows=False):
    caps = np.asarray(capacities, float)
    surv = [e for e in range(len(edges)) if caps[e] > EPS]           # R1: surviving edges only
    zero_cap = len(edges) - len(surv)
    fwd_adj = {n: [] for n in nodes}; bwd_adj = {n: [] for n in nodes}
    for e in surv:
        u, v = edges[e]; fwd_adj[u].append(v); bwd_adj[v].append(u)
    FWD = _reach_sets(nodes, fwd_adj)      # nodes reachable FROM n
    BWD = _reach_sets(nodes, bwd_adj)      # nodes that can REACH n (reverse graph)
    active = [k for k in range(len(od_pairs)) if float(tm_vector[k]) > 0]
    reachable, disconnected = [], []
    for k in active:
        s, t = od_pairs[k]
        (reachable if (s == t or t in FWD[s]) else disconnected).append(k)  # R2

    t0 = time.perf_counter()
    model = pulp.LpProblem("exact_full_mcf", pulp.LpMinimize)
    U = pulp.LpVariable("U", lowBound=0.0)
    x = {}; edge_vars = {e: [] for e in surv}
    n_flow = 0
    # per-commodity usable-edge trimming (R3)
    for k in reachable:
        s, t = od_pairs[k]
        if s == t: continue
        Fs = FWD[s]; Bt = BWD[t]
        for e in surv:
            u, v = edges[e]
            if u in Fs and v in Bt:                    # edge can lie on an s->t path
                var = pulp.LpVariable(f"x_{k}_{e}", lowBound=0.0)
                x[(k, e)] = var; edge_vars[e].append(var); n_flow += 1
    # capacity constraints (all surviving edges)
    for e in surv:
        model += pulp.lpSum(edge_vars[e]) <= U * float(caps[e]), f"cap_{e}"
    # flow conservation per reachable commodity at each node touched by its vars
    n_cons_flow = 0
    out_e = {n: [] for n in nodes}; in_e = {n: [] for n in nodes}
    for e in surv:
        u, v = edges[e]; out_e[u].append(e); in_e[v].append(e)
    for k in reachable:
        s, t = od_pairs[k]
        if s == t: continue
        d = float(tm_vector[k])
        # only nodes on some s->t corridor have vars; enforce conservation at all nodes (empty sums -> 0=rhs ok)
        corridor = (FWD[s] & BWD[t]) | {s, t}
        for n in corridor:
            outf = pulp.lpSum(x[(k, e)] for e in out_e[n] if (k, e) in x)
            inf = pulp.lpSum(x[(k, e)] for e in in_e[n] if (k, e) in x)
            rhs = d if n == s else (-d if n == t else 0.0)
            model += outf - inf == rhs, f"fc_{k}_{n}"; n_cons_flow += 1
    model += U
    build_s = time.perf_counter() - t0

    if solver.lower() == "highs":
        slv = pulp.HiGHS(msg=msg, timeLimit=int(time_limit_sec))
    else:
        slv = pulp.PULP_CBC_CMD(msg=msg, timeLimit=int(time_limit_sec), presolve=True)
    s0 = time.perf_counter(); status_code = model.solve(slv); solve_s = time.perf_counter() - s0
    status = pulp.LpStatus.get(status_code, "Unknown")

    link_load = np.zeros(len(edges))
    for (k, e), var in x.items():
        val = var.value() or 0.0
        if val > EPS: link_load[e] += val
    util = link_load / np.maximum(caps, EPS)
    mlu = float(np.max(util)) if util.size else 0.0
    u_val = float(U.value()) if U.value() is not None else float("inf")
    # capacity violation (should be ~0: load <= U*cap): report max(load_e - u_val*cap_e)
    cap_viol = float(np.max([link_load[e] - u_val * caps[e] for e in surv])) if surv else 0.0
    # flow-conservation residual
    resid = 0.0
    if return_flows or True:
        for k in reachable:
            s, t = od_pairs[k]
            if s == t: continue
            d = float(tm_vector[k])
            net = {}
            for e in surv:
                v = x.get((k, e))
                if v is None: continue
                val = v.value() or 0.0
                u_, w_ = edges[e]; net[u_] = net.get(u_, 0.0) + val; net[w_] = net.get(w_, 0.0) - val
            resid = max(resid, abs(net.get(s, 0.0) - d), abs(net.get(t, 0.0) + d))
    ok = (status == "Optimal")
    audit = dict(nodes=len(nodes), directed_edges=len(edges), surviving_edges=len(surv), zero_cap_edges=zero_cap,
                 reachable_commodities=len(reachable), disconnected_commodities=len(disconnected),
                 flow_vars=n_flow, total_vars=n_flow + 1, cap_constraints=len(surv), flow_conservation_constraints=n_cons_flow,
                 build_s=round(build_s, 2), solve_s=round(solve_s, 2), status=status,
                 mlu=(mlu if ok else float("inf")), u_val=u_val, max_cap_violation=cap_viol,
                 max_flow_conservation_residual=resid, solver=solver)
    return audit
