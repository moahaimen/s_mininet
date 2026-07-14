#!/usr/bin/env python3
"""STRICT FAILURE FULL-MCF reference cache (Section-11 failure-PR repair).

Computes the strict UNRESTRICTED full multi-commodity-flow optimum MLU for the EXACT
scenario state (failed links / degraded capacity / spiked demand) of every failure event,
using te.lp_solver.solve_full_mcf_min_mlu on the FULL edge graph (NO candidate-path library,
NO selected-OD restriction, NO K restriction).

Exact-state identity is guaranteed by reusing the failure runner's own state functions
(make_base_ctx, modify_caps, spike_factor_for, failed_edge_ids, SCENARIOS, TOPOS) from
scripts.phase1_5.run_fix1_failure_baseline_comparison. Physically disconnected ODs (no path
on the surviving graph) are excluded from the strict MCF (demand set to 0) and counted, so the
strict reference matches the method achieved-MLU state instead of becoming infeasible.

Output: results/.../condition_compliant_k10_k50/STRICT_FAILURE_FULL_MCF_PR/
  _partial/<topo>.progress.csv  (resumable, one row per exact state)
Does NOT touch STRICT_FULL_MCF_PR/ (frozen normal cache). Does NOT use any safety-mode artifact.
"""
import sys, time, json, hashlib
from collections import deque
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
from te.lp_solver import solve_full_mcf_min_mlu
import scripts.phase1_5.run_fix1_failure_baseline_comparison as R
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import OUT_ROOT

OUT = OUT_ROOT / "condition_compliant_k10_k50"
SUB = OUT / "STRICT_FAILURE_FULL_MCF_PR"; PART = SUB / "_partial"
PART.mkdir(parents=True, exist_ok=True)
MCF_TIME_LIMIT = 300
ORDER = ["abilene", "geant", "ebone", "cernet", "germany50", "sprintlink", "tiscali", "vtlwavenet2011"]
EPS = 1e-9

def _hash(arr): return hashlib.md5(np.round(np.asarray(arr, float), 9).tobytes()).hexdigest()[:12]

def disconnected_ods(nodes, edges, caps_state, od_pairs, tm_state):
    """ODs with positive demand whose src cannot reach dst on the surviving (cap>0) directed graph."""
    adj = {n: [] for n in nodes}
    for (s, d), c in zip(edges, caps_state):
        if float(c) > EPS: adj[s].append(d)
    def reachable(src, dst):
        if src == dst: return True
        seen = {src}; q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v == dst: return True
                if v not in seen: seen.add(v); q.append(v)
        return False
    disc = []
    for i, (s, d) in enumerate(od_pairs):
        if float(tm_state[i]) > 0 and not reachable(s, d): disc.append(i)
    return disc

def solve_state(base, scenario, t):
    ds = base.ds; nodes = list(ds.nodes); edges = list(ds.edges); odp = list(ds.od_pairs)
    caps_state = R.modify_caps(base.caps0, scenario)
    tm_state = np.asarray(ds.tm[t], float) * R.spike_factor_for(scenario)
    failed = R.failed_edge_ids(base.caps0, scenario)
    disc = disconnected_ods(nodes, edges, caps_state, odp, tm_state)
    tm_solve = tm_state.copy()
    for i in disc: tm_solve[i] = 0.0            # exclude physically disconnected commodities
    r = solve_full_mcf_min_mlu(tm_solve, odp, nodes, edges, caps_state, time_limit_sec=MCF_TIME_LIMIT)
    mlu = float(r.mlu); status = str(r.status)
    return dict(
        topology=base.topo, tm_id=int(t), scenario=scenario,
        event_id=f"{base.topo}|{scenario}|{int(t)}",
        failed_links=",".join(str(x) for x in failed),
        capacity_hash=_hash(caps_state), demand_hash=_hash(tm_state),
        strict_full_mcf_MLU=mlu, solver_status=status,
        capacity_feasible=bool(status == "Optimal" and mlu <= 1.0 + 1e-6),
        max_capacity_violation=float(max(0.0, mlu - 1.0)),
        physically_disconnected_od_count=int(len(disc)),
        provenance="solve_full_mcf_min_mlu(full-edge,no-K,no-pathlib); state=modify_caps+spike_factor")

def run_topo(topo):
    lo, hi = R.TOPOS[topo]; prog = PART / f"{topo}.progress.csv"
    done = {}
    if prog.exists():
        pdf = pd.read_csv(prog)
        done = {(r.scenario, int(r.tm_id)) for r in pdf.itertuples()}
    todo = [(sc, t) for sc in R.SCENARIOS for t in range(lo, hi) if (sc, t) not in done]
    print(f"[run] {topo} states={6*(hi-lo)} remaining={len(todo)}", flush=True)
    if not todo:
        print(f"[skip] {topo}: complete", flush=True); return
    base = R.make_base_ctx(topo, lo, hi)
    write_header = not prog.exists(); t0 = time.perf_counter()
    for i, (sc, t) in enumerate(todo):
        row = solve_state(base, sc, t)
        df1 = pd.DataFrame([row])
        df1.to_csv(prog, mode="a", header=write_header, index=False); write_header = False
        if (i + 1) % 5 == 0 or (i + 1) == len(todo):
            print(f"    {topo}: {i+1}/{len(todo)} ({time.perf_counter()-t0:.0f}s) last={sc}/{t} mlu={row['strict_full_mcf_MLU']:.4f} {row['solver_status']} disc={row['physically_disconnected_od_count']}", flush=True)

if __name__ == "__main__":
    topos = sys.argv[1].split(",") if len(sys.argv) > 1 else ORDER
    for topo in topos:
        run_topo(topo)
    # consolidate
    frames = [pd.read_csv(PART / f"{t}.progress.csv") for t in ORDER if (PART / f"{t}.progress.csv").exists()]
    if frames:
        alldf = pd.concat(frames, ignore_index=True)
        alldf.to_csv(SUB / "strict_failure_full_mcf_reference.csv", index=False)
        print(f"[consolidated] {len(alldf)} states -> strict_failure_full_mcf_reference.csv")
    print("DONE")
