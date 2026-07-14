#!/usr/bin/env python3
"""Same-commodity-set audit for failure PR (denominator/commodity-set defect).

For every (topology, scenario, tm, method) it records whether the strict numerator commodity set
equals the method's achieved-evaluated commodity set, on the exact scenario state.

Evidence (code-traced):
- apply_routing (te/simulator.py) SKIPS ODs whose pruned-library path list is empty -> their demand
  contributes ZERO load -> excluded from achieved_mlu. (no od_paths -> continue)
- run_selected_lp: "no full-OD escalation ever"; commit 97b5247 removed the hidden full-OD fallback.
  => pathlib methods (ECMP, Top-K, Bottleneck Top-K, GNN-only, GNN+LPD, Final RG-GNN-LPD) DROP
     candidate-path-exhausted-but-physically-connected ODs (reduced commodity set).
- OSPF uses build_failure_ospf_path_library (surviving-graph shortest paths via nx.has_path) -> routes
  EVERY physically-connected OD (full physically-reachable commodity set).

strict numerator commodity set = physically-reachable active ODs (all connected; disconnected excluded).
achieved commodity set:
  pathlib methods -> active ODs with a surviving pruned-library path (non-exhausted)
  OSPF           -> active ODs with a surviving OSPF shortest path (= physically reachable)
same_commodity_set PASS iff achieved evaluated count == strict commodity count.

Writes STRICT_FAILURE_FULL_MCF_PR/same_commodity_set_audit.csv. Read-only; no method rerun; no LP solves.
"""
import sys
from collections import deque
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import scripts.phase1_5.run_fix1_failure_baseline_comparison as R
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import OUT_ROOT

OUT = OUT_ROOT / "condition_compliant_k10_k50"
SF = OUT / "STRICT_FAILURE_FULL_MCF_PR"
EPS = 1e-9
PATHLIB_METHODS = ["ECMP", "Top-K Demand (K50)", "Bottleneck Top-K (K50)",
                   "GNN-only fixed K50", "GNN+LPD fixed K50", "Final RG-GNN-LPD"]
OSPF_METHOD = "OSPF-weighted shortest-path routing"

def reach_set(nodes, edges, caps_state, od_pairs, active):
    adj = {n: [] for n in nodes}
    for (s, d), c in zip(edges, caps_state):
        if float(c) > EPS: adj[s].append(d)
    def conn(src, dst):
        if src == dst: return True
        seen = {src}; q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v == dst: return True
                if v not in seen: seen.add(v); q.append(v)
        return False
    return {i for i in active if conn(od_pairs[i][0], od_pairs[i][1])}

def audit_topo(topo, rows_out):
    lo, hi = R.TOPOS[topo]
    base = R.make_base_ctx(topo, lo, hi); ds = base.ds
    nodes = list(ds.nodes); edges = list(ds.edges); odp = list(ds.od_pairs)
    for scenario in R.SCENARIOS:
        caps_state = R.modify_caps(base.caps0, scenario)
        pl_pruned = R.prune_pathlib(base.pl0, caps_state)
        ospf_pl = R.build_failure_ospf_path_library(ds, caps_state, weight_mode=R.REF_WEIGHT_MODE)
        for t in range(lo, hi):
            tm = np.asarray(ds.tm[t], float) * R.spike_factor_for(scenario)
            active = [i for i in range(len(odp)) if tm[i] > 0]
            reach = reach_set(nodes, edges, caps_state, odp, active)
            n_reach = len(reach)
            n_pathlib_eval = sum(1 for i in reach if len(pl_pruned.edge_idx_paths_by_od[i]) > 0)
            n_ospf_eval = sum(1 for i in reach if len(ospf_pl.edge_idx_paths_by_od[i]) > 0)
            n_exh_conn = n_reach - n_pathlib_eval        # exhausted-but-connected (dropped by pathlib methods)
            for m in PATHLIB_METHODS + [OSPF_METHOD]:
                ev = n_ospf_eval if m == OSPF_METHOD else n_pathlib_eval
                rows_out.append(dict(
                    topology=topo, tm_index=int(t), scenario=scenario, event_id=f"{topo}|{scenario}|{int(t)}",
                    method=m, physically_reachable_OD_count=int(n_reach),
                    strict_commodity_count=int(n_reach), achieved_evaluated_commodity_count=int(ev),
                    candidate_exhausted_but_connected_count=int(n_exh_conn),
                    fallback_treatment=("surviving-shortest-path (routes all connected ODs)" if m == OSPF_METHOD
                                        else "none (apply_routing drops empty-path ODs; no full-OD escalation)"),
                    same_commodity_set=("PASS" if ev == n_reach else "FAIL")))
    print(f"[{topo}] done", flush=True)

if __name__ == "__main__":
    topos = sys.argv[1].split(",") if len(sys.argv) > 1 else \
        ["abilene", "geant", "ebone", "cernet", "germany50", "sprintlink", "tiscali", "vtlwavenet2011"]
    rows = []
    for topo in topos:
        audit_topo(topo, rows)
    df = pd.DataFrame(rows)
    df.to_csv(SF / "same_commodity_set_audit.csv", index=False)
    fail = df[df.same_commodity_set == "FAIL"]
    print(f"\nTOTAL rows: {len(df)}  FAIL: {len(fail)}  (audited topos: {topos})")
    if len(fail):
        print("FAIL by method:"); print(fail.groupby("method").size().to_string())
        print("FAIL by topology/scenario:"); print(fail.groupby(["topology", "scenario"]).size().to_string())
        print("PASS-only states (no exhausted-but-connected OD):",
              int((df[df.method=='ECMP'].candidate_exhausted_but_connected_count == 0).sum()),
              "of", len(df[df.method=='ECMP']))
    print("DONE_AUDIT")
