#!/usr/bin/env python3
"""Full-demand feasibility audit + feasibility-penalized strict failure PR (Section-11 repair, evaluation-semantics only).

Rule (user-specified third path):
- ONE exact-state strict full-MCF numerator per event, over the FULL physically-reachable positive-demand OD set
  (= the strict cache value; disconnected ODs already excluded). Same numerator for EVERY method.
- For each method/state, audit whether the method fully serves ALL physically-reachable positive-demand ODs.
  full_demand_routing_feasible = PASS iff unserved_physically_reachable_od_count == 0.
- PASS  -> strict_failure_PR = strict_full_mcf_MLU / achieved_mlu   (both on the full reachable set; no clamp).
- FAIL  -> strict_failure_PR = 0.0, pr_failure_reason = "physically_reachable_demand_unserved" (feasibility penalty).
Never: per-method numerator, invented fallback, min(1,.), strict(full)/reduced-achieved, or disconnected counted as unserved.

Served set (from deployed routing evidence, NO method rerun): an OD is served iff its demand can be placed on a
surviving path of the library the method routes over (apply_routing places full demand when a path exists):
  pathlib methods (ECMP, Top-K, Bottleneck Top-K, GNN-only, GNN+LPD, RG-GNN-LPD) -> pruned deployed library `pl`.
  OSPF -> surviving-graph shortest-path library. Audited under the SAME feasibility rule (no special reference).

Code-repair: physically_disconnected, candidate_path_exhausted_connected, and unserved_physically_reachable are
recorded SEPARATELY (no longer conflated in a single "Disconnected ODs" column).

Output: STRICT_FAILURE_FULL_MCF_PR/full_demand_method_feasibility_audit.csv
Read-only w.r.t. deployed method, Section 13, safety mode, normal N=3976.
"""
import sys
from collections import deque
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import scripts.phase1_5.run_fix1_failure_baseline_comparison as R
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import OUT_ROOT

OUT = OUT_ROOT / "condition_compliant_k10_k50"
CM = OUT / "FINAL_REPORT_FIX1" / "completed_metrics"
SF = OUT / "STRICT_FAILURE_FULL_MCF_PR"
EPS = 1e-9
PATHLIB = ["ECMP", "Top-K Demand (K50)", "Bottleneck Top-K (K50)", "GNN-only fixed K50", "GNN+LPD fixed K50", "Final RG-GNN-LPD"]
OSPF = "OSPF-weighted shortest-path routing"

pc = pd.read_csv(CM / "failure_baseline_comparison_fix1_per_cycle.csv")
ACH = {(r.Method, r.Topology, r.Scenario, int(r.tm_index)): float(r.achieved_mlu) for r in pc.itertuples()}

def load_strict():
    d = {}
    for t in ["abilene","geant","ebone","cernet","germany50","sprintlink","tiscali","vtlwavenet2011"]:
        p = SF / "_partial" / f"{t}.progress.csv"
        if p.exists():
            for r in pd.read_csv(p).itertuples():
                d[(t, r.scenario, int(r.tm_id))] = float(r.strict_full_mcf_MLU)
    return d
STRICT = load_strict()

def reachable_active(nodes, edges, caps_state, od_pairs, tm):
    adj = {n: [] for n in nodes}
    for (s, dd), c in zip(edges, caps_state):
        if float(c) > EPS: adj[s].append(dd)
    def conn(src, dst):
        if src == dst: return True
        seen = {src}; q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v == dst: return True
                if v not in seen: seen.add(v); q.append(v)
        return False
    active = [i for i in range(len(od_pairs)) if tm[i] > 0]
    reach = [i for i in active if conn(od_pairs[i][0], od_pairs[i][1])]
    return active, reach

def audit_topo(topo, out):
    lo, hi = R.TOPOS[topo]
    base = R.make_base_ctx(topo, lo, hi); ds = base.ds
    nodes = list(ds.nodes); edges = list(ds.edges); odp = list(ds.od_pairs)
    for scenario in R.SCENARIOS:
        caps_state = R.modify_caps(base.caps0, scenario)
        pl_pruned = R.prune_pathlib(base.pl0, caps_state)
        ospf_pl = R.build_failure_ospf_path_library(ds, caps_state, weight_mode=R.REF_WEIGHT_MODE)
        for t in range(lo, hi):
            tm = np.asarray(ds.tm[t], float) * R.spike_factor_for(scenario)
            active, reach = reachable_active(nodes, edges, caps_state, odp, tm)
            reach_set = set(reach)
            n_reach = len(reach); n_disc = len(active) - n_reach
            tot_dem = float(sum(tm[i] for i in reach))
            strict = STRICT.get((topo, scenario, int(t)), np.nan)
            for m in PATHLIB + [OSPF]:
                lib = ospf_pl if m == OSPF else pl_pruned
                served = [i for i in reach if len(lib.edge_idx_paths_by_od[i]) > 0]
                unserved = [i for i in reach if i not in set(served)]
                exh_conn = n_reach - len([i for i in reach if len(pl_pruned.edge_idx_paths_by_od[i]) > 0])
                served_dem = float(sum(tm[i] for i in served)); unserved_dem = tot_dem - served_dem
                feasible = (len(unserved) == 0)
                ach = ACH.get((m, topo, scenario, int(t)), np.nan)
                if feasible and np.isfinite(strict) and ach > 0:
                    pr = float(strict / ach); reason = ""
                elif not feasible:
                    pr = 0.0; reason = "physically_reachable_demand_unserved"
                else:
                    pr = np.nan; reason = "strict_pending" if not np.isfinite(strict) else "achieved_missing"
                out.append(dict(
                    topology=topo, tm_index=int(t), scenario=scenario, event_id=f"{topo}|{scenario}|{int(t)}", method=m,
                    physically_reachable_positive_demand_od_count=n_reach, strict_commodity_count=n_reach,
                    fully_served_od_count=len(served), partially_served_od_count=0,
                    unserved_physically_reachable_od_count=len(unserved),
                    candidate_path_exhausted_connected_od_count=int(exh_conn),
                    physically_disconnected_od_count=int(n_disc),
                    total_reachable_demand=round(tot_dem, 6), served_reachable_demand=round(served_dem, 6),
                    unserved_reachable_demand=round(unserved_dem, 6),
                    served_demand_fraction=round(served_dem / tot_dem, 6) if tot_dem > 0 else 1.0,
                    full_demand_routing_feasible="PASS" if feasible else "FAIL",
                    strict_full_mcf_MLU=strict, achieved_mlu_if_feasible=(ach if feasible else np.nan),
                    strict_failure_PR=pr, pr_failure_reason=reason))
    print(f"[{topo}] done", flush=True)

if __name__ == "__main__":
    topos = sys.argv[1].split(",") if len(sys.argv) > 1 else \
        ["abilene","geant","ebone","cernet","germany50","sprintlink","tiscali","vtlwavenet2011"]
    out = []
    for topo in topos:
        audit_topo(topo, out)
    df = pd.DataFrame(out)
    dest = SF / "full_demand_method_feasibility_audit.csv"
    if dest.exists() and set(topos) != {"abilene","geant","ebone","cernet","germany50","sprintlink","tiscali","vtlwavenet2011"}:
        prev = pd.read_csv(dest); prev = prev[~prev.topology.isin(topos)]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(dest, index=False)
    fail = df[df.full_demand_routing_feasible == "FAIL"]
    print(f"\nrows={len(df)}  FAIL={len(fail)}  audited={topos}")
    if len(fail):
        print("FAIL by topology:"); print(fail.groupby("topology").size().to_string())
        print("FAIL by scenario:"); print(fail.groupby("scenario").size().to_string())
        print("FAIL by method:"); print(fail.groupby("method").size().to_string())
        print("max unserved_demand_fraction:", round(float((1 - df.served_demand_fraction).max()), 6))
    print("DONE_FEAS_AUDIT")
