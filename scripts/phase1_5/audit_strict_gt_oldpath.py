#!/usr/bin/env python3
"""Audit every failure state where strict_full_mcf_MLU > old scenario path-library optimum + tolerance.

Mathematically, unrestricted full-MCF cannot exceed a path-restricted optimum on the SAME commodity set
and exact state. So each strict>old state MUST be explained by a verified commodity-set difference:
ODs that are candidate-path-EXHAUSTED (no surviving path in the pruned deployed path library) but
PHYSICALLY CONNECTED (a path exists on the surviving edge graph). The old path-library LP drops such ODs
(no candidate flow variables); the strict unrestricted full-MCF includes them as commodities.

For every strict>old state, records the 13 mandated audit items and the exact exhausted-but-connected OD
identities. If ANY strict>old state has ZERO exhausted-but-connected ODs, it is UNEXPLAINED -> the script
prints a STOP verdict.

Reuses the runner's exact-state functions (make_base_ctx, modify_caps, spike_factor_for, prune_pathlib) so
the reconstructed state is identical to what produced scenario_opt_mlu and achieved_mlu.
Writes STRICT_FAILURE_FULL_MCF_PR/strict_gt_old_path_reference_audit.csv (+ _od_identities.csv).
Read-only w.r.t. the strict cache, Section 13, safety mode, and normal N=3976 evidence.
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
TOL = 1e-6
EPS = 1e-9

pc = pd.read_csv(CM / "failure_baseline_comparison_fix1_per_cycle.csv")
# old path-library optimum per exact state (identical across methods within an event)
oldopt = (pc.groupby(["Topology", "Scenario", "tm_index"])["scenario_opt_mlu"].first().reset_index())
OLD = {(r.Topology, r.Scenario, int(r.tm_index)): float(r.scenario_opt_mlu) for r in oldopt.itertuples()}
# achieved semantics: methods route over the deployed path library, so an exhausted OD carries no achieved
# traffic (except OSPF, which uses a separate surviving-shortest-path library) -> reported per state.

def connected(nodes, edges, caps_state, src, dst):
    if src == dst: return True
    adj = {n: [] for n in nodes}
    for (s, d), c in zip(edges, caps_state):
        if float(c) > EPS: adj[s].append(d)
    seen = {src}; q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v == dst: return True
            if v not in seen: seen.add(v); q.append(v)
    return False

def audit_topo(topo, strict):
    lo, hi = R.TOPOS[topo]
    rows = strict[strict.topology == topo]
    # candidate strict>old states first (cheap filter before building base ctx)
    cand = []
    for r in rows.itertuples():
        key = (topo, r.scenario, int(r.tm_id)); o = OLD.get(key)
        if o is not None and float(r.strict_full_mcf_MLU) > o + TOL:
            cand.append((r.scenario, int(r.tm_id), float(r.strict_full_mcf_MLU), o, int(r.physically_disconnected_od_count)))
    if not cand:
        print(f"[{topo}] strict>old states: 0"); return [], []
    base = R.make_base_ctx(topo, lo, hi); ds = base.ds
    nodes = list(ds.nodes); edges = list(ds.edges); odp = list(ds.od_pairs)
    out, odid = [], []
    for scenario, t, strict_mlu, old_mlu, n_disc in cand:
        caps_state = R.modify_caps(base.caps0, scenario)
        tm_state = np.asarray(ds.tm[t], float) * R.spike_factor_for(scenario)
        pl_pruned = R.prune_pathlib(base.pl0, caps_state)
        active = [i for i in range(len(odp)) if tm_state[i] > 0]
        exhausted = [i for i in active if len(pl_pruned.edge_idx_paths_by_od[i]) == 0]   # no surviving candidate path
        exh_conn = [i for i in exhausted if connected(nodes, edges, caps_state, odp[i][0], odp[i][1])]
        n_conn = sum(1 for i in active if connected(nodes, edges, caps_state, odp[i][0], odp[i][1]))
        explained = len(exh_conn) > 0
        out.append(dict(
            topology=topo, tm_index=t, scenario=scenario,
            event_id=f"{topo}|{scenario}|{t}",
            old_path_library_opt_MLU=round(old_mlu, 6), strict_full_mcf_MLU=round(strict_mlu, 6),
            delta=round(strict_mlu - old_mlu, 6),
            n_physically_connected_ODs=int(n_conn), n_physically_disconnected_ODs=int(n_disc),
            n_candidate_exhausted_but_connected_ODs=int(len(exh_conn)),
            old_path_excluded_exhausted="yes (no candidate flow vars for empty-path ODs)",
            strict_included_exhausted="yes (routed as commodities on full edge graph)",
            achieved_includes_exhausted="no for pathlib methods (ECMP/TopK/GNN/LPD/RG use deployed pl); OSPF may route via surviving-shortest-path lib",
            EXPLAINED=bool(explained)))
        odid.append(dict(event_id=f"{topo}|{scenario}|{t}",
                         exhausted_but_connected_od_indices="|".join(str(i) for i in exh_conn),
                         exhausted_but_connected_od_pairs="|".join(f"{odp[i][0]}->{odp[i][1]}" for i in exh_conn)))
    ne = sum(1 for r in out if not r["EXPLAINED"])
    print(f"[{topo}] strict>old states: {len(out)}  explained: {len(out)-ne}  UNEXPLAINED: {ne}")
    return out, odid

if __name__ == "__main__":
    topos = sys.argv[1].split(",") if len(sys.argv) > 1 else \
        [t for t in ["abilene","geant","ebone","cernet","germany50","sprintlink","tiscali","vtlwavenet2011"]
         if (SF / "_partial" / f"{t}.progress.csv").exists()]
    strict = pd.concat([pd.read_csv(SF / "_partial" / f"{t}.progress.csv") for t in topos], ignore_index=True)
    allout, allod = [], []
    for topo in topos:
        o, d = audit_topo(topo, strict); allout += o; allod += d
    adf = pd.DataFrame(allout); odf = pd.DataFrame(allod)
    if len(adf):
        adf.to_csv(SF / "strict_gt_old_path_reference_audit.csv", index=False)
        odf.to_csv(SF / "strict_gt_old_path_od_identities.csv", index=False)
        print("\n=== strict>old count by topology/scenario ===")
        print(adf.groupby(["topology", "scenario"]).size().to_string())
        unexpl = adf[~adf.EXPLAINED]
        print(f"\nTOTAL strict>old states: {len(adf)}  UNEXPLAINED: {len(unexpl)}")
        if len(unexpl):
            print("STOP: unexplained strict>old states exist:")
            print(unexpl[["event_id","old_path_library_opt_MLU","strict_full_mcf_MLU","n_candidate_exhausted_but_connected_ODs"]].to_string(index=False))
        else:
            print("VERDICT: every strict>old state is explained by candidate-path-exhausted-but-connected OD semantics.")
    else:
        print("No strict>old states found in audited topologies.")
    print("DONE_AUDIT")
