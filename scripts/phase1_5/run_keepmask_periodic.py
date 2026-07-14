#!/usr/bin/env python3
"""FAILURE-MODE KEEP ACTION MASK on the full-stream periodic single-link failure eval (Final RG-GNN-LPD).
On FAILURE cycles (active single-link-down) KEEP is removed from the valid action set; the DDQN still selects
argmax Q over {K50,K100,K200,K300,K500,K800}. On normal cycles (no link down) the full action set is used
(mask inactive -> normal selection unchanged). PR numerator = strict full-MCF (source-aggregated). No report edit.
Replicates the periodic runner's exact state construction (reroute-off failed link, carry-forward)."""
import sys, time, numpy as np, pandas as pd, torch
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, apply_routing, clone_splits, active_od_indices, set_seed
from te.lp_solver import solve_selected_path_lp_dbbudget
from te.disturbance import compute_disturbance
import scripts.phase1_5.agnostic_lib as A
from scripts.phase1_5.bottleneck_lib import ACTIONS, ANAME
from scripts.phase1_5.run_final_iter2 import kp_for, build_mixed, pad_to_lib, bottleneck_rank, GNN_MS
from scripts.phase1_5.strict_mcf_source_agg import solve_source_mcf
import json, pickle
set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"; SUB = OUT / "KEEPMASK_FAILURE"; SUB.mkdir(exist_ok=True)
AGN = OUT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN" / "_cache"
SC = json.load(open(AGN / "scaler.json")); MEAN = np.array(SC["mean"], np.float32); STD = np.array(SC["std"], np.float32)
dim = len(A.AGN_FEAT_NAMES)
net = A.QNet(dim, 7); net.load_state_dict(torch.load(OUT / "FULLDATA_GATED_PRESERVED_FIX1" / "fulldata_gated_model.pt", map_location="cpu")["state_dict"]); net.eval()
teacher = A.QNet(dim, 7); teacher.load_state_dict(torch.load(OUT / "FROZEN_FINAL_LEARNED_RUNTIME_SAFE_ITER2" / "final_learned_4of5_iter2_model.pt", map_location="cpu")["state_dict"]); teacher.eval()
KEEP_IDX = [a for a, v in ACTIONS.items() if v[0] == "keep"][0]
WIN = {"abilene":(2016,4032),"geant":(672,1344),"cernet":(200,400),"sprintlink":(200,400),"tiscali":(200,400),"ebone":(200,400),"germany50":(0,288),"vtlwavenet2011":(0,200)}
NFREQ = {"abilene":12,"geant":4,"cernet":2,"sprintlink":2,"tiscali":2,"ebone":2,"germany50":4,"vtlwavenet2011":2}
struct_cache = {}
def st(raw, keep_mlu, emlu): return A.standardize(A.raw_to_vec(raw, keep_mlu, emlu), MEAN, STD)

def reroute_off(splits, fl):
    out = clone_splits(splits)  # zero out shares on paths using the failed link handled by apply_routing (cap~0)
    return out

def run(topo, mask):
    lo, hi = WIN[topo]; N = NFREQ[topo]
    env = _make_envs([topo], {topo:(lo,hi)}, gnn, hi-lo, 30)[0]; ctx = env.ctx
    ds, pl0, ecmp0 = ctx["ds"], ctx["pl"], ctx["ecmp"]; caps0 = np.asarray(ctx["caps"], float)
    struct0 = A.struct_feats(ds); cap_order = list(np.argsort(caps0)[::-1]); top = [int(i) for i in cap_order[:min(20, int((caps0>0).sum()))]]
    accepted = clone_splits(ecmp0); prev_tm = None; rows = []
    for i, t in enumerate(range(lo, hi)):
        is_fail = (i > 0 and i % N == 0)
        tm = np.asarray(ds.tm[t], float)
        if is_fail:
            fl = top[(i // N) % len(top)]; caps = caps0.copy(); caps[fl] = max(caps0[fl]*1e-9, 1e-9)
            ecmp = clone_splits(ecmp0); acc = clone_splits(accepted)
        else:
            caps = caps0; ecmp = ecmp0; acc = accepted
        scr, _, _ = gnn.score(dataset=ds, tm_vector=tm, path_library=pl0, capacities=caps, ecmp_base=ecmp); scr = np.asarray(scr, float).ravel()
        util = apply_routing(tm, ecmp, pl0, caps).utilization; ranked = bottleneck_rank(tm, ecmp, pl0, caps, scr)
        keep_mlu = float(apply_routing(tm, acc, pl0, caps).mlu); emlu = float(apply_routing(tm, ecmp, pl0, caps).mlu)
        act_list = active_od_indices(tm); av = scr[act_list] if len(act_list) else np.zeros(1)
        dd = dict(ranked={t: np.array(ranked, np.int32)}, tm_cache=ds.tm, num_nodes=len(ds.nodes))
        chg = 0.0 if prev_tm is None else float(np.abs(tm-prev_tm).sum()/(np.abs(prev_tm).sum()+1e-9))
        dpre = dict(tmstat={t:(float(np.log1p(tm.sum())), float(tm.max()/(tm.sum()+1e-9)), min(chg,3.0), len(act_list))},
                    sstat={t:(float(av.mean()), float(np.quantile(av,.95)), float(av.max()))}, emlu={t: emlu})
        raw = A.raw_static(topo, t, dd, dpre, pl0, ecmp, caps, scr, util, struct0)
        s = st(raw, keep_mlu, dpre["emlu"][t]); sv = torch.tensor(s).unsqueeze(0)
        with torch.no_grad(): q = (teacher(sv) if i == 0 else net(sv)).numpy().ravel()
        if mask and is_fail: q[KEEP_IDX] = -np.inf     # KEEP masked on active link-down cycle
        a = int(q.argmax()); kind, K, _ = ACTIONS[a]
        if kind == "keep": mlu = keep_mlu; sp = acc; ms = GNN_MS[topo] + 0.5
        else:
            kp = kp_for(K); sel = list(ranked[:K]); plm = build_mixed(pl0, set(int(o) for o in sel), kp); s0 = time.perf_counter()
            dbb = 1.0 if is_fail else 0.051
            lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp, path_library=plm, capacities=caps, prev_splits=acc, db_budget=dbb, db_weight=1e-6, time_limit_sec=60)
            sp = pad_to_lib(lp.splits, pl0); mlu = float(apply_routing(tm, sp, pl0, caps).mlu); ms = (time.perf_counter()-s0)*1000 + GNN_MS[topo]
        strict = solve_source_mcf(tm, ds.od_pairs, ds.nodes, ds.edges, caps, time_limit_sec=60)["mlu"]
        pr = min(1.0, strict/mlu) if (np.isfinite(strict) and mlu > 0) else 0.0
        rows.append(dict(topology=topo, tm_index=int(t), is_failure=int(is_fail), action=ANAME[a], PR=pr, MLU=mlu, mask=mask))
        accepted = sp; prev_tm = tm
    return pd.DataFrame(rows)

if __name__ == "__main__":
    topos = sys.argv[1].split(",") if len(sys.argv) > 1 else list(WIN)
    allrows = []
    for topo in topos:
        for mask in [False, True]:
            d = run(topo, mask); allrows.append(d)
            ff = d[d.is_failure == 1]
            print(f"  {topo} mask={mask}: fail-min={ff.PR.min():.4f} fail-mean={ff.PR.mean():.4f} fail>=90={(ff.PR>=.9).mean()*100:.0f}% KEEP_on_fail={int((ff.action=='KEEP').sum())}", flush=True)
    pd.concat(allrows).to_csv(SUB / "keepmask_periodic_per_cycle.csv", index=False)
    print("DONE")
