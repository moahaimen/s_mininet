#!/usr/bin/env python3
"""Clean p95 timing re-measurement for the coverage-aware deployment rule on the FROZEN Tier A model.
Runs the rule (raise K to ~5% coverage, cap K500, no topology name) on the affected topologies
(VtlWavenet, CERNET), 2 repetitions, and reports mean/p95/max decision time. Decision: adopt iff
clean p95 < 500 ms for VtlWavenet and CERNET. PR/DB are deterministic (already known); this measures timing only.
Non-destructive: reads frozen model + existing caches; writes coverage_clean_timing.csv only."""
import sys, time, json, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import torch
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, apply_routing, clone_splits, set_seed
from te.lp_solver import solve_selected_path_lp_dbbudget
import scripts.phase1_5.agnostic_lib as A
from scripts.phase1_5.bottleneck_lib import ACTIONS, ANAME
from scripts.phase1_5.run_final_iter2 import kp_for, build_mixed, pad_to_lib, GNN_MS, WIN

set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
AGN = OUT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN" / "_cache"; KP = OUT / "FINAL_LEARNED_4OF5_KPATH4_DDQN" / "_cache"
SC = json.load(open(AGN / "scaler.json")); MEAN = np.array(SC["mean"], np.float32); STD = np.array(SC["std"], np.float32)
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))
COV_TARGET, RUNTIME_CAP_K = 0.05, 500
dim = len(A.AGN_FEAT_NAMES)
ck = torch.load(OUT / "FROZEN_FINAL_LEARNED_RUNTIME_SAFE_ITER2" / "final_learned_4of5_iter2_model.pt", map_location="cpu")
net = A.QNet(dim, 7); net.load_state_dict(ck["state_dict"]); net.eval()

def run(topo):
    lo, hi = WIN[topo]; d = P[(topo, lo, hi)]; caps = np.asarray(d["caps"], float)
    env = _make_envs([topo], {topo:(lo,hi)}, gnn, hi-lo, 30)[0]; ctx = env.ctx
    ds, pl, ecmp = ctx["ds"], ctx["pl"], ctx["ecmp"]
    raws = pickle.load(open(AGN / f"raw_EVAL_{topo}.pkl","rb")); rankings = pickle.load(open(KP / f"rank_EVAL_{topo}.pkl","rb"))
    accepted = clone_splits(ecmp); ms_list = []
    for t in range(lo, hi):
        tm = np.asarray(ds.tm[t], float); keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
        raw, emlu = raws[t]; s = A.standardize(A.raw_to_vec(raw, keep_mlu, emlu), MEAN, STD)
        with torch.no_grad(): a = int(net(torch.tensor(s).unsqueeze(0)).argmax())
        kind, K, _ = ACTIONS[a]; active = int((tm > 0).sum())
        if kind == "opt" and active > 0 and (K/active) < COV_TARGET:   # coverage-aware rule
            K = min(max(K, int(np.ceil(COV_TARGET*active))), RUNTIME_CAP_K)
        if kind == "keep":
            ms = 0.5; sp = accepted
        else:
            kp = kp_for(K); sel = list(rankings[t][:K]); plm = build_mixed(pl, set(int(o) for o in sel), kp); s0 = time.perf_counter()
            lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp, path_library=plm,
                capacities=caps, prev_splits=accepted, db_budget=0.051, db_weight=1e-6, time_limit_sec=120)
            sp = pad_to_lib(lp.splits, pl); _ = float(apply_routing(tm, sp, pl, caps).mlu); ms = (time.perf_counter()-s0)*1000 + GNN_MS[topo]
        ms_list.append(ms); accepted = sp
    return np.array(ms_list)

print(f"Clean coverage-aware timing (frozen model + rule), load={__import__('os').popen('sysctl -n vm.loadavg').read().strip()}\n", flush=True)
rows = []
for rep in range(2):
    for topo in ["vtlwavenet2011", "cernet"]:
        ms = run(topo); p95 = float(np.percentile(ms, 95))
        rows.append(dict(rep=rep+1, topology=topo, mean_ms=round(ms.mean(),1), p95_ms=round(p95,1), max_ms=round(ms.max(),1), under_500=bool(p95 < 500)))
        print(f"  rep{rep+1} {topo:15s} mean={ms.mean():.1f} p95={p95:.1f} max={ms.max():.1f}  p95<500={p95<500}", flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / "COVERAGE_AWARE_K_ABLATION" / "coverage_clean_timing.csv", index=False)
vtl_ok = bool((df[df.topology=='vtlwavenet2011'].p95_ms < 500).all()); cer_ok = bool((df[df.topology=='cernet'].p95_ms < 500).all())
print(f"\nVERDICT: VtlWavenet p95<500 both reps={vtl_ok}; CERNET p95<500 both reps={cer_ok}  -> ADOPT={vtl_ok and cer_ok}")
print("DONE")
