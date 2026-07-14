#!/usr/bin/env python3
"""Test: force a LARGER first-cycle optimization on VtlWavenet (cycle 0 -> K in {500,800,1200} + db=1.0),
cycles>=1 stay frozen (K50, db=0.051). Does it improve VtlWavenet's Min/mean PR, and at what runtime cost?
Compares against frozen (cycle-0 K50). Non-destructive."""
import sys, time, json, pickle
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import torch
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, apply_routing, clone_splits, set_seed
from te.lp_solver import solve_selected_path_lp_dbbudget
from te.disturbance import compute_disturbance
import scripts.phase1_5.agnostic_lib as A
from scripts.phase1_5.bottleneck_lib import ACTIONS, ANAME
from scripts.phase1_5.run_final_iter2 import kp_for, build_mixed, pad_to_lib

set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
AGN = OUT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN" / "_cache"; KP = OUT / "FINAL_LEARNED_4OF5_KPATH4_DDQN" / "_cache"
SC = json.load(open(AGN / "scaler.json")); MEAN = np.array(SC["mean"], np.float32); STD = np.array(SC["std"], np.float32)
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))
TOPO, LO, HI, GNN_MS = "vtlwavenet2011", 0, 40, 140
d = P[(TOPO, LO, HI)]; caps = np.asarray(d["caps"], float)
env = _make_envs([TOPO], {TOPO:(LO,HI)}, gnn, HI-LO, 30)[0]; ctx = env.ctx
ds, pl, ecmp = ctx["ds"], ctx["pl"], ctx["ecmp"]
raws = pickle.load(open(AGN / f"raw_EVAL_{TOPO}.pkl","rb")); rankings = pickle.load(open(KP / f"rank_EVAL_{TOPO}.pkl","rb"))
sp_csv = OUT / "STRICT_FULL_MCF_PR" / "_partial" / f"{TOPO}.csv"; NUM = {}
if sp_csv.exists():
    s = pd.read_csv(sp_csv); NUM = {int(r.tm_index): float(r.strict_full_mcf_MLU) for r in s.itertuples() if getattr(r,"mcf_status","Optimal")=="Optimal"}
dim = len(A.AGN_FEAT_NAMES)
ck = torch.load(OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN" / "final_learned_4of5_iter2_model.pt", map_location="cpu")
net = A.QNet(dim, 7); net.load_state_dict(ck["state_dict"]); net.eval()
def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0

def run(cyc0_K):  # cyc0_K=None -> frozen; else force cycle-0 optimize at this K with db=1.0
    accepted = clone_splits(ecmp); rows = []
    for i, t in enumerate(range(LO, HI)):
        tm = np.asarray(ds.tm[t], float); keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
        raw, emlu = raws[t]; s = A.standardize(A.raw_to_vec(raw, keep_mlu, emlu), MEAN, STD)
        with torch.no_grad(): a = int(net(torch.tensor(s).unsqueeze(0)).argmax())
        kind, K, _ = ACTIONS[a]; dbb = 0.051
        if cyc0_K is not None and i == 0:
            kind, K, dbb = "opt", cyc0_K, 1.0
        if kind == "keep":
            mlu = keep_mlu; ms = 0.5; sp = accepted
        else:
            kp = kp_for(K); sel = list(rankings[t][:K]); plm = build_mixed(pl, set(int(o) for o in sel), kp); s0 = time.perf_counter()
            lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp, path_library=plm,
                capacities=caps, prev_splits=accepted, db_budget=dbb, db_weight=1e-6, time_limit_sec=120)
            sp = pad_to_lib(lp.splits, pl); mlu = float(apply_routing(tm, sp, pl, caps).mlu); ms = (time.perf_counter()-s0)*1000 + GNN_MS
        num = NUM.get(t, d["opt"][t])
        rows.append(dict(tm_index=t, PR=pr_of(num, mlu), DB=float(compute_disturbance(accepted, sp, tm)), ms=ms, cyc0=(i==0))); accepted = sp
    return pd.DataFrame(rows)

print("VtlWavenet first-cycle test (cycles>=1 frozen K50; only cycle 0 forced)\n", flush=True)
print(f"{'cyc0 K':>8s}{'cyc0 PR':>9s}{'min PR':>8s}{'mean PR':>9s}{'mean ms':>9s}{'p95 ms':>8s}{'cyc0 ms':>9s}")
res=[]
for cyc0 in [None, 500, 800, 1200]:
    g = run(cyc0); c0 = g[g.cyc0].iloc[0]
    lbl = "frozen(50)" if cyc0 is None else str(cyc0)
    print(f"{lbl:>8s}{c0.PR:>9.4f}{g.PR.min():>8.4f}{g.PR.mean():>9.4f}{g.ms.mean():>9.1f}{np.percentile(g.ms,95):>8.1f}{c0.ms:>9.1f}", flush=True)
    res.append(dict(cyc0_K=lbl, cyc0_PR=round(c0.PR,4), min_PR=round(g.PR.min(),4), mean_PR=round(g.PR.mean(),4),
        mean_ms=round(g.ms.mean(),1), p95_ms=round(float(np.percentile(g.ms,95)),1), cyc0_ms=round(c0.ms,1)))
pd.DataFrame(res).to_csv(OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN" / "vtl_firstcycle_test.csv", index=False)
print("\nDONE")
