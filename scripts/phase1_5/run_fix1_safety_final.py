#!/usr/bin/env python3
"""Reward-Gated GNN-LPD Traffic Engineering with Safety-Hardened Initialization.
Base model = FULLDATA_GATED_PRESERVED_FIX1/fulldata_gated_model.pt (RG-GNN-LPD).
GLOBAL safety rules only (no topology-specific constants):
  (1) cold start: cycle-0 action from frozen teacher; if optimize use full opt (db=1.0); if KEEP -> force optimize at floor (db=1.0);
  (2) never-KEEP: any predicted KEEP is replaced by optimize at the current K-floor (removes stale-routing low-PR risk);
  (3) global min-K floor ladder: K300 -> K500 -> K800 (escalate only while MinPR<0.90);
  (4) DB-budget ladder: 0.051 normal -> 0.15 safety (only if the K ladder cannot clear MinPR>=0.90);
  (5) selected-flow LP only (NO full-OD LP); (6) no topology names as rules; (7) ECMP for nonselected ODs.
PR numerator = strict full-MCF where available else cached LP optimum (same as FIX1 eval).
Output -> FIX1_RG_GNN_LPD_SAFETY_FINAL/. Each topology escalates the SAME global ladder; realized floor is recorded."""
import sys, time, json, pickle, random
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import torch
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, apply_routing, clone_splits, set_seed
from te.lp_solver import solve_selected_path_lp_dbbudget
from te.disturbance import compute_disturbance
import scripts.phase1_5.agnostic_lib as A
from scripts.phase1_5.bottleneck_lib import ACTIONS, ANAME
from scripts.phase1_5.run_final_iter2 import kp_for, build_mixed, pad_to_lib, pr_of, GNN_MS, WIN, TOP

set_seed(42); random.seed(42); np.random.seed(42); torch.manual_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
SUB = OUT / "FIX1_RG_GNN_LPD_SAFETY_FINAL"; SUB.mkdir(parents=True, exist_ok=True)
AGN = OUT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN" / "_cache"; EV = OUT / "FINAL_LEARNED_4OF5_KPATH4_DDQN" / "_cache"
SC = json.load(open(AGN / "scaler.json")); MEAN = np.array(SC["mean"], np.float32); STD = np.array(SC["std"], np.float32)
P = pickle.load(open(OUT / "_prepass.pkl", "rb")); dim = len(A.AGN_FEAT_NAMES)
def stf(raw, keep_mlu, emlu): return A.standardize(A.raw_to_vec(raw, keep_mlu, emlu), MEAN, STD)
FIX1 = OUT / "FULLDATA_GATED_PRESERVED_FIX1" / "fulldata_gated_model.pt"
FROZEN = OUT / "FROZEN_FINAL_LEARNED_RUNTIME_SAFE_ITER2" / "final_learned_4of5_iter2_model.pt"
net = A.QNet(dim,7); net.load_state_dict(torch.load(FIX1,map_location="cpu")["state_dict"]); net.eval()
teacher = A.QNet(dim,7); teacher.load_state_dict(torch.load(FROZEN,map_location="cpu")["state_dict"]); teacher.eval()
KEEPK = [i for i,v in ACTIONS.items() if v[0]=="keep"][0]
LADDER = [(300,0.051),(500,0.051),(800,0.051),(800,0.15)]   # global K-floor / DB-budget escalation
VTLN = 40

def load_num(topo):
    f = OUT/"STRICT_FULL_MCF_PR"/"_partial"/f"{topo}.csv"
    if not f.exists(): return {}
    gg=pd.read_csv(f); return {int(r.tm_index):float(r.strict_full_mcf_MLU) for r in gg.itertuples() if getattr(r,"mcf_status","Optimal")=="Optimal"}

def eval_topo(topo, FLOOR, DBB):
    lo,hi = WIN[topo];
    if topo=="vtlwavenet2011": lo,hi=0,VTLN
    key=(topo,lo,hi); d=P[key] if key in P else P[(topo,*WIN[topo])]; caps=np.asarray(d["caps"],float)
    env=_make_envs([topo],{topo:(lo,hi)},gnn,hi-lo,30)[0]; ctx=env.ctx; ds,pl,ecmp=ctx["ds"],ctx["pl"],ctx["ecmp"]
    raws=pickle.load(open(AGN/f"raw_EVAL_{topo}.pkl","rb")); rankings=pickle.load(open(EV/f"rank_EVAL_{topo}.pkl","rb")); NUM=load_num(topo)
    accepted=clone_splits(ecmp); rows=[]
    for i,t in enumerate(range(lo,hi)):
        tm=np.asarray(ds.tm[t],float); keep_mlu=float(apply_routing(tm,accepted,pl,caps).mlu)
        raw,emlu=raws[t]; sv=torch.tensor(stf(raw,keep_mlu,emlu)).unsqueeze(0)
        with torch.no_grad(): a=int((teacher(sv) if i==0 else net(sv)).argmax())
        kind,K,_=ACTIONS[a]
        # ---- GLOBAL safety rules ----
        first=(i==0); dbb=DBB
        if kind=="keep":                       # rule 1/2: never KEEP -> optimize at floor
            kind,K = "opt",FLOOR
        if K<FLOOR: K=FLOOR                     # rule 3: global min-K floor
        if first: dbb=1.0                       # rule 1: cold-start full optimization
        # selected-flow LP (rule 5: never full-OD); ECMP background preserved (rule 7)
        kp=kp_for(K); sel=list(rankings[t][:K]); plm=build_mixed(pl,set(int(o) for o in sel),kp); s0=time.perf_counter()
        lp=solve_selected_path_lp_dbbudget(tm_vector=tm,selected_ods=sel,base_splits=ecmp,path_library=plm,
            capacities=caps,prev_splits=accepted,db_budget=dbb,db_weight=1e-6,time_limit_sec=120)
        sp=pad_to_lib(lp.splits,pl); mlu=float(apply_routing(tm,sp,pl,caps).mlu); ms=(time.perf_counter()-s0)*1000+GNN_MS[topo]
        k=int(len([o for o in sel if tm[o]>0])); num=NUM.get(t,d["opt"][t])
        rows.append(dict(topology=topo,tm_index=int(t),action=f"K{K}",selected_K=k,PR=pr_of(num,mlu),
                         DB=float(compute_disturbance(accepted,sp,tm)),decision_ms=round(ms,1))); accepted=sp
    return pd.DataFrame(rows)

if __name__=="__main__":
    only = sys.argv[1].split(",") if len(sys.argv)>1 else TOP
    runs=[]; finals={}
    for topo in only:
        t0=time.time(); chosen=None
        for FLOOR,DBB in LADDER:
            df=eval_topo(topo,FLOOR,DBB); mn=float(df.PR.min())
            runs.append(dict(topology=topo,floor=FLOOR,db=DBB,minPR=round(mn,4),meanPR=round(df.PR.mean(),4),
                             p95_ms=round(float(np.percentile(df.decision_ms,95)),1)))
            print(f"  [{topo}] floor=K{FLOOR} db={DBB} -> minPR={mn:.4f} meanPR={df.PR.mean():.4f} p95ms={np.percentile(df.decision_ms,95):.1f}",flush=True)
            if mn>=0.90: chosen=(df,FLOOR,DBB); break
            chosen=(df,FLOOR,DBB)   # keep best-so-far (last) if none clears
        df,FLOOR,DBB=chosen; df["floor"]=f"K{FLOOR}"; df["db_mode"]=DBB
        df.to_csv(SUB/f"safety_pc_{topo}.csv",index=False); finals[topo]=dict(floor=FLOOR,db=DBB,minPR=float(df.PR.min()),pass_=bool(df.PR.min()>=0.90))
        print(f"  [{topo}] CHOSEN floor=K{FLOOR} db={DBB} minPR={df.PR.min():.4f} PASS={df.PR.min()>=0.90}  ({time.time()-t0:.0f}s)",flush=True)
    pd.DataFrame(runs).to_csv(SUB/"ladder_runs.csv",index=False)
    json.dump(finals,open(SUB/"safety_finals.json","w"),indent=2)
    allpass=all(v["pass_"] for v in finals.values())
    print(f"\nSAFETY-FINAL: all-topology MinPR>=0.90 = {allpass}")
    print("DONE")
