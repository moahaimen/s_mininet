#!/usr/bin/env python3
"""PHASE 1 retrain on the FULL train partitions (NEW VARIANT, non-destructive).
Train data (full partitions, no 160 cap): Abilene 0-2016, GEANT 0-672, CERNET/Sprintlink/Tiscali/Ebone 0-200.
Eval: held-out test (Abilene 2016-4032, GEANT 672-1344, others 200-400) + zero-shot Germany50(288), VtlWavenet(40).
Adds 2 safety features (is_first_tm, k50_coverage -> 35-dim) + shaped reward (under-coverage penalty).
Regenerates the train caches (raw, rankings, opt-tables) for the FULL partitions (the existing ones cover only
the first 160 cycles); refits the scaler on full train; retrains; evaluates all 8. Frozen Tier A untouched.
Output -> RETRAIN_PHASE1_FULLTRAIN/. Precompute is resumable (skips per-topo caches that already exist)."""
import sys, time, json, pickle, random
from collections import deque
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import torch, torch.nn as nn
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, apply_routing, clone_splits, set_seed
from te.lp_solver import solve_selected_path_lp_dbbudget
from te.disturbance import compute_disturbance
import scripts.phase1_5.agnostic_lib as A
from scripts.phase1_5.bottleneck_lib import ACTIONS, ANAME
from scripts.phase1_5.run_final_iter2 import kp_for, build_mixed, pad_to_lib, bottleneck_rank, pr_of, target_pr, GNN_MS, WIN, TOP

set_seed(42); random.seed(42); np.random.seed(42); torch.manual_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
SUB = OUT / "RETRAIN_PHASE1_FULLTRAIN"; FC = SUB / "_cache_full"; FC.mkdir(parents=True, exist_ok=True)
AGN_CACHE = OUT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN" / "_cache"; EVCACHE = OUT / "FINAL_LEARNED_4OF5_KPATH4_DDQN" / "_cache"
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))
TRAIN_FULL = {"abilene":(0,2016),"geant":(0,672),"cernet":(0,200),"sprintlink":(0,200),"tiscali":(0,200),"ebone":(0,200)}
SEEN = list(TRAIN_FULL)
KS = [50,100,200,300,500,800]
def k50cov(nact): return 50.0/max(int(nact),1)

# ---------- PRECOMPUTE full-train caches (raw + rankings + opt-tables), resumable ----------
def precompute_full(topo):
    rawf, rankf, optf = FC/f"raw_{topo}.pkl", FC/f"rank_{topo}.pkl", FC/f"opt_{topo}.pkl"
    if rawf.exists() and rankf.exists() and optf.exists(): print(f"[skip precompute] {topo}", flush=True); return
    lo, hi = TRAIN_FULL[topo]; d = P[(topo, lo, hi)]; caps = np.asarray(d["caps"], float)
    env = _make_envs([topo], {topo:(lo,hi)}, gnn, hi-lo, 30)[0]; ctx = env.ctx
    ds, pl, ecmp = ctx["ds"], ctx["pl"], ctx["ecmp"]
    dd = dict(ranked=d["ranked"], tm_cache=ds.tm, num_nodes=len(ds.nodes)); struct = A.struct_feats(ds)
    raws, rankings, otab = {}, {}, {}; t0 = time.perf_counter()
    for t in range(lo, hi):
        tm = np.asarray(ds.tm[t], float); util = apply_routing(tm, ecmp, pl, caps).utilization
        sc, _, _ = gnn.score(dataset=ds, tm_vector=tm, path_library=pl, capacities=caps, ecmp_base=ecmp); sc = np.asarray(sc, float).ravel()
        raws[t] = (A.raw_static(topo, t, dd, d, pl, ecmp, caps, sc, util, struct), float(d["emlu"][t]))
        ranked = bottleneck_rank(tm, ecmp, pl, caps, sc); rankings[t] = np.array(ranked, np.int32)
        for K in KS:
            kp = kp_for(K); sel = ranked[:K]; sset = set(int(o) for o in sel); plm = build_mixed(pl, sset, kp); s0 = time.perf_counter()
            lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp, path_library=plm,
                capacities=caps, prev_splits=ecmp, db_budget=1.0, db_weight=1e-6, time_limit_sec=60)
            ms = (time.perf_counter()-s0)*1000 + GNN_MS[topo]; splits8 = pad_to_lib(lp.splits, pl)
            mlu = float(apply_routing(tm, splits8, pl, caps).mlu)
            so = np.array(sorted(int(o) for o in sel if tm[o] > 0), np.int32); ss = [np.asarray(splits8[int(o)], np.float32) for o in so]
            otab[(t, K)] = dict(mlu=mlu, ms=float(ms), sel_ods=so, sel_splits=ss)
        if (t-lo) % 200 == 0: print(f"  [precompute {topo}] {t-lo}/{hi-lo} ({time.perf_counter()-t0:.0f}s)", flush=True)
    pickle.dump(raws, open(rawf,"wb")); pickle.dump(rankings, open(rankf,"wb")); pickle.dump(otab, open(optf,"wb"))
    print(f"[precompute done] {topo} {time.perf_counter()-t0:.0f}s", flush=True)

# ---------- extended scaler refit on FULL train ----------
def fit_scaler():
    vecs = []
    for topo in SEEN:
        raws = pickle.load(open(FC/f"raw_{topo}.pkl","rb")); lo,hi = TRAIN_FULL[topo]; d = P[(topo,lo,hi)]
        env = _make_envs([topo], {topo:(lo,hi)}, gnn, hi-lo, 30)[0]; ctx = env.ctx; pl, ecmp, caps = ctx["pl"], ctx["ecmp"], np.asarray(ctx["caps"],float); ds = ctx["ds"]
        acc = clone_splits(ecmp)
        for t in range(lo,hi):
            tm = np.asarray(ds.tm[t],float); keep_mlu = float(apply_routing(tm, acc, pl, caps).mlu)
            raw, emlu = raws[t]; base = np.asarray(A.raw_to_vec(raw, keep_mlu, emlu), np.float32)
            vecs.append(np.concatenate([base, [1.0 if t==lo else 0.0, k50cov(d["tmstat"][t][3])]]))
    V = np.array(vecs, np.float32); mean = V.mean(0); std = V.std(0); std[std<1e-6] = 1e-6
    json.dump({"mean":mean.tolist(),"std":std.tolist(),"feature_names":A.AGN_FEAT_NAMES+["is_first_tm","k50_coverage"]}, open(SUB/"scaler_ext.json","w"), indent=2)
    return mean, std

# ---------- shaped reward (target-aware + under-coverage penalty) ----------
W_PR,W_MLU,W_DB,W_MS,W_K = 10.0,5.0,20.0,0.003,0.5
BONUS,TARGET_GATE,KEEP_GATE,KEEP_FLAT,MS_GATE = 10.0,25.0,25.0,4.0,25.0
COV_TARGET,W_COV = 0.05,15.0
GAMMA,BATCH,BUFCAP,WARMUP,TUPD,EPISODES = 0.5,128,50000,500,500,22
EPS0,EPS1,EPSDECAY = 1.0,0.05,16000
def reward(PR,mex,DB,ms,k,nact,is_keep,tgt,feas):
    r = W_PR*PR - W_MLU*mex - W_DB*DB - W_MS*ms - W_K*(k/max(nact,1))
    if PR>=tgt: r+=BONUS
    else:
        if is_keep:
            if feas:
                r-=TARGET_GATE*(tgt-PR)+KEEP_FLAT
                if PR<0.90: r-=KEEP_GATE*(0.90-PR)
        else:
            r-=TARGET_GATE*(tgt-PR); cov=k/max(nact,1)
            if cov<COV_TARGET: r-=W_COV*(COV_TARGET-cov)
    if ms>500.0: r-=MS_GATE*((ms-500.0)/500.0)
    return r

DIM = len(A.AGN_FEAT_NAMES)+2
def train(MEAN, STD):
    def ext(raw,keep_mlu,emlu,is_first,nact):
        base=A.raw_to_vec(raw,keep_mlu,emlu); v=np.concatenate([np.asarray(base,np.float32),np.array([1.0 if is_first else 0.0,k50cov(nact)],np.float32)]); return ((v-MEAN)/STD).astype(np.float32)
    CTX={}
    for topo in SEEN:
        lo,hi=TRAIN_FULL[topo]; d=P[(topo,lo,hi)]; caps=np.asarray(d["caps"],float); env=_make_envs([topo],{topo:(lo,hi)},gnn,hi-lo,30)[0]; ctx=env.ctx
        raws=pickle.load(open(FC/f"raw_{topo}.pkl","rb")); otab=pickle.load(open(FC/f"opt_{topo}.pkl","rb")); tgt=target_pr(topo)
        feas={t: any(pr_of(d["opt"][t],otab[(t,K)]["mlu"])>=tgt and otab[(t,K)]["ms"]<500.0 for K in KS) for t in range(lo,hi)}
        CTX[topo]=dict(d=d,caps=caps,ds=ctx["ds"],pl=ctx["pl"],ecmp=ctx["ecmp"],raws=raws,otab=otab,lo=lo,hi=hi,tgt=tgt,feas=feas)
    def recon(ecmp,e):
        f=clone_splits(ecmp)
        for i,od in enumerate(e["sel_ods"]): f[int(od)]=np.asarray(e["sel_splits"][i],float)
        return f
    online=A.QNet(DIM,7); target=A.QNet(DIM,7); target.load_state_dict(online.state_dict()); target.eval()
    opt=torch.optim.Adam(online.parameters(),1e-3); huber=nn.SmoothL1Loss(); replay=deque(maxlen=BUFCAP); CNT=dict(env_steps=0,td_updates=0,target_updates=0,ce_updates=0); g=0
    def eps_at(s): return max(EPS1,EPS0-(EPS0-EPS1)*s/EPSDECAY)
    def upd():
        if len(replay)<max(WARMUP,BATCH): return None
        b=random.sample(replay,BATCH); s=torch.tensor(np.array([x[0] for x in b])); a=torch.tensor([x[1] for x in b]).long().unsqueeze(1)
        r=torch.tensor([x[2] for x in b]).float().unsqueeze(1); s2=torch.tensor(np.array([x[3] for x in b])); dn=torch.tensor([x[4] for x in b]).float().unsqueeze(1)
        q=online(s).gather(1,a)
        with torch.no_grad(): astar=online(s2).argmax(1,keepdim=True); qn=target(s2).gather(1,astar); y=r+GAMMA*qn*(1-dn)
        loss=huber(q,y); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(online.parameters(),10.0); opt.step(); CNT["td_updates"]+=1; return float(loss.item())
    tlog=[]
    for ep in range(EPISODES):
        order=SEEN[:]; random.shuffle(order); losses,rewards=[],[]
        for topo in order:
            c=CTX[topo]; accepted=clone_splits(c["ecmp"]); prev=None
            for t in range(c["lo"],c["hi"]):
                tm=np.asarray(c["ds"].tm[t],float); opt_mlu=c["d"]["opt"][t]; nact=c["d"]["tmstat"][t][3]
                keep_mlu=float(apply_routing(tm,accepted,c["pl"],c["caps"]).mlu); raw,emlu=c["raws"][t]
                s=ext(raw,keep_mlu,emlu,t==c["lo"],nact); eps=eps_at(g)
                a=random.randrange(7) if random.random()<eps else int(online(torch.tensor(s).unsqueeze(0)).argmax())
                kind,K,_=ACTIONS[a]; is_keep=(kind=="keep")
                if is_keep: mlu=keep_mlu; ms=0.5; k=0; newacc=accepted
                else: e=c["otab"][(t,K)]; mlu=e["mlu"]; ms=e["ms"]; k=int(len(e["sel_ods"])); newacc=recon(c["ecmp"],e)
                DB=0.0 if is_keep else min(float(compute_disturbance(accepted,newacc,tm)),0.10)
                PR=pr_of(opt_mlu,mlu); mex=max(0.0,mlu/opt_mlu-1.0) if opt_mlu>0 else 0.0
                r=reward(PR,mex,DB,ms,k,nact,is_keep,c["tgt"],c["feas"][t]); rewards.append(r)
                if prev is not None: replay.append((prev[0],prev[1],prev[2],s,0.0))
                prev=(s,a,r); accepted=newacc; g+=1; CNT["env_steps"]+=1
                l=upd()
                if l is not None: losses.append(l)
                if g%TUPD==0: target.load_state_dict(online.state_dict()); CNT["target_updates"]+=1
            if prev is not None: replay.append((prev[0],prev[1],prev[2],np.zeros(DIM,np.float32),1.0))
        ml=float(np.mean(losses)) if losses else float("nan"); mr=float(np.mean(rewards)) if rewards else float("nan")
        tlog.append(dict(episode=ep+1,mean_td_loss=round(ml,5),mean_reward=round(mr,4),epsilon=round(eps_at(g),4)))
        print(f"  [fulltrain] ep{ep+1:2d} td_loss={ml:.4f} mean_r={mr:.3f} eps={eps_at(g):.3f}",flush=True)
    torch.save({"state_dict":online.state_dict(),"dim":DIM,"n_act":7,"phase1_fulltrain":True},SUB/"phase1_full_model.pt")
    pd.DataFrame(tlog).to_csv(SUB/"phase1_full_train_log.csv",index=False); json.dump(CNT,open(SUB/"phase1_full_counters.json","w"),indent=2)
    print(f"[saved] full-train model. counters={CNT}",flush=True)

def load_num(topo):
    f=OUT/"STRICT_FULL_MCF_PR"/"_partial"/f"{topo}.csv"
    if not f.exists(): return {}
    g=pd.read_csv(f); return {int(r.tm_index):float(r.strict_full_mcf_MLU) for r in g.itertuples() if getattr(r,"mcf_status","Optimal")=="Optimal"}
def evaluate(MEAN, STD):
    def ext(raw,keep_mlu,emlu,is_first,nact):
        base=A.raw_to_vec(raw,keep_mlu,emlu); v=np.concatenate([np.asarray(base,np.float32),np.array([1.0 if is_first else 0.0,k50cov(nact)],np.float32)]); return ((v-MEAN)/STD).astype(np.float32)
    ck=torch.load(SUB/"phase1_full_model.pt",map_location="cpu"); net=A.QNet(DIM,7); net.load_state_dict(ck["state_dict"]); net.eval(); pcs=[]
    for topo in TOP:
        lo,hi=WIN[topo]; d=P[(topo,lo,hi)]; caps=np.asarray(d["caps"],float); env=_make_envs([topo],{topo:(lo,hi)},gnn,hi-lo,30)[0]; ctx=env.ctx; ds,pl,ecmp=ctx["ds"],ctx["pl"],ctx["ecmp"]
        raws=pickle.load(open(AGN_CACHE/f"raw_EVAL_{topo}.pkl","rb")); rankings=pickle.load(open(EVCACHE/f"rank_EVAL_{topo}.pkl","rb")); NUM=load_num(topo)
        accepted=clone_splits(ecmp); rows=[]; print(f"[eval] {topo}",flush=True)
        for i,t in enumerate(range(lo,hi)):
            tm=np.asarray(ds.tm[t],float); nact=len(rankings[t]); keep_mlu=float(apply_routing(tm,accepted,pl,caps).mlu)
            raw,emlu=raws[t]; s=ext(raw,keep_mlu,emlu,i==0,nact)
            with torch.no_grad(): a=int(net(torch.tensor(s).unsqueeze(0)).argmax())
            kind,K,_=ACTIONS[a]
            if kind=="keep": mlu=keep_mlu; ms=0.5; k=0; sp=accepted
            else:
                kp=kp_for(K); sel=list(rankings[t][:K]); plm=build_mixed(pl,set(int(o) for o in sel),kp); s0=time.perf_counter()
                lp=solve_selected_path_lp_dbbudget(tm_vector=tm,selected_ods=sel,base_splits=ecmp,path_library=plm,capacities=caps,prev_splits=accepted,db_budget=0.051,db_weight=1e-6,time_limit_sec=120)
                sp=pad_to_lib(lp.splits,pl); mlu=float(apply_routing(tm,sp,pl,caps).mlu); ms=(time.perf_counter()-s0)*1000+GNN_MS[topo]; k=int(len([o for o in sel if tm[o]>0]))
            num=NUM.get(t,d["opt"][t])
            rows.append(dict(topology=topo,tm_index=int(t),action=ANAME[a],selected_K=int(k),PR=pr_of(num,mlu),DB=float(compute_disturbance(accepted,sp,tm)),MLU=mlu,decision_ms=round(ms,1))); accepted=sp
        pcs.append(pd.DataFrame(rows))
    pc=pd.concat(pcs,ignore_index=True); pc.to_csv(SUB/"phase1_full_eval_per_cycle.csv",index=False); return pc

if __name__=="__main__":
    print("PHASE 1 FULL-TRAIN: full partitions (2016/672/200), +2 features + shaped reward\n",flush=True)
    for topo in SEEN: precompute_full(topo)
    MEAN, STD = fit_scaler(); print("[scaler] refit on full train", flush=True)
    train(MEAN, STD); pc=evaluate(MEAN, STD)
    frozen=pd.read_csv(OUT/"FROZEN_FINAL_LEARNED_RUNTIME_SAFE_ITER2"/"final_learned_4of5_iter2_eval_per_cycle.csv")
    rows=[]
    for t in TOP:
        f=frozen[frozen.topology==t]; p=pc[pc.topology==t]
        rows.append(dict(topology=t, fz_meanPR=round(f.PR.mean(),4), p1_meanPR=round(p.PR.mean(),4), fz_minPR=round(f.PR.min(),4), p1_minPR=round(p.PR.min(),4),
            fz_pr90=round((f.PR>=0.90).mean()*100,1), p1_pr90=round((p.PR>=0.90).mean()*100,1), fz_meanDB=round(f.DB.mean(),4), p1_meanDB=round(p.DB.mean(),4),
            fz_p95ms=round(np.percentile(f.decision_ms,95),1), p1_p95ms=round(np.percentile(p.decision_ms,95),1)))
    cmp=pd.DataFrame(rows); cmp.to_csv(SUB/"comparison_vs_frozen.csv",index=False)
    print("\n=== PHASE 1 FULL-TRAIN vs FROZEN Tier A ==="); print(cmp.to_string(index=False)); print("\nDONE")
