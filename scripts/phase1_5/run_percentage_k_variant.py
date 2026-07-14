#!/usr/bin/env python3
"""PERCENTAGE-K ACTION-SPACE VARIANT (new non-destructive method variant).
Action space (NO topology name; K computed per cycle from active OD count):
  0 KEEP
  1 K_3%   -> ceil(0.03*active)
  2 K_5%   -> ceil(0.05*active)
  3 K_10%  -> ceil(0.10*active)
  4 K_20%  -> ceil(0.20*active)
  5 K_30%  -> ceil(0.30*active)
  6 K_800CAP -> min(800, active)
Rules: K = clamp(max(K_percent, K_MIN), 800); K_MIN=20 floor (small topos); K800 absolute cap.
Reuses full-train raw+rankings (action-independent) from RETRAIN_PHASE1_FULLTRAIN/_cache_full; regenerates
opt-tables for the percentage actions; trains from scratch (same train/test split; Germany50+VtlWavenet zero-shot);
evaluates all 8 and compares to frozen Tier A. Frozen Tier A + report untouched. Output -> PERCENTAGE_K_ACTION_VARIANT/."""
import sys, time, json, pickle, random
from collections import deque
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import torch, torch.nn as nn
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, apply_routing, clone_splits, set_seed
from te.lp_solver import solve_selected_path_lp_dbbudget
from te.disturbance import compute_disturbance
import scripts.phase1_5.agnostic_lib as A
from scripts.phase1_5.run_final_iter2 import kp_for, build_mixed, pad_to_lib, pr_of, target_pr, GNN_MS, WIN, TOP

set_seed(42); random.seed(42); np.random.seed(42); torch.manual_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
SUB = OUT / "PERCENTAGE_K_ACTION_VARIANT"; OC = SUB / "_cache"; OC.mkdir(parents=True, exist_ok=True)
FULL = OUT / "RETRAIN_PHASE1_FULLTRAIN" / "_cache_full"          # reuse full-train raw + rankings
AGN_CACHE = OUT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN" / "_cache"; EVCACHE = OUT / "FINAL_LEARNED_4OF5_KPATH4_DDQN" / "_cache"
SC = json.load(open(AGN_CACHE / "scaler.json")); MEAN = np.array(SC["mean"], np.float32); STD = np.array(SC["std"], np.float32)
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))
TRAIN_FULL = {"abilene":(0,2016),"geant":(0,672),"cernet":(0,200),"sprintlink":(0,200),"tiscali":(0,200),"ebone":(0,200)}
SEEN = list(TRAIN_FULL); K_MIN, K_CAP = 20, 800
ANAMES = ["KEEP","K_3%","K_5%","K_10%","K_20%","K_30%","K_800CAP"]; PCTS = [None,0.03,0.05,0.10,0.20,0.30,None]
def k_for(idx, active):
    if idx == 0: return 0                                   # KEEP
    if idx == 6: return int(min(K_CAP, active))             # K_800CAP
    return int(min(max(int(np.ceil(PCTS[idx]*active)), K_MIN), K_CAP))
dim = len(A.AGN_FEAT_NAMES)
def st(raw, keep_mlu, emlu): return A.standardize(A.raw_to_vec(raw, keep_mlu, emlu), MEAN, STD)

# ---- regenerate opt-tables for percentage actions (resumable; dedup distinct K per cycle) ----
def precompute_opt(topo):
    f = OC / f"optpct_{topo}.pkl"
    if f.exists(): print(f"[skip optpct] {topo}", flush=True); return
    lo, hi = TRAIN_FULL[topo]; d = P[(topo, lo, hi)]; caps = np.asarray(d["caps"], float)
    env = _make_envs([topo], {topo:(lo,hi)}, gnn, hi-lo, 30)[0]; ctx = env.ctx; ds, pl, ecmp = ctx["ds"], ctx["pl"], ctx["ecmp"]
    rankings = pickle.load(open(FULL / f"rank_{topo}.pkl","rb")); otab = {}; t0 = time.perf_counter()
    for t in range(lo, hi):
        tm = np.asarray(ds.tm[t], float); active = int(d["tmstat"][t][3]); ranked = rankings[t]
        solved = {}                                          # K -> result (dedup)
        for idx in range(1, 7):
            K = k_for(idx, active)
            if K not in solved:
                kp = kp_for(K); sel = list(ranked[:K]); sset = set(int(o) for o in sel); plm = build_mixed(pl, sset, kp); s0 = time.perf_counter()
                lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp, path_library=plm,
                    capacities=caps, prev_splits=ecmp, db_budget=1.0, db_weight=1e-6, time_limit_sec=60)
                ms = (time.perf_counter()-s0)*1000 + GNN_MS[topo]; sp8 = pad_to_lib(lp.splits, pl)
                mlu = float(apply_routing(tm, sp8, pl, caps).mlu)
                so = np.array(sorted(int(o) for o in sel if tm[o] > 0), np.int32); ss = [np.asarray(sp8[int(o)], np.float32) for o in so]
                solved[K] = dict(mlu=mlu, ms=float(ms), sel_ods=so, sel_splits=ss, K=K)
            otab[(t, idx)] = solved[K]
        if (t-lo) % 200 == 0: print(f"  [optpct {topo}] {t-lo}/{hi-lo} ({time.perf_counter()-t0:.0f}s)", flush=True)
    pickle.dump(otab, open(f,"wb")); print(f"[optpct done] {topo} {time.perf_counter()-t0:.0f}s", flush=True)

# ---- frozen target-aware reward (unchanged; isolates the action-design change) ----
W_PR,W_MLU,W_DB,W_MS,W_K = 10.0,5.0,20.0,0.003,0.5
BONUS,TARGET_GATE,KEEP_GATE,KEEP_FLAT,MS_GATE = 10.0,25.0,25.0,4.0,25.0
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
        else: r-=TARGET_GATE*(tgt-PR)
    if ms>500.0: r-=MS_GATE*((ms-500.0)/500.0)
    return r

def train():
    CTX={}
    for topo in SEEN:
        lo,hi=TRAIN_FULL[topo]; d=P[(topo,lo,hi)]; caps=np.asarray(d["caps"],float); env=_make_envs([topo],{topo:(lo,hi)},gnn,hi-lo,30)[0]; ctx=env.ctx
        raws=pickle.load(open(FULL/f"raw_{topo}.pkl","rb")); otab=pickle.load(open(OC/f"optpct_{topo}.pkl","rb")); tgt=target_pr(topo)
        feas={t: any(pr_of(d["opt"][t],otab[(t,i)]["mlu"])>=tgt and otab[(t,i)]["ms"]<500.0 for i in range(1,7)) for t in range(lo,hi)}
        CTX[topo]=dict(d=d,caps=caps,ds=ctx["ds"],pl=ctx["pl"],ecmp=ctx["ecmp"],raws=raws,otab=otab,lo=lo,hi=hi,tgt=tgt,feas=feas)
    def recon(ecmp,e):
        f=clone_splits(ecmp)
        for i,od in enumerate(e["sel_ods"]): f[int(od)]=np.asarray(e["sel_splits"][i],float)
        return f
    online=A.QNet(dim,7); target=A.QNet(dim,7); target.load_state_dict(online.state_dict()); target.eval()
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
                keep_mlu=float(apply_routing(tm,accepted,c["pl"],c["caps"]).mlu); raw,emlu=c["raws"][t]; s=st(raw,keep_mlu,emlu); eps=eps_at(g)
                a=random.randrange(7) if random.random()<eps else int(online(torch.tensor(s).unsqueeze(0)).argmax())
                is_keep=(a==0)
                if is_keep: mlu=keep_mlu; ms=0.5; k=0; newacc=accepted
                else: e=c["otab"][(t,a)]; mlu=e["mlu"]; ms=e["ms"]; k=int(len(e["sel_ods"])); newacc=recon(c["ecmp"],e)
                DB=0.0 if is_keep else min(float(compute_disturbance(accepted,newacc,tm)),0.10)
                PR=pr_of(opt_mlu,mlu); mex=max(0.0,mlu/opt_mlu-1.0) if opt_mlu>0 else 0.0
                r=reward(PR,mex,DB,ms,k,nact,is_keep,c["tgt"],c["feas"][t]); rewards.append(r)
                if prev is not None: replay.append((prev[0],prev[1],prev[2],s,0.0))
                prev=(s,a,r); accepted=newacc; g+=1; CNT["env_steps"]+=1
                l=upd()
                if l is not None: losses.append(l)
                if g%TUPD==0: target.load_state_dict(online.state_dict()); CNT["target_updates"]+=1
            if prev is not None: replay.append((prev[0],prev[1],prev[2],np.zeros(dim,np.float32),1.0))
        ml=float(np.mean(losses)) if losses else float("nan"); mr=float(np.mean(rewards)) if rewards else float("nan")
        tlog.append(dict(episode=ep+1,mean_td_loss=round(ml,5),mean_reward=round(mr,4),epsilon=round(eps_at(g),4)))
        print(f"  [pctK] ep{ep+1:2d} td_loss={ml:.4f} mean_r={mr:.3f} eps={eps_at(g):.3f}",flush=True)
    torch.save({"state_dict":online.state_dict(),"dim":dim,"n_act":7,"action_space":"percentage_K","action_names":ANAMES,"K_MIN":K_MIN,"K_CAP":K_CAP},SUB/"pctk_model.pt")
    pd.DataFrame(tlog).to_csv(SUB/"pctk_train_log.csv",index=False); json.dump(CNT,open(SUB/"pctk_counters.json","w"),indent=2)
    print(f"[saved] pctK model. counters={CNT}",flush=True)

def load_num(topo):
    f=OUT/"STRICT_FULL_MCF_PR"/"_partial"/f"{topo}.csv"
    if not f.exists(): return {}
    g=pd.read_csv(f); return {int(r.tm_index):float(r.strict_full_mcf_MLU) for r in g.itertuples() if getattr(r,"mcf_status","Optimal")=="Optimal"}
def evaluate():
    ck=torch.load(SUB/"pctk_model.pt",map_location="cpu"); net=A.QNet(dim,7); net.load_state_dict(ck["state_dict"]); net.eval(); pcs=[]
    for topo in TOP:
        lo,hi=WIN[topo]; d=P[(topo,lo,hi)]; caps=np.asarray(d["caps"],float); env=_make_envs([topo],{topo:(lo,hi)},gnn,hi-lo,30)[0]; ctx=env.ctx; ds,pl,ecmp=ctx["ds"],ctx["pl"],ctx["ecmp"]
        raws=pickle.load(open(AGN_CACHE/f"raw_EVAL_{topo}.pkl","rb")); rankings=pickle.load(open(EVCACHE/f"rank_EVAL_{topo}.pkl","rb")); NUM=load_num(topo)
        accepted=clone_splits(ecmp); rows=[]; print(f"[eval] {topo}",flush=True)
        for i,t in enumerate(range(lo,hi)):
            tm=np.asarray(ds.tm[t],float); active=len(rankings[t]); keep_mlu=float(apply_routing(tm,accepted,pl,caps).mlu)
            raw,emlu=raws[t]; s=st(raw,keep_mlu,emlu)
            with torch.no_grad(): a=int(net(torch.tensor(s).unsqueeze(0)).argmax())
            if a==0: mlu=keep_mlu; ms=0.5; k=0; sp=accepted; Ku=0
            else:
                Ku=k_for(a,active); kp=kp_for(Ku); sel=list(rankings[t][:Ku]); plm=build_mixed(pl,set(int(o) for o in sel),kp); s0=time.perf_counter()
                lp=solve_selected_path_lp_dbbudget(tm_vector=tm,selected_ods=sel,base_splits=ecmp,path_library=plm,capacities=caps,prev_splits=accepted,db_budget=0.051,db_weight=1e-6,time_limit_sec=120)
                sp=pad_to_lib(lp.splits,pl); mlu=float(apply_routing(tm,sp,pl,caps).mlu); ms=(time.perf_counter()-s0)*1000+GNN_MS[topo]; k=int(len([o for o in sel if tm[o]>0]))
            num=NUM.get(t,d["opt"][t])
            rows.append(dict(topology=topo,tm_index=int(t),action=ANAMES[a],K_used=int(Ku),selected_K=int(k),PR=pr_of(num,mlu),DB=float(compute_disturbance(accepted,sp,tm)),MLU=mlu,decision_ms=round(ms,1))); accepted=sp
        pcs.append(pd.DataFrame(rows))
    pc=pd.concat(pcs,ignore_index=True); pc.to_csv(SUB/"pctk_eval_per_cycle.csv",index=False); return pc

if __name__=="__main__":
    print(f"PERCENTAGE-K VARIANT (K_MIN={K_MIN}, cap={K_CAP}); reuse full-train raw+rank; regen opt-tables\n",flush=True)
    for topo in SEEN: precompute_opt(topo)
    train(); pc=evaluate()
    frozen=pd.read_csv(OUT/"FROZEN_FINAL_LEARNED_RUNTIME_SAFE_ITER2"/"final_learned_4of5_iter2_eval_per_cycle.csv")
    GN={'abilene':3,'geant':7,'cernet':22,'sprintlink':27,'tiscali':33,'ebone':12,'germany50':26,'vtlwavenet2011':140}
    frozen['dms']=[r.decision_ms if r.action!='KEEP' else GN[r.topology]+0.5 for r in frozen.itertuples()]
    pc['dms']=[r.decision_ms if r.action!='KEEP' else GN[r.topology]+0.5 for r in pc.itertuples()]
    rows=[]
    for t in TOP:
        f=frozen[frozen.topology==t]; p=pc[pc.topology==t]
        rows.append(dict(topology=t, fz_meanPR=round(f.PR.mean(),4), pk_meanPR=round(p.PR.mean(),4), fz_minPR=round(f.PR.min(),4), pk_minPR=round(p.PR.min(),4),
            fz_pr90=round((f.PR>=0.90).mean()*100,1), pk_pr90=round((p.PR>=0.90).mean()*100,1), fz_meanDB=round(f.DB.mean(),4), pk_meanDB=round(p.DB.mean(),4),
            fz_p95DB=round(np.percentile(f.DB,95),4), pk_p95DB=round(np.percentile(p.DB,95),4),
            fz_p95ms=round(np.percentile(f.dms,95),1), pk_p95ms=round(np.percentile(p.dms,95),1), pk_maxms=round(p.dms.max(),1)))
    cmp=pd.DataFrame(rows); cmp.to_csv(SUB/"comparison_vs_frozen.csv",index=False)
    ad=[dict(Topology=t, **{a:int((pc[pc.topology==t].action==a).sum()) for a in ANAMES}, median_K=int(pc[(pc.topology==t)&(pc.K_used>0)].K_used.median()) if ((pc.topology==t)&(pc.K_used>0)).any() else 0) for t in TOP]
    pd.DataFrame(ad).to_csv(SUB/"action_distribution.csv",index=False)
    print("\n=== PERCENTAGE-K VARIANT vs FROZEN Tier A ==="); print(cmp.to_string(index=False))
    print("\n=== action distribution ==="); print(pd.DataFrame(ad).to_string(index=False)); print("\nDONE")
