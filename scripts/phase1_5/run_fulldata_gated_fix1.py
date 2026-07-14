#!/usr/bin/env python3
"""FULL-DATA gated DDQN, initialized from the frozen gated policy and FINE-TUNED with behavior-preservation
(distillation) regularization. Satisfies the full-data-training requirement WITHOUT re-opening the rejected
free full-retrain (which collapsed Germany50/Tiscali/Abilene).

Method UNCHANGED: Topology-Agnostic Bottleneck-Ranking DDQN + Gated First-Cycle Initialization (fixed-K actions,
bottleneck/GNN ranking, selected-flow LP, ECMP for nonselected, gated first-cycle rule). No percentage-K, no
coverage-aware wrapper, no RF, no topology-specific rules, no full-OD LP.

Full training partitions (no 160 cap): Abilene 2016, GEANT 672, CERNET/Sprintlink/Tiscali/Ebone 200.
Germany50 + VtlWavenet = zero-shot (eval only). Reuses full-partition caches (raw/rank/opt) from
RETRAIN_PHASE1_FULLTRAIN/_cache_full. Output -> FULLDATA_GATED_PRESERVED/.

Objective per batch: Double-DQN TD loss + LAMBDA_DISTILL * MSE(online_Q, frozen_teacher_Q)  (anchors policy to
the good frozen gated behavior). Low LR + low epsilon (conservative fine-tune). Checkpoint per epoch; select the
checkpoint on a real held-out validation split from the seen-topology training partitions, with teacher action-match
kept only as an auxiliary diagnostic."""
import sys, time, json, pickle, random, copy
from collections import deque
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import torch, torch.nn as nn
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, apply_routing, clone_splits, set_seed
from te.lp_solver import solve_selected_path_lp_dbbudget
from te.disturbance import compute_disturbance
import scripts.phase1_5.agnostic_lib as A
from scripts.phase1_5.bottleneck_lib import ACTIONS, ANAME
from scripts.phase1_5.run_final_iter2 import kp_for, build_mixed, pad_to_lib, pr_of, target_pr, GNN_MS, WIN, TOP

set_seed(42); random.seed(42); np.random.seed(42); torch.manual_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
SUB = OUT / "FULLDATA_GATED_PRESERVED_FIX1"; SUB.mkdir(parents=True, exist_ok=True)
AGN = OUT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN" / "_cache"; EV = OUT / "FINAL_LEARNED_4OF5_KPATH4_DDQN" / "_cache"
FC = OUT / "RETRAIN_PHASE1_FULLTRAIN" / "_cache_full"
SC = json.load(open(AGN / "scaler.json")); MEAN = np.array(SC["mean"], np.float32); STD = np.array(SC["std"], np.float32)
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))
TRAIN_FULL = {"abilene":(0,2016),"geant":(0,672),"cernet":(0,200),"sprintlink":(0,200),"tiscali":(0,200),"ebone":(0,200)}
VAL_COUNTS = {"abilene":202, "geant":67, "cernet":20, "sprintlink":20, "tiscali":20, "ebone":20}
TRAIN_SPLIT = {topo: (lo, hi - VAL_COUNTS[topo]) for topo, (lo, hi) in TRAIN_FULL.items()}
VAL_SPLIT = {topo: (hi - VAL_COUNTS[topo], hi) for topo, (lo, hi) in TRAIN_FULL.items()}
SEEN = list(TRAIN_FULL); dim = len(A.AGN_FEAT_NAMES); KEEP_IDX = [i for i,v in ACTIONS.items() if v[0]=="keep"][0]
FROZEN = OUT / "FROZEN_FINAL_LEARNED_RUNTIME_SAFE_ITER2" / "final_learned_4of5_iter2_model.pt"
def st(raw, keep_mlu, emlu): return A.standardize(A.raw_to_vec(raw, keep_mlu, emlu), MEAN, STD)

# ---- frozen target-aware reward (UNCHANGED) ----
W_PR,W_MLU,W_DB,W_MS,W_K = 10.0,5.0,20.0,0.003,0.5
BONUS,TARGET_GATE,KEEP_GATE,KEEP_FLAT,MS_GATE = 10.0,25.0,25.0,4.0,25.0
GAMMA,BATCH,BUFCAP,WARMUP,TUPD = 0.5,128,50000,500,500
EPOCHS, LR, EPS, LAMBDA_DISTILL = 4, 2e-4, 0.07, 5.0     # FIX1: stronger distillation + lower LR/eps
FCAP_W = 12.0     # first-cycle action-preservation (CE to frozen argmax on cold-start states)
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

def finetune():
    teacher = A.QNet(dim,7); teacher.load_state_dict(torch.load(FROZEN, map_location="cpu")["state_dict"]); teacher.eval()
    online = A.QNet(dim,7); online.load_state_dict(teacher.state_dict())          # START from frozen gated checkpoint
    target = A.QNet(dim,7); target.load_state_dict(online.state_dict()); target.eval()
    opt = torch.optim.Adam(online.parameters(), LR); huber = nn.SmoothL1Loss(); mse = nn.MSELoss(); replay = deque(maxlen=BUFCAP)
    CTX = {}
    for topo in SEEN:
        full_lo, full_hi = TRAIN_FULL[topo]
        train_lo, train_hi = TRAIN_SPLIT[topo]
        d = P[(topo,full_lo,full_hi)]; caps = np.asarray(d["caps"],float); env=_make_envs([topo],{topo:(full_lo,full_hi)},gnn,full_hi-full_lo,30)[0]; ctx=env.ctx
        raws=pickle.load(open(FC/f"raw_{topo}.pkl","rb")); otab=pickle.load(open(FC/f"opt_{topo}.pkl","rb")); tgt=target_pr(topo)
        feas={t: any(pr_of(d["opt"][t],otab[(t,K)]["mlu"])>=tgt and otab[(t,K)]["ms"]<500.0 for K in [50,100,200,300,500,800]) for t in range(train_lo,train_hi)}
        CTX[topo]=dict(d=d,caps=caps,ds=ctx["ds"],pl=ctx["pl"],ecmp=ctx["ecmp"],raws=raws,otab=otab,train_lo=train_lo,train_hi=train_hi,full_lo=full_lo,full_hi=full_hi,tgt=tgt,feas=feas)
    def recon(ecmp,e):
        f=clone_splits(ecmp)
        for i,od in enumerate(e["sel_ods"]): f[int(od)]=np.asarray(e["sel_splits"][i],float)
        return f
    CNT=dict(td_updates=0,distill_updates=0); g=0; tlog=[]; ckpts=[]
    def upd():
        if len(replay)<max(WARMUP,BATCH): return None,None
        b=random.sample(replay,BATCH); s=torch.tensor(np.array([x[0] for x in b])); a=torch.tensor([x[1] for x in b]).long().unsqueeze(1)
        r=torch.tensor([x[2] for x in b]).float().unsqueeze(1); s2=torch.tensor(np.array([x[3] for x in b])); dn=torch.tensor([x[4] for x in b]).float().unsqueeze(1)
        q=online(s).gather(1,a)
        with torch.no_grad(): astar=online(s2).argmax(1,keepdim=True); qn=target(s2).gather(1,astar); y=r+GAMMA*qn*(1-dn)
        td=huber(q,y)
        with torch.no_grad(): tq=teacher(s)                       # behavior-preservation target
        dl=mse(online(s),tq); loss=td+LAMBDA_DISTILL*dl
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(online.parameters(),10.0); opt.step()
        CNT["td_updates"]+=1; CNT["distill_updates"]+=1; return float(td.item()), float(dl.item())
    for ep in range(EPOCHS):
        order=SEEN[:]; random.shuffle(order); tds,dls,rs=[],[],[]
        for topo in order:
            c=CTX[topo]; accepted=clone_splits(c["ecmp"]); prev=None
            for t in range(c["train_lo"],c["train_hi"]):
                tm=np.asarray(c["ds"].tm[t],float); opt_mlu=c["d"]["opt"][t]; nact=c["d"]["tmstat"][t][3]
                keep_mlu=float(apply_routing(tm,accepted,c["pl"],c["caps"]).mlu); raw,emlu=c["raws"][t]; s=st(raw,keep_mlu,emlu)
                a=random.randrange(7) if random.random()<EPS else int(online(torch.tensor(s).unsqueeze(0)).argmax())
                kind,K,_=ACTIONS[a]; is_keep=(kind=="keep")
                if is_keep: mlu=keep_mlu; ms=0.5; k=0; newacc=accepted
                else: e=c["otab"][(t,K)]; mlu=e["mlu"]; ms=e["ms"]; k=int(len(e["sel_ods"])); newacc=recon(c["ecmp"],e)
                DB=0.0 if is_keep else min(float(compute_disturbance(accepted,newacc,tm)),0.10)
                PR=pr_of(opt_mlu,mlu); mex=max(0.0,mlu/opt_mlu-1.0) if opt_mlu>0 else 0.0
                rr=reward(PR,mex,DB,ms,k,nact,is_keep,c["tgt"],c["feas"][t]); rs.append(rr)
                if prev is not None: replay.append((prev[0],prev[1],prev[2],s,0.0))
                prev=(s,a,rr); accepted=newacc; g+=1
                td,dl=upd()
                if td is not None: tds.append(td); dls.append(dl)
                if g%TUPD==0: target.load_state_dict(online.state_dict())
            if prev is not None: replay.append((prev[0],prev[1],prev[2],np.zeros(dim,np.float32),1.0))
        ck = SUB/f"ckpt_ep{ep+1}.pt"; torch.save({"state_dict":copy.deepcopy(online.state_dict()),"dim":dim,"n_act":7,"epoch":ep+1}, ck); ckpts.append(ck)
        mt=float(np.mean(tds)) if tds else float("nan"); md=float(np.mean(dls)) if dls else float("nan"); mr=float(np.mean(rs))
        tlog.append(dict(epoch=ep+1,mean_td=round(mt,5),mean_distill=round(md,6),mean_reward=round(mr,4)))
        print(f"  [finetune] ep{ep+1} td={mt:.4f} distill={md:.5f} reward={mr:.3f}",flush=True)
    pd.DataFrame(tlog).to_csv(SUB/"finetune_train_log.csv",index=False); json.dump(CNT,open(SUB/"finetune_counters.json","w"),indent=2)
    return teacher, ckpts

def gated_eval(model, teacher, vtl_n=200):
    net=A.QNet(dim,7); net.load_state_dict(model.state_dict() if hasattr(model,'state_dict') else model); net.eval()
    EVALW=dict(WIN); EVALW["vtlwavenet2011"]=(0,vtl_n)
    def load_num(topo):
        f=OUT/"STRICT_FULL_MCF_PR"/"_partial"/f"{topo}.csv"
        if not f.exists(): return {}
        gg=pd.read_csv(f); return {int(r.tm_index):float(r.strict_full_mcf_MLU) for r in gg.itertuples() if getattr(r,"mcf_status","Optimal")=="Optimal"}
    pcs=[]
    for topo in TOP:
        lo,hi=EVALW[topo]; d=P[(topo,lo,hi)] if (topo,lo,hi) in P else P[(topo,*WIN[topo])]; caps=np.asarray(d["caps"],float)
        env=_make_envs([topo],{topo:(lo,hi)},gnn,hi-lo,30)[0]; ctx=env.ctx; ds,pl,ecmp=ctx["ds"],ctx["pl"],ctx["ecmp"]
        raws=pickle.load(open(AGN/f"raw_EVAL_{topo}.pkl","rb")); rankings=pickle.load(open(EV/f"rank_EVAL_{topo}.pkl","rb")); NUM=load_num(topo)
        accepted=clone_splits(ecmp); rows=[]
        for i,t in enumerate(range(lo,hi)):
            tm=np.asarray(ds.tm[t],float); nact=len(rankings[t]); keep_mlu=float(apply_routing(tm,accepted,pl,caps).mlu)
            raw,emlu=raws[t]; s=st(raw,keep_mlu,emlu); sv=torch.tensor(s).unsqueeze(0)
            with torch.no_grad():
                q = teacher(sv) if i==0 else net(sv)
                a = int(q.argmax())   # FIX1: first-TM action from FROZEN policy (cold-start preservation); cycles>=1 = fine-tuned
            if i == 0 and a == KEEP_IDX:
                # Cycle 0 has no previously accepted non-ECMP routing to "keep".
                # Guard against an invalid frozen-teacher KEEP by forcing the best non-KEEP optimize action.
                qn = q.clone()
                qn[..., KEEP_IDX] = -1e9
                a = int(qn.argmax())
            kind,K,_=ACTIONS[a]; dbb=0.051
            if i==0 and kind!="keep" and K>=300: dbb=1.0          # GATED first-cycle rule after the cycle-0 KEEP guard
            if kind=="keep": mlu=keep_mlu; ms=GNN_MS[topo]+0.5; k=0; sp=accepted
            else:
                kp=kp_for(K); sel=list(rankings[t][:K]); plm=build_mixed(pl,set(int(o) for o in sel),kp); s0=time.perf_counter()
                lp=solve_selected_path_lp_dbbudget(tm_vector=tm,selected_ods=sel,base_splits=ecmp,path_library=plm,capacities=caps,prev_splits=accepted,db_budget=dbb,db_weight=1e-6,time_limit_sec=120)
                sp=pad_to_lib(lp.splits,pl); mlu=float(apply_routing(tm,sp,pl,caps).mlu); ms=(time.perf_counter()-s0)*1000+GNN_MS[topo]; k=int(len([o for o in sel if tm[o]>0]))
            num=NUM.get(t,d["opt"][t])
            rows.append(dict(topology=topo,tm_index=int(t),action=ANAME[a],selected_K=k,PR=pr_of(num,mlu),DB=float(compute_disturbance(accepted,sp,tm)),decision_ms=round(ms,1))); accepted=sp
        pcs.append(pd.DataFrame(rows)); print(f"  [eval] {topo} done",flush=True)
    return pd.concat(pcs,ignore_index=True)

def action_match(ckpt, teacher):
    """cheap proxy: fraction of test cycles where ckpt argmax == frozen argmax (keep_mlu approx = emlu)."""
    net=A.QNet(dim,7); net.load_state_dict(torch.load(ckpt,map_location="cpu")["state_dict"]); net.eval(); m=tot=0
    for topo in TOP:
        raws=pickle.load(open(AGN/f"raw_EVAL_{topo}.pkl","rb"))
        for t,(raw,emlu) in raws.items():
            s=torch.tensor(st(raw,emlu,emlu)).unsqueeze(0)
            with torch.no_grad():
                if int(net(s).argmax())==int(teacher(s).argmax()): m+=1
            tot+=1
    return m/max(tot,1)


def validation_eval(ckpt, teacher):
    net=A.QNet(dim,7); net.load_state_dict(torch.load(ckpt,map_location="cpu")["state_dict"]); net.eval()
    rows=[]
    for topo in SEEN:
        full_lo, full_hi = TRAIN_FULL[topo]
        lo, hi = VAL_SPLIT[topo]
        d=P[(topo,full_lo,full_hi)]; caps=np.asarray(d["caps"],float); env=_make_envs([topo],{topo:(full_lo,full_hi)},gnn,full_hi-full_lo,30)[0]; ctx=env.ctx; ds,pl,ecmp=ctx["ds"],ctx["pl"],ctx["ecmp"]
        raws=pickle.load(open(FC/f"raw_{topo}.pkl","rb")); rankings=pickle.load(open(FC/f"rank_{topo}.pkl","rb"))
        accepted=clone_splits(ecmp)
        for i,t in enumerate(range(lo,hi)):
            tm=np.asarray(ds.tm[t],float); nact=len(rankings[t]); keep_mlu=float(apply_routing(tm,accepted,pl,caps).mlu)
            raw,emlu=raws[t]; s=st(raw,keep_mlu,emlu); sv=torch.tensor(s).unsqueeze(0)
            with torch.no_grad():
                q = teacher(sv) if i==0 else net(sv)
                a = int(q.argmax())
            if i == 0 and a == KEEP_IDX:
                qn = q.clone(); qn[..., KEEP_IDX] = -1e9; a = int(qn.argmax())
            kind,K,_=ACTIONS[a]; dbb=0.051
            if i==0 and kind!="keep" and K>=300: dbb=1.0
            if kind=="keep": mlu=keep_mlu; ms=GNN_MS[topo]+0.5; k=0; sp=accepted
            else:
                kp=kp_for(K); sel=list(rankings[t][:K]); plm=build_mixed(pl,set(int(o) for o in sel),kp); s0=time.perf_counter()
                lp=solve_selected_path_lp_dbbudget(tm_vector=tm,selected_ods=sel,base_splits=ecmp,path_library=plm,capacities=caps,prev_splits=accepted,db_budget=dbb,db_weight=1e-6,time_limit_sec=120)
                sp=pad_to_lib(lp.splits,pl); mlu=float(apply_routing(tm,sp,pl,caps).mlu); ms=(time.perf_counter()-s0)*1000+GNN_MS[topo]; k=int(len([o for o in sel if tm[o]>0]))
            num=d["opt"][t]
            rows.append(dict(topology=topo,tm_index=int(t),action=ANAME[a],selected_K=k,PR=pr_of(num,mlu),DB=float(compute_disturbance(accepted,sp,tm)),decision_ms=round(ms,1))); accepted=sp
    pc=pd.DataFrame(rows)
    return dict(mean_PR=float(pc.PR.mean()), pr90=float((pc.PR>=0.90).mean()*100.0), pr95=float((pc.PR>=0.95).mean()*100.0), mean_DB=float(pc.DB.mean()), mean_ms=float(pc.decision_ms.mean()), p95_ms=float(np.percentile(pc.decision_ms,95)))


def write_validation_split_protocol():
    rows = []
    for topo in SEEN:
        full_lo, full_hi = TRAIN_FULL[topo]
        train_lo, train_hi = TRAIN_SPLIT[topo]
        val_lo, val_hi = VAL_SPLIT[topo]
        rows.append(
            dict(
                topology=topo,
                full_train_tms=full_hi - full_lo,
                train_tms=train_hi - train_lo,
                validation_tms=val_hi - val_lo,
                train_range=f"[{train_lo}, {train_hi})",
                validation_range=f"[{val_lo}, {val_hi})",
            )
        )
    pd.DataFrame(rows).to_csv(SUB / "validation_split_protocol.csv", index=False)


def selection_key(entry):
    return (
        float(entry["pr95"]),
        float(entry["mean_PR"]),
        -float(entry["mean_DB"]),
        -float(entry["mean_ms"]),
        float(entry["action_match"]),
    )

if __name__=="__main__":
    print("FULL-DATA GATED (distillation-anchored fine-tune from frozen gated checkpoint)\n",flush=True)
    teacher, ckpts = finetune()
    write_validation_split_protocol()
    # checkpoint selection: held-out validation split first; frozen teacher action-match is auxiliary only
    sel=[]
    for ck in ckpts:
        vm = validation_eval(ck, teacher)
        am = action_match(ck, teacher)
        rec = {"checkpoint": ck.name, "action_match": am, **vm}
        sel.append(rec)
        print(
            f"  [select] {ck.name} val_pr95={vm['pr95']:.2f} val_meanPR={vm['mean_PR']:.4f} "
            f"val_meanDB={vm['mean_DB']:.4f} val_mean_ms={vm['mean_ms']:.1f} action_match={am:.4f}",
            flush=True,
        )
    sel = sorted(sel, key=selection_key, reverse=True)
    pd.DataFrame(sel).to_csv(SUB/"validation_selection_log.csv",index=False)
    best = sel[0]; best_ck = SUB / best["checkpoint"]
    print(
        f"\n[selected] {best_ck.name} "
        f"(val_pr95={best['pr95']:.2f}, val_meanPR={best['mean_PR']:.4f}, "
        f"val_meanDB={best['mean_DB']:.4f}, val_mean_ms={best['mean_ms']:.1f}, "
        f"action_match={best['action_match']:.4f})",
        flush=True,
    )
    bm=A.QNet(dim,7); bm.load_state_dict(torch.load(best_ck,map_location="cpu")["state_dict"])
    torch.save({"state_dict":bm.state_dict(),"dim":dim,"n_act":7,
                "claim":"Full-data-trained gated DDQN, initialized from the previous frozen policy and fine-tuned with behavior-preservation regularization",
                "fine_tuned":True,
                "from_scratch":False,
                "selected_checkpoint":best_ck.name,
                "selection_protocol":"held-out validation split on seen topologies; sort by (val_pr95, val_meanPR, -val_meanDB, -val_mean_ms, action_match)",
                "train_split":TRAIN_SPLIT,
                "validation_split":VAL_SPLIT}, SUB/"fulldata_gated_model.pt")
    json.dump(
        {
            "selected_checkpoint": best_ck.name,
            "selection_protocol": "held-out validation split on seen topologies; sort by (val_pr95, val_meanPR, -val_meanDB, -val_mean_ms, action_match)",
            "selected_validation_metrics": best,
        },
        open(SUB/"selected_validation_metrics.json","w"),
        indent=2,
    )
    print("\nSaved validation-selected FIX1 checkpoint and split protocol. Final N=3976 metrics must be regenerated by scripts/phase1_5/run_fix1_targeted_repairs.py.\nDONE")
