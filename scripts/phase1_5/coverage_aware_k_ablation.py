#!/usr/bin/env python3
"""ABLATION (non-destructive): COVERAGE-AWARE K rule.
Reproduces frozen Tier A exactly (same DDQN, ranking, LP, ECMP, db_budget=0.051, carry-forward), with ONE change:
when the DDQN chooses an OPTIMIZE action whose coverage (selected_K / active_OD_count) < 5%, raise K to the
smallest value reaching ~5% coverage, capped at a runtime-safe K500. NO topology name is used; the rule keys only
on the per-cycle coverage fraction (today it fires only on VtlWavenet, the only topology under 5%).
Runs baseline (validate vs frozen) + coverage-aware, all 8 topos. Saves to COVERAGE_AWARE_K_ABLATION/."""
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
SUB = OUT / "COVERAGE_AWARE_K_ABLATION"; SUB.mkdir(parents=True, exist_ok=True)
AGN = OUT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN" / "_cache"; KP = OUT / "FINAL_LEARNED_4OF5_KPATH4_DDQN" / "_cache"
SC = json.load(open(AGN / "scaler.json")); MEAN = np.array(SC["mean"], np.float32); STD = np.array(SC["std"], np.float32)
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))
WIN = {"abilene":(2016,4032),"geant":(672,1344),"cernet":(200,400),"sprintlink":(200,400),"tiscali":(200,400),"ebone":(200,400),"germany50":(0,288),"vtlwavenet2011":(0,40)}
GNN_MS = {"abilene":3,"geant":7,"cernet":22,"sprintlink":27,"tiscali":33,"ebone":12,"germany50":26,"vtlwavenet2011":140}
FLEX = {"abilene":(0.958,0.0513),"cernet":(0.975,0.0183),"geant":(0.995,0.0296),"sprintlink":(0.999,0.0510)}
DB_STEADY = 0.051
COV_TARGET = 0.05      # raise K toward 5% coverage
RUNTIME_CAP_K = 500    # runtime-safe cap (largest K with p95<500 per scalability sweep)
dim = len(A.AGN_FEAT_NAMES)
ck = torch.load(OUT / "FINAL_LEARNED_4OF5_ITER2_DDQN" / "final_learned_4of5_iter2_model.pt", map_location="cpu")
net = A.QNet(dim, 7); net.load_state_dict(ck["state_dict"]); net.eval()
def pr_of(o, m): return float(min(1.0, o / m)) if m > 0 else 0.0
def load_num(topo):
    p = OUT / "STRICT_FULL_MCF_PR" / "_partial" / f"{topo}.csv"; NUM = {}
    if p.exists():
        s = pd.read_csv(p); NUM = {int(r.tm_index): float(r.strict_full_mcf_MLU) for r in s.itertuples() if getattr(r,"mcf_status","Optimal")=="Optimal"}
    return NUM

def run(topo, apply_rule):
    lo, hi = WIN[topo]; d = P[(topo, lo, hi)]; caps = np.asarray(d["caps"], float)
    env = _make_envs([topo], {topo:(lo,hi)}, gnn, hi-lo, 30)[0]; ctx = env.ctx
    ds, pl, ecmp = ctx["ds"], ctx["pl"], ctx["ecmp"]
    raws = pickle.load(open(AGN / f"raw_EVAL_{topo}.pkl","rb")); rankings = pickle.load(open(KP / f"rank_EVAL_{topo}.pkl","rb"))
    NUM = load_num(topo); accepted = clone_splits(ecmp); rows = []
    for i, t in enumerate(range(lo, hi)):
        tm = np.asarray(ds.tm[t], float); keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
        raw, emlu = raws[t]; s = A.standardize(A.raw_to_vec(raw, keep_mlu, emlu), MEAN, STD)
        with torch.no_grad(): a = int(net(torch.tensor(s).unsqueeze(0)).argmax())
        kind, K, _ = ACTIONS[a]; fired = False; active = int((tm > 0).sum())
        if apply_rule and kind == "opt" and active > 0 and (K / active) < COV_TARGET:
            target = int(np.ceil(COV_TARGET * active))
            newK = min(max(K, target), RUNTIME_CAP_K)
            if newK > K: K, fired = newK, True
        if kind == "keep":
            mlu = keep_mlu; ms = 0.5; ksel = 0; sp = accepted
        else:
            kp = kp_for(K); sel = list(rankings[t][:K]); plm = build_mixed(pl, set(int(o) for o in sel), kp); s0 = time.perf_counter()
            lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp, path_library=plm,
                capacities=caps, prev_splits=accepted, db_budget=DB_STEADY, db_weight=1e-6, time_limit_sec=120)
            sp = pad_to_lib(lp.splits, pl); mlu = float(apply_routing(tm, sp, pl, caps).mlu); ms = (time.perf_counter()-s0)*1000 + GNN_MS[topo]; ksel = int(len([o for o in sel if tm[o] > 0]))
        num = NUM.get(t, d["opt"][t])
        rows.append(dict(topology=topo, tm_index=t, action=ANAME[a], selected_K=ksel, K_used=(0 if kind=='keep' else K),
            rule_fired=fired, PR=pr_of(num, mlu), DB=float(compute_disturbance(accepted, sp, tm)), decision_ms=round(ms,1))); accepted = sp
    return pd.DataFrame(rows)

frozen = pd.read_csv(OUT / "FROZEN_FINAL_LEARNED_RUNTIME_SAFE_ITER2" / "final_learned_4of5_iter2_eval_per_cycle.csv")
base_all, abl_all, comp = [], [], []
for topo in WIN:
    b = run(topo, False); f = run(topo, True)
    b["variant"]="baseline_repro"; f["variant"]="coverage_aware"; base_all.append(b); abl_all.append(f)
    fr = frozen[frozen.topology==topo]
    def st(g): return dict(meanPR=g.PR.mean(), minPR=g.PR.min(), pr90=(g.PR>=0.90).mean()*100, meanDB=g.DB.mean(),
        p95DB=np.percentile(g.DB,95), meanms=g.decision_ms.mean(), p95ms=np.percentile(g.decision_ms,95), maxms=g.decision_ms.max())
    sf, sb, sa = st(fr), st(b), st(f); fires=int(f.rule_fired.sum())
    comp.append(dict(topology=topo, rule_fires=fires,
        frozen_meanPR=round(sf["meanPR"],4), abl_meanPR=round(sa["meanPR"],4), frozen_minPR=round(sf["minPR"],4), abl_minPR=round(sa["minPR"],4),
        frozen_pr90=round(sf["pr90"],1), abl_pr90=round(sa["pr90"],1), frozen_meanDB=round(sf["meanDB"],4), abl_meanDB=round(sa["meanDB"],4),
        frozen_p95DB=round(sf["p95DB"],4), abl_p95DB=round(sa["p95DB"],4), frozen_meanms=round(sf["meanms"],1), abl_meanms=round(sa["meanms"],1),
        frozen_p95ms=round(sf["p95ms"],1), abl_p95ms=round(sa["p95ms"],1), frozen_maxms=round(sf["maxms"],1), abl_maxms=round(sa["maxms"],1),
        repro_ok=bool(abs(sb["meanPR"]-sf["meanPR"])<0.002)))
    print(f"[{topo}] fires={fires} meanPR {sf['meanPR']:.4f}->{sa['meanPR']:.4f} minPR {sf['minPR']:.4f}->{sa['minPR']:.4f} p95ms {sf['p95ms']:.0f}->{sa['p95ms']:.0f} maxms {sf['maxms']:.0f}->{sa['maxms']:.0f}", flush=True)
pd.concat(base_all).to_csv(SUB/"baseline_repro_per_cycle.csv", index=False)
pd.concat(abl_all).to_csv(SUB/"coverage_aware_per_cycle.csv", index=False)
cmp = pd.DataFrame(comp); cmp.to_csv(SUB/"comparison_summary.csv", index=False)
acts=["KEEP","K50","K100","K200","K300","K500","K800","RAISED"]; adf=pd.concat(abl_all)
ad=[]
for t in WIN:
    g=adf[adf.topology==t]; row=dict(Topology=t)
    for a in ["KEEP","K50","K100","K200","K300","K500","K800"]: row[a]=int((g.action==a).sum())
    row["rule_raised_cycles"]=int(g.rule_fired.sum()); row["median_K_used"]=int(g[g.K_used>0].K_used.median()) if (g.K_used>0).any() else 0
    ad.append(row)
pd.DataFrame(ad).to_csv(SUB/"action_distribution.csv", index=False)
print("\n=== COMPARISON (frozen Tier A vs coverage-aware K) ==="); print(cmp.to_string(index=False))
print("\nReplication ok (repro==frozen):", bool(cmp.repro_ok.all()))
print("saved to", SUB, "\nDONE")
