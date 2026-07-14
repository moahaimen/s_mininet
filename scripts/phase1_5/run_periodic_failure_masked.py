#!/usr/bin/env python3
"""PERIODIC-FAILURE evaluation (Spec 2) for the GATED first-cycle-opt model (Frozen DDQN + cycle-0 rule).
No retrain. Frozen Tier A untouched. Output -> PERIODIC_FAILURE_GATED/ (resumable, per-topo).

Setup per your spec:
  topo        test TMs   fail-every-N-TMs
  abilene     2016       12
  geant       672        4
  cernet      200        2
  sprintlink  200        2
  tiscali     200        2
  ebone       200        2
  germany50   288        4   (zero-shot)
  vtlwavenet  200        2   (zero-shot; computed live, only 0-40 was cached)

Failure model: at every Nth TM a transient single-link failure occurs (one link down for that TM, rotating
through the highest-capacity links); paths using it are pruned; if the carried routing used it, the baseline is
reset to surviving-path ECMP; the gated DDQN decides; PR = path-LP optimum on the (possibly failed) topology /
our MLU. Non-failure TMs run normally with carry-forward. Gated rule: at the FIRST TM only, if the DDQN picks an
optimize action with K>=300, use db_budget=1.0 (cold-start full optimization)."""
import sys, time, json, pickle, os
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import torch
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, apply_routing, clone_splits, set_seed
from te.lp_solver import solve_selected_path_lp_dbbudget, solve_all_od_path_lp
from te.disturbance import compute_disturbance
from te.baselines import ecmp_splits
from te.paths import PathLibrary
import scripts.phase1_5.agnostic_lib as A
from scripts.phase1_5.bottleneck_lib import ACTIONS, ANAME
from scripts.phase1_5.run_final_iter2 import kp_for, build_mixed, pad_to_lib, bottleneck_rank, pr_of, GNN_MS
from scripts.phase1_5.strict_mcf_source_agg import solve_source_mcf
KEEP_MASK = os.environ.get('KEEP_MASK','0')=='1'

set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"
SUB = OUT / "PERIODIC_FAILURE_MASK"; SUB.mkdir(parents=True, exist_ok=True)
MODE = "masked" if KEEP_MASK else "unmasked"
AGN = OUT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN" / "_cache"; KP = OUT / "FINAL_LEARNED_4OF5_KPATH4_DDQN" / "_cache"
SC = json.load(open(AGN / "scaler.json")); MEAN = np.array(SC["mean"], np.float32); STD = np.array(SC["std"], np.float32)
P = pickle.load(open(OUT / "_prepass.pkl", "rb"))
JOBS = {"abilene":(2016,4032,12),"geant":(672,1344,4),"cernet":(200,400,2),"sprintlink":(200,400,2),
        "tiscali":(200,400,2),"ebone":(200,400,2),"germany50":(0,288,4),"vtlwavenet2011":(0,200,2)}
dim = len(A.AGN_FEAT_NAMES)
net = A.QNet(dim, 7); net.load_state_dict(torch.load(OUT / "FULLDATA_GATED_PRESERVED_FIX1" / "fulldata_gated_model.pt", map_location="cpu")["state_dict"]); net.eval()
teacher = A.QNet(dim, 7); teacher.load_state_dict(torch.load(OUT / "FROZEN_FINAL_LEARNED_RUNTIME_SAFE_ITER2" / "final_learned_4of5_iter2_model.pt", map_location="cpu")["state_dict"]); teacher.eval()
KEEP_IDX = [i for i,v in ACTIONS.items() if v[0]=="keep"][0]

def prune(pl, caps):
    nps,eps,ips,cps = [],[],[],[]
    for i in range(len(pl.edge_idx_paths_by_od)):
        idx=[j for j,ep in enumerate(pl.edge_idx_paths_by_od[i]) if all(float(caps[e])>0 for e in ep)]
        nps.append([pl.node_paths_by_od[i][j] for j in idx]); eps.append([pl.edge_paths_by_od[i][j] for j in idx])
        ips.append([pl.edge_idx_paths_by_od[i][j] for j in idx]); cps.append([pl.costs_by_od[i][j] for j in idx])
    return PathLibrary(od_pairs=pl.od_pairs, node_paths_by_od=nps, edge_paths_by_od=eps, edge_idx_paths_by_od=ips, costs_by_od=cps)
def state_vec(topo, t, ds, pl, ecmp, caps, struct, scr, util, accepted, prev_tm):
    tm = np.asarray(ds.tm[t], float); act=[od for od in range(len(tm)) if tm[od]>0]
    av = scr[act] if len(act) else np.zeros(1); ranked = bottleneck_rank(tm, ecmp, pl, caps, scr)
    keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu); emlu = float(apply_routing(tm, ecmp, pl, caps).mlu)
    chg = 0.0 if prev_tm is None else float(np.abs(tm-prev_tm).sum()/(np.abs(prev_tm).sum()+1e-9))
    dd = dict(ranked={t: np.array(ranked, np.int32)}, tm_cache=ds.tm, num_nodes=len(ds.nodes))
    dpre = dict(tmstat={t:(float(np.log1p(tm.sum())), float(tm.max()/(tm.sum()+1e-9)), min(chg,3.0), len(act))},
                sstat={t:(float(av.mean()), float(np.quantile(av,.95)), float(av.max()))}, emlu={t:emlu})
    raw = A.raw_static(topo, t, dd, dpre, pl, ecmp, caps, scr, util, struct)
    return A.standardize(A.raw_to_vec(raw, keep_mlu, emlu), MEAN, STD), ranked, keep_mlu

def run(topo, lo, hi, N):
    out_f = SUB / f"periodic_{MODE}_{topo}.csv"
    if out_f.exists(): print(f"[skip] {topo}", flush=True); return pd.read_csv(out_f)
    env = _make_envs([topo], {topo:(lo,hi)}, gnn, hi-lo, 30)[0]; ctx = env.ctx
    d_pp = P[(topo,lo,hi)] if (topo,lo,hi) in P else None
    # use PREPASS caps so apply_routing MLU is on the SAME scale as the prepass opt numerator (frozen-eval convention)
    ds, pl0 = ctx["ds"], ctx["pl"]; caps0 = np.asarray(ctx["caps"], float)   # env caps + live opt -> consistent scale & dims (run_failure_current convention)
    struct0 = A.struct_feats(ds)
    cap_order = list(np.argsort(caps0)[::-1]); top = [int(i) for i in cap_order[:min(20, int((caps0>0).sum()))]]  # rotate failures over top-cap links
    normal_opt = None
    pl = pl0   # path library is FIXED; a failure is modeled as the failed link's capacity going to ~0 + rerouting off it
    epaths = pl0.edge_idx_paths_by_od
    def reroute_off(splits, fl):
        out = []
        for od in range(len(splits)):
            s = np.asarray(splits[od], float).copy()
            for p in range(min(len(s), len(epaths[od]))):
                if fl in epaths[od][p]: s[p] = 0.0
            tot = s.sum(); out.append(s/tot if tot > 0 else s)
        return out
    ecmp0 = ctx["ecmp"]
    accepted = clone_splits(ecmp0); prev_tm = None; rows = []; fev = 0; t0 = time.perf_counter()
    for i, t in enumerate(range(lo, hi)):
        tm = np.asarray(ds.tm[t], float)
        is_fail = (i > 0 and i % N == 0)
        if is_fail:
            fl = top[fev % len(top)]; fev += 1
            caps = caps0.copy(); caps[fl] = max(caps0[fl]*1e-9, 1e-9)        # link down (tiny cap, no div0)
            caps_strict = caps0.copy(); caps_strict[fl] = 0.0               # strict full-MCF: downed link is GONE (cap=0), avoids near-zero-cap ill-conditioning
            failed_lbl = f"{ds.edges[fl][0]}->{ds.edges[fl][1]}" if hasattr(ds,'edges') else str(fl)
            ecmp = reroute_off(ecmp0, fl); accepted = reroute_off(accepted, fl)   # reroute baseline + carried routing OFF the failed link (carry-forward preserved)
            opt_mlu = float(solve_all_od_path_lp(tm, pl0, caps, time_limit_sec=60).mlu)
            act_live = [od for od in range(len(tm)) if tm[od] > 0]
            disc = len([od for od in act_live if all((fl in ep) for ep in epaths[od])])   # OD whose ALL paths use the failed link
        else:
            caps = caps0; caps_strict = caps0; ecmp = ecmp0; failed_lbl = ""; disc = 0
            opt_mlu = float(normal_opt[t]) if normal_opt is not None else float(solve_all_od_path_lp(tm, pl0, caps, time_limit_sec=60).mlu)
        scr,_,_ = gnn.score(dataset=ds, tm_vector=tm, path_library=pl0, capacities=caps, ecmp_base=ecmp); scr=np.asarray(scr,float).ravel()
        util = apply_routing(tm, ecmp, pl0, caps).utilization
        s, ranked, keep_mlu = state_vec(topo, t, ds, pl0, ecmp, caps, struct0, scr, util, accepted, prev_tm)
        with torch.no_grad():
            sv=torch.tensor(s).unsqueeze(0); q = teacher(sv) if i==0 else net(sv); a = int(q.argmax())   # FIX1: first-TM frozen-action
        if i == 0 and a == KEEP_IDX:
            # Cycle 0 cannot validly reuse a previously accepted routing; force best non-KEEP action.
            qn = q.clone(); qn[..., KEEP_IDX] = -1e9; a = int(qn.argmax())
        if KEEP_MASK and is_fail and a == KEEP_IDX:
            # FAILURE-MODE KEEP ACTION MASK: KEEP invalid while a scheduled link-down failure is active;
            # DDQN still argmax-selects among {K50,K100,K200,K300,K500,K800}. No force-K800, no manual replacement.
            qn = q.clone(); qn[..., KEEP_IDX] = float('-inf'); a = int(qn.argmax())
        kind, K, _ = ACTIONS[a]; dbb = 0.051
        if i == 0 and kind != "keep" and K >= 300: dbb = 1.0        # gated first-cycle rule after cycle-0 KEEP guard
        if is_fail: dbb = 1.0                                        # failure: allow full reroute (tight DB would be infeasible)
        if kind == "keep":
            mlu = keep_mlu; ms = GNN_MS[topo]+0.5; k = 0; sp = accepted
        else:
            sel = list(ranked[:K])
            if is_fail: sel = [o for o in sel if not all((fl in ep) for ep in epaths[o])]   # drop disconnected ODs (unroutable)
            plm = pl0; s0=time.perf_counter()                        # full path library (robust: no kp-truncation under failure)
            lp = solve_selected_path_lp_dbbudget(tm_vector=tm, selected_ods=sel, base_splits=ecmp, path_library=plm,
                capacities=caps, prev_splits=accepted, db_budget=dbb, db_weight=1e-6, time_limit_sec=60)
            sp = pad_to_lib(lp.splits, pl0); mlu = float(apply_routing(tm, sp, pl0, caps).mlu); ms=(time.perf_counter()-s0)*1000+GNN_MS[topo]; k=int(len([o for o in sel if tm[o]>0]))
        _sres = solve_source_mcf(tm, ds.od_pairs, ds.nodes, ds.edges, caps_strict, time_limit_sec=300)   # strict full-MCF on EXACT per-cycle state (failed link removed)
        strict = _sres["mlu"]
        if not np.isfinite(strict):
            raise RuntimeError(f"strict full-MCF non-optimal for {topo} t={t} is_fail={int(is_fail)}: status={_sres['status']} (no path-LP fallback allowed)")
        pr_strict = float(min(1.0, strict/mlu)) if (np.isfinite(strict) and mlu>0) else 0.0
        rows.append(dict(topology=topo, tm_index=t, is_failure=int(is_fail), failed_link=failed_lbl, action=ANAME[a],
            selected_K=k, PR_pathlp=pr_of(opt_mlu, mlu), PR_strict=pr_strict, strict_mlu=strict, MLU=mlu,
            DB=float(compute_disturbance(accepted, sp, tm)), decision_ms=round(ms,1), disconnected_ODs=disc)); accepted = sp; prev_tm = tm
        if i % 200 == 0: print(f"  [{topo}] {i}/{hi-lo} ({time.perf_counter()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); df.to_csv(out_f, index=False); print(f"[done] {topo} {time.perf_counter()-t0:.0f}s  failures={fev}", flush=True)
    return df

if __name__ == "__main__":
    print("PERIODIC-FAILURE eval (gated first-cycle model, frozen DDQN, no retrain)\n", flush=True)
    only = sys.argv[1].split(",") if len(sys.argv)>1 else list(JOBS)
    allrows = []
    for topo in only:
        lo,hi,N = JOBS[topo]; allrows.append(run(topo, lo, hi, N))
    pc = pd.concat(allrows, ignore_index=True); pc.to_csv(SUB/"periodic_failure_all.csv", index=False)
    D={'abilene':'Abilene','geant':'GEANT','cernet':'CERNET','sprintlink':'Sprintlink','tiscali':'Tiscali','ebone':'Ebone','germany50':'Germany50','vtlwavenet2011':'VtlWavenet'}
    print("\n=== PERIODIC-FAILURE RESULTS (full test stream, fail every N TMs) ===")
    print(f"{'Topology':11s}{'N_fail/N':>9s}{'#fail':>6s}{'meanPR':>8s}{'minPR':>8s}{'PR>=.90':>8s}{'meanPR_fail':>12s}{'meanDB':>8s}{'mean_ms':>8s}{'p95_ms':>7s}{'maxDisc':>8s}")
    rows=[]
    for topo,(lo,hi,N) in JOBS.items():
        g=pc[pc.topology==topo]; gf=g[g.is_failure==1]
        r=dict(topology=topo, period_N=N, num_failures=int(g.is_failure.sum()), mean_PR=round(g.PR.mean(),4), min_PR=round(g.PR.min(),4),
            pr90=round((g.PR>=0.90).mean()*100,1), mean_PR_failcycles=round(gf.PR.mean(),4) if len(gf) else None,
            mean_DB=round(g.DB.mean(),4), mean_ms=round(g.decision_ms.mean(),1), p95_ms=round(float(np.percentile(g.decision_ms,95)),1), max_disc=int(g.disconnected_ODs.max()))
        rows.append(r)
        print(f"{D[topo]:11s}{('1/'+str(N)):>9s}{r['num_failures']:>6d}{r['mean_PR']:>8.4f}{r['min_PR']:>8.4f}{r['pr90']:>7.1f}%{(r['mean_PR_failcycles'] or 0):>12.4f}{r['mean_DB']:>8.4f}{r['mean_ms']:>8.1f}{r['p95_ms']:>7.1f}{r['max_disc']:>8d}")
    pd.DataFrame(rows).to_csv(SUB/"periodic_failure_summary.csv", index=False)
    print("\nDONE")
