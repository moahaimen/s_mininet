#!/usr/bin/env python3
"""FAILURE-MODE KEEP ACTION MASK on the 20-TM stress tests (Final RG-GNN-LPD only; baselines unchanged).
Rule (action-validity, NOT a force-K800 override):
  if active_link_down_failure:  valid = {K50,K100,K200,K300,K500,K800}   (KEEP removed)
  else:                          valid = {KEEP,K50,...,K800}
  selected_action = argmax Q(action) over valid_actions   (DDQN still selects)
Active link-down scenarios: single_link_failure, two_link_failure, three_link_failure, mixed_spike_failure.
NOT masked: spike (no link down), capacity_degradation_50 (no link down), normal.
Reruns both passes (unmasked baseline reproduce + masked) and reports old vs new. No report edit."""
import sys, numpy as np, pandas as pd, torch, glob
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import scripts.phase1_5.run_fix1_failure_baseline_comparison as R
import scripts.phase1_5.run_fix1_completed_metrics as M
from scripts.phase1_5.run_final_iter2 import bottleneck_rank
from te.disturbance import compute_disturbance
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import OUT_ROOT, clone_splits

OUT = OUT_ROOT / "condition_compliant_k10_k50"; SUB = OUT / "KEEPMASK_FAILURE"; SUB.mkdir(exist_ok=True)
KEEP_IDX = [a for a, (k, _, _) in M.ACTIONS.items() if k == "keep"][0]
ACTIVE_LINKDOWN = {"single_link_failure", "two_link_failure", "three_link_failure", "mixed_spike_failure"}
strict = {}
for f in glob.glob(str(OUT / "STRICT_FAILURE_FULL_MCF_PR" / "_partial" / "*.progress.csv")):
    for r in pd.read_csv(f).itertuples(): strict[(r.topology, r.scenario, int(r.tm_id))] = float(r.strict_full_mcf_MLU)

def select_action(q_vec, active_link_down):
    q = q_vec.copy()
    if active_link_down: q[KEEP_IDX] = -np.inf     # KEEP removed from valid action set
    return int(np.argmax(q))                        # DDQN still selects argmax over remaining valid actions

def eval_cycle(fctx, topo, i, tm, accepted, prev, mask):
    neural = M.neural_scores(fctx, tm); ranked = bottleneck_rank(tm, fctx.ecmp, fctx.pl, fctx.caps, neural)
    raw, emlu = M.compute_raw_for_state(fctx, topo, -1, tm, prev, ranked, neural)
    keep_mlu = float(M.apply_routing(tm, accepted, fctx.pl, fctx.caps).mlu)
    s = M.standardize(raw, keep_mlu, emlu); sv = torch.tensor(s).unsqueeze(0)
    with torch.no_grad(): q = (M.TEACHER(sv) if i == 0 else M.FINAL_NET(sv)).numpy().ravel()
    a = select_action(q, mask); kind, K, _ = M.ACTIONS[a]
    if kind == "keep": mlu, sp, ms = keep_mlu, accepted, float(M.GNN_MS[topo]) + 0.5
    else:
        db = 1.0 if (i == 0 and int(K) >= 300) else 0.051
        lp = R.run_selected_lp_failure(fctx, list(ranked[:K]), tm, accepted, db, action_k=int(K))
        mlu, sp = float(lp["mlu"]), lp["splits"]; ms = float(lp["decision_lp_ms"]) + float(M.GNN_MS[topo])
    return mlu, sp, M.ANAME[a], float(compute_disturbance(accepted, sp, tm)), ms

rows = []
for topo, (lo, hi) in R.TOPOS.items():
    base = R.make_base_ctx(topo, lo, hi); ds = base.ds
    for scen in R.SCENARIOS:
        active = scen in ACTIVE_LINKDOWN
        caps = R.modify_caps(base.caps0, scen); pl = R.prune_pathlib(base.pl0, caps); fctx = R.make_failure_ctx(base, pl, caps)
        for mask_on in ([False, True] if active else [False]):   # unaffected scenarios: single pass
            accepted = clone_splits(fctx.ecmp); prev = None
            for i, t in enumerate(range(lo, hi)):
                tm = np.asarray(ds.tm[t], float) * R.spike_factor_for(scen)
                mlu, sp, act, DB, ms = eval_cycle(fctx, topo, i, tm, accepted, prev, mask_on and active)
                st = strict[(topo, scen, int(t))]; pr = min(1.0, st / mlu) if mlu > 0 else 0.0
                rows.append(dict(topology=topo, scenario=scen, tm_index=int(t), mask=mask_on, active_linkdown=active,
                                 action=act, PR=pr, DB=DB, ms=ms))
                accepted = sp; prev = tm
    print(f"  {topo} done", flush=True)
df = pd.DataFrame(rows); df.to_csv(SUB / "keepmask_stress_per_cycle.csv", index=False)

def agg(g): return pd.Series(dict(meanPR=g.PR.mean()*100, pr90=(g.PR>=.9).mean()*100, pr95=(g.PR>=.95).mean()*100,
                                  meanDB=g.DB.mean()*100, p95DB=np.percentile(g.DB,95)*100, meanms=g.ms.mean(), p95ms=np.percentile(g.ms,95)))
print("\n=== ABILENE old(unmasked) vs new(masked) ===")
for scen in ["single_link_failure","two_link_failure","three_link_failure","mixed_spike_failure"]:
    o = agg(df[(df.topology=="abilene")&(df.scenario==scen)&(df.mask==False)])
    n = agg(df[(df.topology=="abilene")&(df.scenario==scen)&(df.mask==True)])
    print(f"  {scen:22s} PR {o.meanPR:.3f}->{n.meanPR:.3f}  >=90 {o.pr90:.1f}->{n.pr90:.1f}  >=95 {o.pr95:.1f}->{n.pr95:.1f}  DB {o.meanDB:.3f}->{n.meanDB:.3f}  ms {o.meanms:.0f}->{n.meanms:.0f}")
# KEEP-under-active-failure counts
km = df[(df.mask==True)&(df.active_linkdown==True)]
print(f"\nKEEP under active link-down (masked pass): {int((km.action=='KEEP').sum())} (must be 0)")
changed = 0
for (topo,scen), g in df[df.active_linkdown==True].groupby(["topology","scenario"]):
    o=g[g.mask==False].set_index("tm_index").action; n=g[g.mask==True].set_index("tm_index").action
    changed += int((o!=n).sum())
print(f"cycles whose action changed due to KEEP mask: {changed}")
# replacement action distribution (where old==KEEP under active failure)
oldkeep = df[(df.mask==False)&(df.active_linkdown==True)&(df.action=="KEEP")][["topology","scenario","tm_index"]]
repl = df[(df.mask==True)&(df.active_linkdown==True)].merge(oldkeep, on=["topology","scenario","tm_index"])
print("replacement action distribution (former KEEP cycles):", repl.action.value_counts().to_dict())
print("DONE")
