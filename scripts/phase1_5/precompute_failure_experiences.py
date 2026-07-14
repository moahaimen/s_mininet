#!/usr/bin/env python3
"""Precompute FAILURE-STATE training experiences for a method-consistent failure-aware fine-tune.
For sampled seen-topology training TMs under single/two/three-link failures, compute the 33-dim DDQN state
and the reward of EACH authorized action (KEEP,K50,K100,K200,K300,K500,K800) using the SAME reward as FIX1 and
the strict full-MCF failure optimum as the PR numerator (source-aggregated solver). No method change here.
Output: FAILAWARE_FINETUNE/failure_experiences.pkl  (list of dict: state, reward[7], topo, tm, scenario)."""
import sys, pickle, random, time
from collections import deque
import numpy as np, torch
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import scripts.phase1_5.run_fix1_failure_baseline_comparison as R
import scripts.phase1_5.run_fix1_completed_metrics as M
from scripts.phase1_5.run_final_iter2 import bottleneck_rank, target_pr
from scripts.phase1_5.strict_mcf_source_agg import solve_source_mcf
from te.disturbance import compute_disturbance
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import OUT_ROOT, clone_splits

random.seed(42); np.random.seed(42)
OUT = OUT_ROOT / "condition_compliant_k10_k50"; SUB = OUT / "FAILAWARE_FINETUNE"; SUB.mkdir(parents=True, exist_ok=True)
# reward weights (identical to FIX1 / iter2)
W_PR, W_MLU, W_DB, W_MS, W_K = 10.0, 5.0, 20.0, 0.003, 0.5
BONUS, TARGET_GATE, KEEP_GATE, KEEP_FLAT, MS_GATE = 10.0, 25.0, 25.0, 4.0, 25.0
def reward(PR, mex, DB, ms, k, nact, is_keep, tgt, feas):
    r = W_PR*PR - W_MLU*mex - W_DB*DB - W_MS*ms - W_K*(k/max(nact, 1))
    if PR >= tgt: r += BONUS
    else:
        if is_keep:
            if feas:
                r -= TARGET_GATE*(tgt-PR) + KEEP_FLAT
                if PR < 0.90: r -= KEEP_GATE*(0.90-PR)
        else: r -= TARGET_GATE*(tgt-PR)
    if ms > 500.0: r -= MS_GATE*((ms-500.0)/500.0)
    return r
def pr_of(o, m): return float(min(1.0, o/m)) if m > 0 else 0.0

# seen topos + training windows (use the training split, sample TMs); skip slow tiscali/sprintlink (topology-agnostic transfer)
POOL = {"abilene": (0, 1900), "geant": (0, 600), "cernet": (0, 180), "ebone": (0, 180), "germany50": (0, 260)}
SCEN = ["single_link_failure", "two_link_failure", "three_link_failure"]
N_SAMPLE = {"abilene": 40, "geant": 40, "cernet": 30, "ebone": 30, "germany50": 30}
ACT = list(M.ACTIONS.items())  # [(idx,(kind,K,_))...]

exps = []; t0 = time.perf_counter()
for topo, (lo, hi) in POOL.items():
    base = R.make_base_ctx(topo, lo, hi); ds = base.ds; tgt = target_pr(topo)
    tms = sorted(random.sample(range(lo, hi), min(N_SAMPLE[topo], hi-lo)))
    for scen in SCEN:
        caps = R.modify_caps(base.caps0, scen); pl = R.prune_pathlib(base.pl0, caps); fctx = R.make_failure_ctx(base, pl, caps)
        for t in tms:
            tm = np.asarray(ds.tm[t], float)
            strict = solve_source_mcf(tm, ds.od_pairs, ds.nodes, ds.edges, caps, time_limit_sec=60)["mlu"]
            if not np.isfinite(strict) or strict <= 0: continue
            neural = M.neural_scores(fctx, tm); ranked = bottleneck_rank(tm, fctx.ecmp, fctx.pl, fctx.caps, neural)
            accepted = clone_splits(fctx.ecmp)   # transient failure baseline = surviving ECMP
            keep_mlu = float(M.apply_routing(tm, accepted, fctx.pl, fctx.caps).mlu)
            raw, emlu = M.compute_raw_for_state(fctx, topo, -1, tm, None, ranked, neural)
            state = M.standardize(raw, keep_mlu, emlu).astype(np.float32)
            nact = int((tm > 0).sum())
            rr = np.zeros(7, np.float32); feas = False; opts = {}
            for a, (kind, K, _) in ACT:
                if kind == "keep": mlu, k, ms, DB = keep_mlu, 0, 0.5, 0.0
                else:
                    lp = R.run_selected_lp_failure(fctx, list(ranked[:K]), tm, accepted, 0.051, action_k=int(K))
                    mlu = float(lp["mlu"]); k = int(len([o for o in ranked[:K] if tm[o] > 0])); ms = 60.0
                    DB = min(float(compute_disturbance(accepted, lp["splits"], tm)), 0.10)
                    if pr_of(strict, mlu) >= tgt and ms < 500: feas = True
                opts[a] = (mlu, k, ms, DB)
            for a, (kind, K, _) in ACT:
                mlu, k, ms, DB = opts[a]; PR = pr_of(strict, mlu); mex = max(0.0, mlu/strict - 1.0)
                rr[a] = reward(PR, mex, DB, ms, k, nact, kind == "keep", tgt, feas)
            exps.append(dict(state=state, reward=rr, topo=topo, tm=int(t), scenario=scen))
    print(f"  {topo}: {sum(1 for e in exps if e['topo']==topo)} exps ({time.perf_counter()-t0:.0f}s)", flush=True)
pickle.dump(exps, open(SUB / "failure_experiences.pkl", "wb"))
print(f"[done] {len(exps)} failure experiences -> failure_experiences.pkl ({time.perf_counter()-t0:.0f}s)")
print("DONE")
