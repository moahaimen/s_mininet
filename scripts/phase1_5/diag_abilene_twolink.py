#!/usr/bin/env python3
"""DIAGNOSTIC ONLY (no method change, no report edit): cycle-by-cycle investigation of Abilene two-link failure.
Replays the frozen RG-GNN-LPD trajectory (DDQN selects the action), records per-cycle state/Q-values/scores,
then for every sub-0.90 cycle runs an OFFLINE COUNTERFACTUAL ACTION SWEEP over all authorized actions
(KEEP,K50,K100,K200,K300,K500,K800) from the SAME frozen carried state, and classifies the root cause:
  A action-selection error (some authorized action reaches PR>=0.90)
  B path-coverage-limited (all authorized actions < 0.90; candidate-path exhaustion)
  C DB-constraint (best action reaches >=0.90 only with a larger disturbance budget)
  D other."""
import sys
from collections import deque
import numpy as np, pandas as pd, torch, time
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import scripts.phase1_5.run_fix1_failure_baseline_comparison as R
import scripts.phase1_5.run_fix1_completed_metrics as M
from scripts.phase1_5.run_final_iter2 import bottleneck_rank
from te.disturbance import compute_disturbance

RC = "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50"
strict = {int(r.tm_id): float(r.strict_full_mcf_MLU)
          for r in pd.read_csv(f"{RC}/STRICT_FAILURE_FULL_MCF_PR/_partial/abilene.progress.csv").itertuples()
          if r.scenario == "two_link_failure"}
topo = "abilene"; lo, hi = R.TOPOS[topo]; SCEN = "two_link_failure"
base = R.make_base_ctx(topo, lo, hi); ds = base.ds
caps = R.modify_caps(base.caps0, SCEN); pl = R.prune_pathlib(base.pl0, caps); fctx = R.make_failure_ctx(base, pl, caps)
nodes = list(ds.nodes); edges = list(ds.edges); odp = list(ds.od_pairs); EPS = 1e-9
ACTS = list(M.ACTIONS.values()); ANAME = M.ANAME

def reach(src, dst, capv):
    if src == dst: return True
    adj = {n: [] for n in nodes}
    for (u, v), c in zip(edges, capv):
        if c > EPS: adj[u].append(v)
    seen = {src}; q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v == dst: return True
            if v not in seen: seen.add(v); q.append(v)
    return False

def exhausted_connected(tm, capv):
    n = 0
    for i in range(len(odp)):
        if tm[i] > 0 and len(pl.edge_idx_paths_by_od[i]) == 0 and reach(odp[i][0], odp[i][1], capv): n += 1
    return n

def lp_for(K, tm, accepted, ranked, db):
    sel = list(ranked[:K])
    lp = R.run_selected_lp_failure(fctx, sel, tm, accepted, db, action_k=int(K))
    return float(lp["mlu"]), lp["splits"]

# ---- replay trajectory, record per cycle ----
accepted = R.clone_splits(fctx.ecmp); prev_tm = None; rec = []
for i, t in enumerate(range(lo, hi)):
    tm = np.asarray(ds.tm[t], float) * R.spike_factor_for(SCEN)
    neural = M.neural_scores(fctx, tm); ranked = bottleneck_rank(tm, fctx.ecmp, fctx.pl, fctx.caps, neural)
    raw, emlu = M.compute_raw_for_state(fctx, topo, -1, tm, prev_tm, ranked, neural)
    keep_mlu = float(M.apply_routing(tm, accepted, fctx.pl, fctx.caps).mlu)
    state = M.standardize(raw, keep_mlu, emlu); sv = torch.tensor(state).unsqueeze(0)
    with torch.no_grad(): q = (M.TEACHER(sv) if i == 0 else M.FINAL_NET(sv)).numpy().ravel()
    a = int(q.argmax()); kind, K, _ = ACTS[a]
    db = 1.0 if (i == 0 and int(K) >= 300) else 0.051
    if kind == "keep": mlu, sp = keep_mlu, accepted
    else: mlu, sp = lp_for(K, tm, accepted, ranked, db)
    s = strict[int(t)]; pr = min(1.0, s / mlu) if mlu > 0 else 0.0
    qs = q.copy(); qs[a] = -1e9; alt = int(qs.argmax())
    rec.append(dict(i=i, tm=int(t), action=ANAME[a], K=int(K), achieved=mlu, strict=s, PR=pr,
                    passed=pr >= 0.90, q_sel=float(q[a]), q_alt=float(q[alt]), alt=ANAME[alt],
                    DB=float(compute_disturbance(accepted, sp, tm)), exh=exhausted_connected(tm, fctx.caps),
                    neural_mean=float(np.mean(neural)), neural_p95=float(np.quantile(neural, .95)),
                    accepted=R.clone_splits(accepted), ranked=ranked, tmv=tm))
    accepted = sp; prev_tm = tm

df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("accepted", "ranked", "tmv")} for r in rec])
print("=== Abilene two-link: all 20 cycles ===")
print(df[["tm", "action", "K", "achieved", "strict", "PR", "passed", "q_sel", "q_alt", "alt", "DB", "exh"]].to_string(index=False))
low = [r for r in rec if r["PR"] < 0.90]
print(f"\nSub-0.90 cycles: {len(low)} of 20   (mean PR {df.PR.mean()*100:.3f}%, PR>=0.90 {(df.PR>=0.9).mean()*100:.1f}%)")

# ---- counterfactual action sweep for sub-0.90 cycles ----
print("\n=== counterfactual action sweep (from same frozen carried state) ===")
cls = {}
for r in low:
    tm, accepted, ranked, s = r["tmv"], r["accepted"], r["ranked"], r["strict"]
    keep_mlu = float(M.apply_routing(tm, accepted, fctx.pl, fctx.caps).mlu)
    sweep = {}
    for a, (kind, K, _) in M.ACTIONS.items():
        if kind == "keep": mlu = keep_mlu; sp = accepted
        else: mlu, sp = lp_for(K, tm, accepted, ranked, 0.051)
        sweep[ANAME[a]] = (mlu, min(1.0, s / mlu) if mlu > 0 else 0.0, float(compute_disturbance(accepted, sp, tm)))
    best = max(sweep.items(), key=lambda kv: kv[1][1])
    # DB check: does best action reach >=0.90 only with larger DB?
    db_help = ""
    if best[1][1] < 0.90 and best[0] != "KEEP":
        Kb = M.ACTIONS[[k for k, v in M.ACTIONS.items() if ANAME[k] == best[0]][0]][1]
        mlu_hi, sp_hi = lp_for(Kb, tm, accepted, ranked, 1.0)
        if min(1.0, s / mlu_hi) >= 0.90: db_help = f" (db=1.0 -> PR={min(1.0,s/mlu_hi):.3f})"
    klass = "A_action_error" if best[1][1] >= 0.90 else ("C_DB" if db_help else "B_path_coverage")
    cls[r["tm"]] = klass
    print(f"tm{r['tm']} DDQN={r['action']}(PR{r['PR']:.3f}) exh={r['exh']} best={best[0]}(PR{best[1][1]:.3f}){db_help} -> {klass}")
    print("   " + "  ".join(f"{a}:PR{v[1]:.3f}/DB{v[2]:.3f}" for a, v in sweep.items()))

# ---- summary ----
from collections import Counter
c = Counter(cls.values())
print(f"\n=== CLASSIFICATION SUMMARY (Abilene two-link, {len(low)} sub-0.90 cycles) ===")
print(f"  A action-selection error (a better authorized action reaches >=0.90): {c.get('A_action_error',0)}")
print(f"  B path-coverage-limited (all authorized actions < 0.90):             {c.get('B_path_coverage',0)}")
print(f"  C DB-constraint (best action needs larger DB):                        {c.get('C_DB',0)}")
def oracle_pr(r):
    tm, accepted, ranked, s = r["tmv"], r["accepted"], r["ranked"], r["strict"]
    km = float(M.apply_routing(tm, accepted, fctx.pl, fctx.caps).mlu)
    best = 0.0
    for a, (kind, K, _) in M.ACTIONS.items():
        mlu = km if kind == "keep" else lp_for(K, tm, accepted, ranked, 0.051)[0]
        best = max(best, min(1.0, s / mlu) if mlu > 0 else 0.0)
    return best
oracle = [oracle_pr(r) for r in rec]
print(f"  best-achievable two-link mean PR (oracle over authorized actions, db=0.051): {np.mean(oracle)*100:.3f}%")
print(f"  best-achievable two-link PR>=0.90 rate (oracle):                            {np.mean(np.array(oracle)>=0.9)*100:.1f}%")
print(f"  low cycles where NO authorized action reaches 0.90 (path-coverage floor):   {int(np.sum([o<0.90 for o in oracle]))}")
print("DONE")
