#!/usr/bin/env python3
"""GLOBAL failure-mode rule: never KEEP under a detected failure (KEEP reuses stale routing off a just-failed
link -> objectively wrong). When the DDQN would KEEP during a failure scenario, force the optimize action at the
max budget (K800) instead. Applies to ALL topologies; only Abilene ever KEEPs under failure, so only Abilene changes.
Re-evaluates each failure cycle two ways (original policy vs never-KEEP), PR = min(1, strict/achieved).
PRODUCES NUMBERS ONLY; does not edit the report or the deployed normal method."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import scripts.phase1_5.run_fix1_failure_baseline_comparison as R
import scripts.phase1_5.run_fix1_completed_metrics as M

RC = "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50"
strict = {}
for r in pd.read_csv(f"{RC}/STRICT_FAILURE_FULL_MCF_PR/_partial/abilene.progress.csv").itertuples():
    strict[(r.scenario, int(r.tm_id))] = float(r.strict_full_mcf_MLU)

topo = "abilene"; lo, hi = R.TOPOS[topo]
base = R.make_base_ctx(topo, lo, hi); ds = base.ds

def eval_cycle(fctx, cycle_idx, tm, accepted, prev_tm, never_keep):
    ranked, neural = R.ranked_list_for_mode(fctx, topo, tm, prev_tm, "cache_final")
    raw, emlu = M.compute_raw_for_state(fctx, topo, -1, tm, prev_tm, ranked, neural)
    keep_mlu = float(M.apply_routing(tm, accepted, fctx.pl, fctx.caps).mlu)
    state = M.standardize(raw, keep_mlu, emlu)
    a = M.action_from_final(fctx, cycle_idx, state, teacher_cycle0=True)
    kind, k, _ = M.ACTIONS[a]; act = M.ANAME[a]
    if kind == "keep" and not never_keep:
        return keep_mlu, accepted, "KEEP"
    if kind == "keep" and never_keep:
        k = 800; act = "K800(forced)"                       # never-KEEP under failure -> max optimize
    sel = list(ranked[:k]); db = 1.0 if (cycle_idx == 0 and int(k) >= 300) else 0.051
    lp = R.run_selected_lp_failure(fctx, sel, tm, accepted, db, action_k=int(k))
    return float(lp["mlu"]), lp["splits"], act

def run_pass(never_keep):
    out = {}
    for scenario in R.SCENARIOS:
        caps = R.modify_caps(base.caps0, scenario); pl = R.prune_pathlib(base.pl0, caps)
        fctx = R.make_failure_ctx(base, pl, caps)
        accepted = R.clone_splits(fctx.ecmp); prev_tm = None; prs = []; acts = []
        for cycle_idx, t in enumerate(range(lo, hi)):
            tm = np.asarray(ds.tm[t], float) * R.spike_factor_for(scenario)
            mlu, sp, act = eval_cycle(fctx, cycle_idx, tm, accepted, prev_tm, never_keep)
            s = strict[(scenario, int(t))]; prs.append(min(1.0, s / mlu) if mlu > 0 else 0.0); acts.append(act)
            accepted = sp; prev_tm = tm
        out[scenario] = (np.array(prs), acts)
    return out

print("Re-evaluating Abilene failure with never-KEEP-under-failure rule...\n", flush=True)
old = run_pass(False); new = run_pass(True)
print(f"{'Scenario':24s}{'oldMeanPR':>10s}{'newMeanPR':>10s}{'old>=90':>8s}{'new>=90':>8s}{'KEEP->opt':>10s}")
for sc in R.SCENARIOS:
    po, ao = old[sc]; pn, an = new[sc]
    forced = sum(1 for x in an if "forced" in x)
    print(f"{sc:24s}{po.mean()*100:>9.3f}%{pn.mean()*100:>9.3f}%{(po>=.9).mean()*100:>7.1f}%{(pn>=.9).mean()*100:>7.1f}%{forced:>10d}", flush=True)
# overall
po_all = np.concatenate([old[sc][0] for sc in R.SCENARIOS]); pn_all = np.concatenate([new[sc][0] for sc in R.SCENARIOS])
print(f"\n{'ABILENE OVERALL':24s}{po_all.mean()*100:>9.3f}%{pn_all.mean()*100:>9.3f}%{(po_all>=.9).mean()*100:>7.1f}%{(pn_all>=.9).mean()*100:>7.1f}%")
print("DONE")
