#!/usr/bin/env python3
"""Evaluate the failure-aware fine-tuned model: (1) Abilene single/two-link failure recovery,
(2) normal 8-topology regression check vs FIX1. DDQN selects (no override). No report edit."""
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import scripts.phase1_5.run_fix1_failure_baseline_comparison as R
import scripts.phase1_5.run_fix1_completed_metrics as M
from scripts.phase1_5.run_final_iter2 import bottleneck_rank
import scripts.phase1_5.agnostic_lib as A
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import OUT_ROOT, clone_splits
OUT = OUT_ROOT / "condition_compliant_k10_k50"
FA = torch.load(OUT / "FAILAWARE_FINETUNE" / "failaware_model.pt", map_location="cpu")["state_dict"]
NET = A.QNet(len(A.AGN_FEAT_NAMES), 7); NET.load_state_dict(FA); NET.eval()

# ---- (1) Abilene failure recovery (failaware net selects; cycle-0 keeps teacher) ----
topo = "abilene"; lo, hi = R.TOPOS[topo]; base = R.make_base_ctx(topo, lo, hi); ds = base.ds
strict = {}
for r in pd.read_csv(f"{OUT}/STRICT_FAILURE_FULL_MCF_PR/_partial/abilene.progress.csv").itertuples():
    strict[(r.scenario, int(r.tm_id))] = float(r.strict_full_mcf_MLU)
print("=== Abilene failure with FAILAWARE model (DDQN selects) ===")
for SCEN in ["single_link_failure", "two_link_failure"]:
    caps = R.modify_caps(base.caps0, SCEN); pl = R.prune_pathlib(base.pl0, caps); fctx = R.make_failure_ctx(base, pl, caps)
    accepted = clone_splits(fctx.ecmp); prev = None; prs = []; acts = []
    for i, t in enumerate(range(lo, hi)):
        tm = np.asarray(ds.tm[t], float) * R.spike_factor_for(SCEN)
        neural = M.neural_scores(fctx, tm); ranked = bottleneck_rank(tm, fctx.ecmp, fctx.pl, fctx.caps, neural)
        raw, emlu = M.compute_raw_for_state(fctx, topo, -1, tm, prev, ranked, neural)
        keep_mlu = float(M.apply_routing(tm, accepted, fctx.pl, fctx.caps).mlu)
        s = M.standardize(raw, keep_mlu, emlu); sv = torch.tensor(s).unsqueeze(0)
        with torch.no_grad(): a = int((M.TEACHER(sv) if i == 0 else NET(sv)).argmax())
        kind, K, _ = M.ACTIONS[a]
        if kind == "keep": mlu, sp = keep_mlu, accepted
        else:
            db = 1.0 if (i == 0 and int(K) >= 300) else 0.051
            lp = R.run_selected_lp_failure(fctx, list(ranked[:K]), tm, accepted, db, action_k=int(K)); mlu, sp = float(lp["mlu"]), lp["splits"]
        st = strict[(SCEN, int(t))]; prs.append(min(1.0, st / mlu) if mlu > 0 else 0.0); acts.append(M.ANAME[a])
        accepted = sp; prev = tm
    prs = np.array(prs)
    print(f"  {SCEN:20s} mean PR={prs.mean()*100:.3f}%  PR>=0.90={(prs>=.9).mean()*100:.1f}%  KEEP count={acts.count('KEEP')}")

# ---- (2) normal regression check: failaware vs FIX1 on the FIX1 normal per-cycle states ----
print("\n=== Normal regression check (failaware vs FIX1 argmax on FIX1 eval states) ===")
# reuse FIX1 eval per-cycle actions as the reference trajectory; compare argmax on the same states is not stored,
# so recompute normal eval with the failaware net via the FIX1 gated_eval harness.
import importlib.util
spec = importlib.util.spec_from_file_location("fix1", "/Users/moahaimentalib/Desktop/f_flex_network_code_clean/scripts/phase1_5/run_fulldata_gated_fix1.py")
fix1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(fix1)
teacher = A.QNet(len(A.AGN_FEAT_NAMES), 7); teacher.load_state_dict(torch.load(OUT / "FROZEN_FINAL_LEARNED_RUNTIME_SAFE_ITER2" / "final_learned_4of5_iter2_model.pt", map_location="cpu")["state_dict"])
pc_new = fix1.gated_eval(NET, teacher, vtl_n=40)
old = pd.read_csv(OUT / "FULLDATA_GATED_PRESERVED_FIX1" / "fulldata_gated_eval_per_cycle.csv")
print(f"{'Topology':16s}{'FIX1 mean':>10s}{'FA mean':>10s}{'FIX1 min':>10s}{'FA min':>10s}{'dMean':>8s}")
for t in old.topology.unique():
    go = old[old.topology == t]; gn = pc_new[pc_new.topology == t]
    print(f"{t:16s}{go.PR.mean()*100:>9.3f}%{gn.PR.mean()*100:>9.3f}%{go.PR.min()*100:>9.3f}%{gn.PR.min()*100:>9.3f}%{(gn.PR.mean()-go.PR.mean())*100:>7.3f}")
print(f"\nPOOLED  FIX1 mean={old.PR.mean()*100:.3f}%  FA mean={pc_new.PR.mean()*100:.3f}%  (delta {(pc_new.PR.mean()-old.PR.mean())*100:+.3f})")
pc_new.to_csv(OUT / "FAILAWARE_FINETUNE" / "failaware_normal_eval_per_cycle.csv", index=False)
print("DONE")
