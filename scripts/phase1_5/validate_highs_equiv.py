#!/usr/bin/env python3
"""Validate HiGHS exact solver == stored CBC strict values (solved failure states), then probe VtlWavenet."""
import sys, numpy as np, pandas as pd, pickle
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import CONFIG, _build_spec_lookup, build_context
from phase1_reactive.eval.common import load_bundle
import scripts.phase1_5.run_fix1_failure_baseline_comparison as R
from scripts.phase1_5.strict_mcf_highs import solve_exact_mcf_highs

P = pickle.load(open("results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/_prepass.pkl", "rb"))
SF = "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/STRICT_FAILURE_FULL_MCF_PR/_partial"
b = load_bundle(CONFIG); lk = _build_spec_lookup(b)

print("=== EQUIVALENCE: HiGHS exact vs stored CBC strict (failure states) ===", flush=True)
for topo, win in [("abilene", (2016, 2036)), ("geant", (672, 692)), ("tiscali", (200, 220))]:
    base = R.make_base_ctx(topo, win[0], win[1]); ds = base.ds; caps0 = np.asarray(base.caps0, float)
    cb = pd.read_csv(f"{SF}/{topo}.progress.csv")
    for scen in ["single_link_failure", "three_link_failure"]:
        t = win[0]
        rr = cb[(cb.scenario == scen) & (cb.tm_id == t)]
        if rr.empty: continue
        row = rr.iloc[0]
        if not np.isfinite(row.strict_full_mcf_MLU):
            print(f"{topo} {scen} tm{t}: stored CBC not finite (skip)"); continue
        caps_f = R.modify_caps(caps0, scen); tm = np.asarray(ds.tm[t], float) * R.spike_factor_for(scen)
        a = solve_exact_mcf_highs(tm, ds.od_pairs, ds.nodes, ds.edges, caps_f, time_limit_sec=300)
        diff = abs(a["mlu"] - float(row.strict_full_mcf_MLU))
        print(f"{topo:9s} {scen:20s} tm{t}: HiGHS={a['mlu']:.8f} CBC={row.strict_full_mcf_MLU:.8f} "
              f"|diff|={diff:.2e} capViol={a['max_cap_violation']:.2e} fcResid={a['max_flow_conservation_residual']:.2e} "
              f"solve={a['solve_s']}s status={a['status']}", flush=True)

print("\n=== VtlWavenet probe (was intractable on CBC) ===", flush=True)
base = R.make_base_ctx("vtlwavenet2011", 0, 20); ds = base.ds; caps0 = np.asarray(base.caps0, float)
t = 4
# normal (full caps) then single_link
for label, caps in [("normal", caps0), ("single_link", R.modify_caps(caps0, "single_link_failure"))]:
    tm = np.asarray(ds.tm[t], float)
    a = solve_exact_mcf_highs(tm, ds.od_pairs, ds.nodes, ds.edges, caps, time_limit_sec=600)
    print(f"vtl {label:12s} tm{t}: status={a['status']} mlu={a['mlu']:.8f} vars={a['total_vars']} nnz={a['nonzeros']} "
          f"build={a['build_s']}s solve={a['solve_s']}s ipm_iters={a['ipm_iters']} capViol={a['max_cap_violation']:.2e} "
          f"fcResid={a['max_flow_conservation_residual']:.2e}", flush=True)
print("DONE")
