#!/usr/bin/env python3
"""Validate source-aggregated MCF == per-OD strict cache (exactness), then solve VtlWavenet failure states."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import scripts.phase1_5.run_fix1_failure_baseline_comparison as R
from scripts.phase1_5.strict_mcf_source_agg import solve_source_mcf

SF = "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/STRICT_FAILURE_FULL_MCF_PR/_partial"
print("=== EXACTNESS: source-aggregated vs per-OD strict cache (7 solved topos) ===", flush=True)
for topo, win in [("abilene", (2016, 2036)), ("geant", (672, 692)), ("cernet", (200, 220)),
                  ("tiscali", (200, 220)), ("germany50", (0, 20))]:
    base = R.make_base_ctx(topo, win[0], win[1]); ds = base.ds; caps0 = np.asarray(base.caps0, float)
    cb = pd.read_csv(f"{SF}/{topo}.progress.csv")
    for scen in ["single_link_failure", "three_link_failure"]:
        t = win[0]; row = cb[(cb.scenario == scen) & (cb.tm_id == t)]
        if row.empty or not np.isfinite(row.iloc[0].strict_full_mcf_MLU): continue
        caps = R.modify_caps(caps0, scen); tm = np.asarray(ds.tm[t], float) * R.spike_factor_for(scen)
        a = solve_source_mcf(tm, ds.od_pairs, ds.nodes, ds.edges, caps, time_limit_sec=300)
        ref = float(row.iloc[0].strict_full_mcf_MLU); diff = abs(a["mlu"] - ref)
        print(f"{topo:9s} {scen:20s} tm{t}: src-agg={a['mlu']:.8f} perOD={ref:.8f} |diff|={diff:.2e} "
              f"K_src={a['source_commodities']} vars={a['total_vars']} solve={a['solve_s']}s "
              f"capViol={a['max_cap_violation']:.2e} resid={a['max_flow_conservation_residual']:.2e} status={a['status']}", flush=True)

print("\n=== VtlWavenet failure states (previously intractable) ===", flush=True)
base = R.make_base_ctx("vtlwavenet2011", 0, 20); ds = base.ds; caps0 = np.asarray(base.caps0, float)
for scen in ["single_link_failure", "two_link_failure", "three_link_failure",
             "capacity_degradation_50", "spike", "mixed_spike_failure"]:
    t = 4; caps = R.modify_caps(caps0, scen); tm = np.asarray(ds.tm[t], float) * R.spike_factor_for(scen)
    a = solve_source_mcf(tm, ds.od_pairs, ds.nodes, ds.edges, caps, time_limit_sec=300)
    print(f"vtl {scen:24s} tm{t}: status={a['status']} mlu={a['mlu']:.6f} K_src={a['source_commodities']} "
          f"vars={a['total_vars']} disc_od={a['disconnected_od']} solve={a['solve_s']}s "
          f"capViol={a['max_cap_violation']:.2e} resid={a['max_flow_conservation_residual']:.2e}", flush=True)
print("DONE")
