#!/usr/bin/env python3
"""Solve the 1 Tiscali failure state that timed out on CBC, using the validated HiGHS exact solver,
and patch it into the tiscali strict-failure cache."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import scripts.phase1_5.run_fix1_failure_baseline_comparison as R
from scripts.phase1_5.strict_mcf_highs import solve_exact_mcf_highs

SF = "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/STRICT_FAILURE_FULL_MCF_PR/_partial/tiscali.progress.csv"
d = pd.read_csv(SF)
bad = d[~np.isfinite(d.strict_full_mcf_MLU)]
print("timed-out tiscali states:", bad[["scenario", "tm_id"]].to_dict("records"), flush=True)
if bad.empty:
    print("none — tiscali already complete"); sys.exit(0)
base = R.make_base_ctx("tiscali", 200, 220); ds = base.ds; caps0 = np.asarray(base.caps0, float)
for r in bad.itertuples():
    caps = R.modify_caps(caps0, r.scenario); tm = np.asarray(ds.tm[int(r.tm_id)], float) * R.spike_factor_for(r.scenario)
    a = solve_exact_mcf_highs(tm, ds.od_pairs, ds.nodes, ds.edges, caps, time_limit_sec=600)
    print(f"tiscali {r.scenario} tm{int(r.tm_id)}: status={a['status']} mlu={a['mlu']:.6f} solve={a['solve_s']}s "
          f"capViol={a['max_cap_violation']:.2e} fcResid={a['max_flow_conservation_residual']:.2e}", flush=True)
    if a["status"].lower().startswith("optimal"):
        d.loc[(d.scenario == r.scenario) & (d.tm_id == r.tm_id), "strict_full_mcf_MLU"] = a["mlu"]
        d.loc[(d.scenario == r.scenario) & (d.tm_id == r.tm_id), "solver_status"] = "Optimal_HiGHS"
d.to_csv(SF, index=False)
import numpy as np2
print("tiscali now: inf remaining =", int((~np.isfinite(d.strict_full_mcf_MLU)).sum()), "/", len(d))
print("DONE")
