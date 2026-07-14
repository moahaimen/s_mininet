#!/usr/bin/env python3
"""Probe HiGHS solver methods on VtlWavenet failure state (CBC could not solve it)."""
import sys, numpy as np
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import scripts.phase1_5.run_fix1_failure_baseline_comparison as R
from scripts.phase1_5.strict_mcf_highs import solve_exact_mcf_highs

base = R.make_base_ctx("vtlwavenet2011", 0, 20); ds = base.ds; caps0 = np.asarray(base.caps0, float)
t = 4
for label, caps in [("normal", caps0), ("single_link", R.modify_caps(caps0, "single_link_failure"))]:
    tm = np.asarray(ds.tm[t], float)
    for method in ["choose", "simplex"]:
        a = solve_exact_mcf_highs(tm, ds.od_pairs, ds.nodes, ds.edges, caps, time_limit_sec=900,
                                  method=method, presolve=True)
        print(f"vtl {label:12s} tm{t} method={method:8s}: status={a['status']} mlu={a['mlu']:.6f} "
              f"solve={a['solve_s']}s simplex_iters={a['simplex_iters']} capViol={a['max_cap_violation']:.2e} "
              f"fcResid={a['max_flow_conservation_residual']:.2e}", flush=True)
        if a["status"].lower().startswith("optimal"):
            print(f"  -> SOLVED with method={method}", flush=True); break
print("DONE")
