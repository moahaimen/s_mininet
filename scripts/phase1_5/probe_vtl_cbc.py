#!/usr/bin/env python3
"""Probe the ORIGINAL CBC solver (solve_full_mcf_min_mlu) on vtl normal vs failure with generous time.
CBC solved the vtl NORMAL strict cache, so this checks whether failure states just need more time."""
import sys, time, numpy as np
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import scripts.phase1_5.run_fix1_failure_baseline_comparison as R
from te.lp_solver import solve_full_mcf_min_mlu

base = R.make_base_ctx("vtlwavenet2011", 0, 20); ds = base.ds; caps0 = np.asarray(base.caps0, float)
t = 4; tm = np.asarray(ds.tm[t], float)
for label, caps, tl in [("normal", caps0, 600), ("single_link", R.modify_caps(caps0, "single_link_failure"), 1200)]:
    s0 = time.perf_counter()
    r = solve_full_mcf_min_mlu(tm, ds.od_pairs, ds.nodes, ds.edges, caps, time_limit_sec=tl)
    print(f"CBC vtl {label:12s} tm{t} (limit={tl}s): status={r.status} mlu={r.mlu} wall={time.perf_counter()-s0:.1f}s", flush=True)
print("DONE")
