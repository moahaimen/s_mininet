#!/usr/bin/env python3
"""Compute VtlWavenet's strict full-MCF failure cache (all 120 states) via the EXACT source-aggregated solver,
completing STRICT_FAILURE_FULL_MCF_PR for all 8 topologies. Same exact-state protocol as the other 7 topos
(modify_caps + spike_factor). Values are genuine strict full-MCF (validated == per-OD optimum)."""
import sys, time, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
import scripts.phase1_5.run_fix1_failure_baseline_comparison as R
from scripts.phase1_5.strict_mcf_source_agg import solve_source_mcf
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import OUT_ROOT

OUT = OUT_ROOT / "condition_compliant_k10_k50"
PART = OUT / "STRICT_FAILURE_FULL_MCF_PR" / "_partial"
def _hash(a): return hashlib.md5(np.round(np.asarray(a, float), 9).tobytes()).hexdigest()[:12]

topo = "vtlwavenet2011"; lo, hi = R.TOPOS[topo]
base = R.make_base_ctx(topo, lo, hi); ds = base.ds; caps0 = np.asarray(base.caps0, float)
rows = []; t0 = time.perf_counter(); i = 0
for scen in R.SCENARIOS:
    caps = R.modify_caps(caps0, scen); failed = R.failed_edge_ids(caps0, scen)
    for t in range(lo, hi):
        tm = np.asarray(ds.tm[t], float) * R.spike_factor_for(scen)
        a = solve_source_mcf(tm, ds.od_pairs, ds.nodes, ds.edges, caps, time_limit_sec=300)
        rows.append(dict(topology=topo, tm_id=int(t), scenario=scen, event_id=f"{topo}|{scen}|{int(t)}",
            failed_links=",".join(str(x) for x in failed), capacity_hash=_hash(caps), demand_hash=_hash(tm),
            strict_full_mcf_MLU=a["mlu"], solver_status=("Optimal_sourceagg" if a["status"].lower().startswith("optimal") else a["status"]),
            capacity_feasible=bool(a["status"].lower().startswith("optimal") and a["mlu"] <= 1.0 + 1e-6),
            max_capacity_violation=float(a["max_cap_violation"]),
            physically_disconnected_od_count=int(a["disconnected_od"]),
            provenance="source-aggregated full-MCF (HiGHS); exact == per-OD optimum; state=modify_caps+spike_factor"))
        i += 1
        if i % 20 == 0: print(f"  vtl {i}/120 ({time.perf_counter()-t0:.0f}s) last={scen}/{t} mlu={a['mlu']:.2f} {a['status']}", flush=True)
df = pd.DataFrame(rows); df.to_csv(PART / f"{topo}.progress.csv", index=False)
inf = int((~np.isfinite(df.strict_full_mcf_MLU)).sum())
print(f"[done] vtl {len(df)}/120  Optimal={(df.solver_status.str.startswith('Optimal')).sum()}  inf={inf}  ({time.perf_counter()-t0:.0f}s)")
print("DONE")
