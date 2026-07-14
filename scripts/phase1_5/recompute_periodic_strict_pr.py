#!/usr/bin/env python3
"""Recompute the Full-Stream Periodic Single-Link Failure PR with the STRICT full-MCF numerator
(source-aggregated exact solver), replacing the path-LP numerator. Per-cycle exact state:
  caps_state = caps0 with the failed link set to ~0 (failure cycles) else caps0;  tm = ds.tm[t] (no spike).
new_PR = min(1, strict_full_mcf / achieved_MLU). Achieved MLU is read from the periodic per-cycle CSV (unchanged).
PRODUCES NUMBERS ONLY; does not edit the report."""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import _make_envs, GNNLPDScorer, GNN_CHECKPOINT_DEFAULT, OUT_ROOT, set_seed
from scripts.phase1_5.strict_mcf_source_agg import solve_source_mcf

set_seed(42)
gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
OUT = OUT_ROOT / "condition_compliant_k10_k50"; PF = OUT / "PERIODIC_FAILURE_FIX1"
OUTD = OUT / "STRICT_FAILURE_FULL_MCF_PR" / "periodic_strict"; OUTD.mkdir(parents=True, exist_ok=True)
DISP = {"abilene":"Abilene","geant":"GEANT","cernet":"CERNET","sprintlink":"Sprintlink","tiscali":"Tiscali","ebone":"Ebone","germany50":"Germany50","vtlwavenet2011":"VtlWavenet"}
NFREQ = {"abilene":12,"geant":4,"cernet":2,"sprintlink":2,"tiscali":2,"ebone":2,"germany50":4,"vtlwavenet2011":2}

def run(topo):
    d = pd.read_csv(PF / f"periodic_{topo}.csv")
    lo, hi = int(d.tm_index.min()), int(d.tm_index.max()) + 1
    env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi - lo, 30)[0]; ctx = env.ctx
    ds = ctx["ds"]; caps0 = np.asarray(ctx["caps"], float)
    name2idx = {f"{s}->{t}": e for e, (s, t) in enumerate(ds.edges)}
    prs = []; t0 = time.perf_counter()
    for r in d.itertuples():
        caps = caps0.copy()
        if int(r.is_failure) == 1 and isinstance(r.failed_link, str) and r.failed_link in name2idx:
            caps[name2idx[r.failed_link]] = 1e-9
        tm = np.asarray(ds.tm[int(r.tm_index)], float)
        strict = solve_source_mcf(tm, ds.od_pairs, ds.nodes, ds.edges, caps, time_limit_sec=120)["mlu"]
        ach = float(r.MLU)
        prs.append(min(1.0, strict / ach) if (np.isfinite(strict) and ach > 0) else 0.0)
    d["PR_strict"] = prs
    d.to_csv(OUTD / f"periodic_strict_{topo}.csv", index=False)
    gf = d[d.is_failure == 1]
    print(f"{DISP[topo]:11s} TMs={len(d):5d} fail={int(d.is_failure.sum()):4d} every=1/{NFREQ[topo]:<2d} "
          f"| OLD meanPR={d.PR.mean():.4f} minPR={d.PR.min():.4f} PR>=90={ (d.PR>=.9).mean()*100:3.0f}% "
          f"| STRICT meanPR={np.mean(prs):.4f} minPR={min(prs):.4f} PR>=90={np.mean(np.array(prs)>=.9)*100:3.0f}% "
          f"PRfail={np.mean([prs[i] for i in range(len(prs)) if d.is_failure.iloc[i]==1]):.4f} ({time.perf_counter()-t0:.0f}s)", flush=True)
    return d

if __name__ == "__main__":
    topos = sys.argv[1].split(",") if len(sys.argv) > 1 else list(NFREQ)
    print(f"{'Topology':11s} {'':32s} OLD(path-LP) vs STRICT(full-MCF)", flush=True)
    for t in topos:
        run(t)
    print("DONE")
