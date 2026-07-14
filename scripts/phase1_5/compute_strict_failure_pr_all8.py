#!/usr/bin/env python3
"""Compute the corrected Section-11 failure PR = min(1, strict_full_mcf / achieved) for ALL 8 topologies,
using the complete strict-failure cache. PRODUCES NUMBERS ONLY (writes CSV artifacts); does NOT edit the report."""
import numpy as np, pandas as pd
from pathlib import Path

RC = Path("results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50")
CM = RC / "FINAL_REPORT_FIX1" / "completed_metrics"
SF = RC / "STRICT_FAILURE_FULL_MCF_PR"
OUTD = SF / "corrected_section11_numbers"; OUTD.mkdir(exist_ok=True)

pc = pd.read_csv(CM / "failure_baseline_comparison_fix1_per_cycle.csv")
strict = {}
for t in ["abilene","geant","ebone","cernet","germany50","sprintlink","tiscali","vtlwavenet2011"]:
    for r in pd.read_csv(SF / "_partial" / f"{t}.progress.csv").itertuples():
        strict[(t, r.scenario, int(r.tm_id))] = float(r.strict_full_mcf_MLU)

missing = [(r.Topology, r.Scenario, int(r.tm_index)) for r in pc.itertuples() if (r.Topology, r.Scenario, int(r.tm_index)) not in strict]
assert not missing, f"missing strict for {len(set(missing))} states"

pc["strict"] = [strict[(r.Topology, r.Scenario, int(r.tm_index))] for r in pc.itertuples()]
# scale sanity: strict should be <= achieved (optimum <= achieved) in the vast majority
pc["ratio"] = pc.strict / pc.achieved_mlu.where(pc.achieved_mlu > 0, np.nan)
pc["PR_old"] = pc.PR
pc["PR_new"] = np.minimum(1.0, pc.ratio).fillna(0.0)

print("=== scale sanity (strict/achieved should be ~<=1; PR is clipped) ===")
for t in ["abilene","tiscali","vtlwavenet2011"]:
    g = pc[pc.Topology == t]
    print(f"  {t:16s} strict range [{g.strict.min():.4g},{g.strict.max():.4g}] achieved [{g.achieved_mlu.min():.4g},{g.achieved_mlu.max():.4g}] ratio>1.001 frac={(g.ratio>1.001).mean()*100:.1f}%")

pc.to_csv(OUTD / "failure_baseline_per_cycle_STRICT.csv", index=False)

def agg(df, keys):
    return df.groupby(keys).apply(lambda g: pd.Series(dict(
        meanPR_old=g.PR_old.mean()*100, meanPR_new=g.PR_new.mean()*100,
        pr90_old=(g.PR_old>=.9).mean()*100, pr90_new=(g.PR_new>=.9).mean()*100,
        pr95_old=(g.PR_old>=.95).mean()*100, pr95_new=(g.PR_new>=.95).mean()*100)), include_groups=False).reset_index()

by_sm = agg(pc, ["Scenario","Method"]); by_sm.to_csv(OUTD / "by_scenario_method_old_vs_strict.csv", index=False)
fin = pc[pc.Method == "Final RG-GNN-LPD"]
by_st = agg(fin, ["Scenario","Topology"]); by_st.to_csv(OUTD / "final_by_scenario_topology_old_vs_strict.csv", index=False)

print("\n=== Final RG-GNN-LPD: old(path-lib) -> new(strict full-MCF) mean PR by scenario (all 8 topos) ===")
for sc in ["single_link_failure","two_link_failure","three_link_failure","capacity_degradation_50","spike","mixed_spike_failure"]:
    g = fin[fin.Scenario == sc]
    print(f"  {sc:24s} old={g.PR_old.mean()*100:7.3f}%  new={g.PR_new.mean()*100:7.3f}%  (n={len(g)})")
print(f"\n  OVERALL Final RG-GNN-LPD failure PR: old={fin.PR_old.mean()*100:.3f}%  new={fin.PR_new.mean()*100:.3f}%")
print("\n=== VtlWavenet final-method by scenario (old -> strict) ===")
for sc in ["single_link_failure","two_link_failure","three_link_failure","capacity_degradation_50","spike","mixed_spike_failure"]:
    g = fin[(fin.Scenario==sc)&(fin.Topology=="vtlwavenet2011")]
    print(f"  vtl {sc:24s} old={g.PR_old.mean()*100:7.3f}%  new={g.PR_new.mean()*100:7.3f}%")
print("\nSaved corrected numbers ->", OUTD)
print("DONE")
