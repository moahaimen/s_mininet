# Safety-Hardened Mode — STATUS pointer (2026-07-07)

Experiment: "RG-GNN-LPD with Safety-Hardened Initialization" (global safety rules on the FIX1 model).
VERDICT: DOES NOT fully pass. 7/8 topologies clear MinPR>=0.90; Germany50 = 0.8974 at the max global
ladder rung (K800+DB0.15). Do NOT claim all-PR>0.90. Established final method stays RG-GNN-LPD (FIX1).

Full details, table, and resume options:
  results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50/FIX1_RG_GNN_LPD_SAFETY_FINAL/VERDICT.md

Artifacts (same folder): safety_summary.csv, safety_pc_<topo>.csv, ladder_runs.csv, safety_finals.json
Script: scripts/phase1_5/run_fix1_safety_final.py
Runtime caveat: safety hardening pushes large topos OVER 500 ms (VtlWavenet p95 654, Tiscali p95 523).
NOTE: separate from the N=3976 "strict-all" completed_metrics track described in PROJECT_HANDOFF_FIX1_FINAL.md.
