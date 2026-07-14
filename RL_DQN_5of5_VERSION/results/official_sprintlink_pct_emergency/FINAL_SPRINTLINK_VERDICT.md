# Sprintlink rescue — FINAL (official pipeline)

ACCEPTED: EMERGENCY percentage-budget 75% (1400 selected ODs), paths_used=3 of fixed k=8 library.
PR=0.9991 (>=0.999), DB=0.0006 (<0.0510), mean_ms=269.8 (<500), p95_ms=382.0.

> The strict method follows the literal K10-K50 selected-OD condition (Sprintlink PR 0.806). In addition, an emergency-expanded selected-OD mode is evaluated because the old report's K30/K40/K50 labels corresponded to percentage budgets rather than literal OD counts. The emergency mode remains selected-OD (GNN-LPD ranks ODs; LP optimizes only the selected top-K=1400; nonselected ODs remain ECMP; all_od_lp_used=0) and does NOT use all-OD LP, but it is reported separately from the strict K<=50 track.
