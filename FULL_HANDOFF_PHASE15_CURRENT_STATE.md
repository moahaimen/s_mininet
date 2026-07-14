# Full Handoff — Phase 1.5 Network TE — Current State

> **Read this fully before doing anything.** Older reports/results in this project are
> NOT automatically trusted — the method went through several corrections, and one
> critical deployability blocker (R1) is still open. Do **not** run zero-shot training,
> do **not** rewrite the report, do **not** overwrite any existing result files.

---

## 1. Project identity

- **Project:** Phase 1.5 Network Traffic Engineering.
- **Method family:** GNN-LPD (LP-distilled GNN OD-ranking) + DQN/KEEP controller +
  selected-flow DB-budgeted LP.
- **Repository:** `f_flex_network_code` (working clone: `f_flex_network_code_clean`).
- **Main goal:** compare the method against FlexDATE / source-locked references and
  against **corrected true-ECMP** baselines, under a `< 500 ms` decision-time budget.
- **Status:** Multiple methodological corrections have been applied. A critical
  blocker (R1, Section 10) means the latest "calibrated FlexDATE wins" are currently
  **oracle-informed upper bounds**, not proven deployable results.

---

## 2. Repository and likely paths

Likely repo roots (verify which is canonical):
```
/Users/moahaimentalib/Desktop/f_flex_network_code_clean   <- work happened here
/Users/moahaimentalib/Desktop/f_flex_network_code_test
```

Known important scripts/files:
```
scripts/phase1_5/gnn_lpd_dqn_selective_db_lp.py     <- main method: action space, env, single-K solve, eval
scripts/phase1_5/run_failure_validation_clean.py
scripts/phase1_5/run_sdn_mininet_clean.py
scripts/phase1_5/audit_gnn_lpd_dqn_clean_method.py  <- legacy-format audit (expects per_cycle.csv etc.)
scripts/phase1_5/windows_smoke_test.py
scripts/phase1_5/gnn_lp_inference.py                <- score_lp_gnn_cycle (GNN scoring)
te/lp_solver.py                                     <- solve_all_od_path_lp, solve_selected_path_lp_dbbudget
te/simulator.py                                     <- apply_routing (returns .mlu, .link_loads, .utilization)
te/baselines.py                                     <- ecmp_splits, clone_splits
```

Important result folders:
```
results/gnn_lpd_dqn_selective_db_lp/
results/gnn_lpd_dqn_selective_db_lp/final_N3976/                       <- LEGACY (contaminated semantics) — do not trust as final
results/gnn_lpd_dqn_selective_db_lp/corrected_true_ecmp_eval/          <- CORRECTED pipeline
results/gnn_lpd_dqn_selective_db_lp/corrected_true_ecmp_eval/unified_dqn/        <- deployable trained DQN (cheap features)
results/gnn_lpd_dqn_selective_db_lp/corrected_true_ecmp_eval/flexdate_constrained/  <- calibrated FlexDATE (oracle-informed KEEP — see R1)
results/gnn_lpd_dqn_selective_db_lp/corrected_true_ecmp_eval/stage3/             <- degree-cap provenance + dense-topo prepass
```

**Reproducibility risk:** several calibration / sweep scripts currently live ONLY in
`/tmp` and may have been cleared (the harness wiped `/tmp` at least once mid-project):
```
/tmp/flexdate_calibration.py, /tmp/flexdate_calibration_v2.py, /tmp/flexdate_final.py,
/tmp/geant_sprint_sweep.py, /tmp/internal_calibration.py, /tmp/vtl_eval.py,
/tmp/assemble_all.py, /tmp/unified_dqn.py, /tmp/build_policy.py,
/tmp/stage1_corrected_eval.py, /tmp/stage2_retrain_dqn.py, /tmp/stage3_capacity_eval.py,
/tmp/runtime_oracle_full.py, /tmp/md_to_docx.py
```
**Identify which scripts are still needed and copy them into `scripts/phase1_5/`.**
If a script is gone, record that reproducibility is incomplete for that artifact.

---

## 3. Original problem and contamination history

The legacy pipeline (`final_N3976/`) had contamination risks:
- **KEEP was preserving previously OPTIMIZED routing and was conflated with ECMP.**
  The recorded "ECMP/KEEP baseline" therefore looked near-optimal (PR ≈ 0.98), making
  optimization look unnecessary and collapsing the policy toward KEEP. This was a
  **measurement artifact**, not a real result.
- Some legacy results relied on full-OD fallback / escalation behavior.
- Some older report rows mixed legacy and clean results.

The correction was to **separate three semantics** that had been entangled:
```
USE_ECMP_BASELINE      = true raw static ECMP for the CURRENT TM (recomputed every cycle)
KEEP_ACCEPTED_ROUTING  = reuse the LAST committed routing (which may be optimized)
OPTIMIZE_K             = selected-K GNN-LPD DB-budgeted LP (single solve)
```

Explicit facts:
```
USE_ECMP_BASELINE and KEEP_ACCEPTED_ROUTING are NOT the same.
At cycle reset, accepted routing starts as ECMP (env: prev_splits = clone_splits(ctx["ecmp"])).
At cycle 0, KEEP_ACCEPTED == ECMP ONLY because no optimized routing exists yet.
After any optimization, KEEP_ACCEPTED reuses the last accepted OPTIMIZED routing.
```

Against TRUE raw ECMP (recomputed each cycle), ECMP is far from optimal
(PR ≈ 0.34–0.92, Section 7); optimization is genuinely needed. The path-optimal
reference (`pathopt_mlu`) was verified equal to a fresh full path-LP (`solve_all_od_path_lp`),
ratio = 1.000 — i.e. the reference is correct; the earlier "ECMP wins" was the KEEP
contamination above.

---

## 4. Clean method semantics (intended)

```
GNN-LPD ranks OD pairs by criticality/improvability.
K = number of TOP-ranked OD pairs selected for the LP. Non-selected ODs stay on ECMP.
The LP optimizes ONLY the selected ODs (selected-flow), single-K, one solve.
No hidden K escalation. No hidden full-OD fallback. No RandomForest gate.
No sticky legacy gate. No Stage-2 disturbance finalization.
No solve_selected_path_lp_min_db. No heuristic criticality as the final selector.
```

Required audit fields (must hold on every cycle):
```
full_od_lp_used = 0
k_escalation_used = 0
final_selected_k <= action K
single_K_solve = true
hidden_full_od_fallback = false
criticality_backend = gnn_lpd
heuristic_used = 0
rf_used = 0
sticky_used = 0
stage2_used = 0
disturbance_finalization_used = 0
```

Note on `decision_ms`: in the current main script, `t0` was moved to BEFORE GNN scoring,
so `decision_ms` now INCLUDES GNN-LPD scoring + LP (the honest full per-cycle decision).
GNN scoring is cheap (~22–33 ms even for dense topologies); the LP dominates for large K.

---

## 5. Score / LP definitions

A fusion score formula was found in EARLIER ablation code:
```python
score = 0.60 * g + 0.25 * lpd + 0.10 * alt + 0.05 * demand
# g = trained GNN score, lpd = load/path-diversity, alt = alternate-path signal, demand = demand signal
```
**Verify whether the currently active selector uses this exact fusion or a pure
trained GNN score**, and record the provenance. (The corrected pipeline calls
`gnn_lp_inference.score_lp_gnn_cycle` → `GNNFlowSelector`; confirm the active fusion.)

DB-budgeted selected-K LP (`solve_selected_path_lp_dbbudget` in `te/lp_solver.py`),
intended semantics:
```
Minimize MLU (max link utilization) over candidate paths of the SELECTED ODs,
  subject to: capacity constraints on all links,
              disturbance(prev_accepted, new) <= db_budget,
  with a tiny db_weight (1e-6) tie-breaker.
base   = ECMP for non-selected ODs.
prev   = accepted routing (cycle-to-cycle disturbance reference).
db_budget controls allowed cycle-to-cycle routing change.
```
**Read `te/lp_solver.py` and write the EXACT objective, variables, constraints,
and the precise disturbance (DB) definition into the handoff** (`solve_all_od_path_lp` ~line 83;
`solve_selected_path_lp_dbbudget` ~line 552). `apply_routing` returns `.mlu`, `.link_loads`,
`.utilization` (see `te/simulator.py` ~line 224).

---

## 6. Dataset protocol and topology roles

```
Seen / training-or-calibrated topologies:  Abilene, CERNET, GEANT, Sprintlink, Tiscali, Ebone
Zero-shot / unseen topologies:             Germany50, VtlWavenet2011
FlexDATE source-locked reference topos:    Abilene, CERNET, GEANT, Sprintlink
```
Corrections that MUST be respected:
```
Tiscali is NOT unseen. Tiscali = internal SEEN / NO source-locked FlexDATE reference.
Germany50 and VtlWavenet2011 are the two unseen / zero-shot topologies.
Tiscali must NOT be counted as a FlexDATE win/loss row (no source-locked reference).
```
TEST ranges (legacy N=3976 protocol, abilene/.../vtl): abilene 2016, cernet 200, geant 672,
sprintlink 200, tiscali 200, ebone 200, germany50 288, vtlwavenet2011 200 (= 3976 total).

---

## 7. Corrected true-ECMP and capacity-model history

Corrected true-ECMP PR (raw equal-split ECMP vs verified path-optimal reference):
```
Abilene        true ECMP PR = 0.494
CERNET         true ECMP PR = 0.917
GEANT          true ECMP PR = 0.403
Sprintlink     true ECMP PR = 0.692
Germany50      true ECMP PR = 0.344
Tiscali        true ECMP PR = 0.797
VtlWavenet2011 true ECMP PR = 0.918
```

Capacity model:
```
Abilene, GEANT       : original real capacities (no change).
Germany50            : original uniform capacities (~40).
CERNET, Sprintlink,
  Tiscali, Vtl       : placeholder uniform caps = 10 in Topology Zoo -> CORRECTED with
                       degree-based heterogeneous capacities:
                       capacity_e = base * sqrt(deg_u * deg_v), demand rescaled to
                       target optimal utilization ~= 0.85.
```
Provenance file: `corrected_true_ecmp_eval/stage3/capacity_model_provenance.json`
(degree_sqrt, formula, target util 0.85). **Note:** the degree-cap correction was applied
to CERNET/Sprintlink/Tiscali in `stage3/`; VtlWavenet2011's degree-cap eval is in
`corrected_true_ecmp_eval/vtl_corrected_eval.json` (built later, 15-cycle reduced run due
to 8372 ODs / ~13.3 s optimal solves). Confirm a single consolidated
provenance file covers all four corrected topologies.

---

## 8. Confirmed calibrated all-topology result table (current)

`[rt]` = best runtime-compliant row; `[acc]` = best-accuracy row. Pareto = no single
current policy satisfies both PR target and mean runtime < 500 ms.
Source CSV: `corrected_true_ecmp_eval/ALL_TOPOLOGIES_FULL_TABLE.csv`
(+ `all_topologies_final_table.csv`, `internal_unseen_final_table.csv`).

```
Topology            Role             ECMP_PR Our_PR PR_tgt PRwin Our_DB  DB_tgt DBwin Mean_ms P95  <500 MeanK MaxK Policy           Status
Abilene             FlexDATE         0.494   0.9825 0.958  yes   0.0057  0.0513 yes   10      20   yes  26    80   KEEP+K80         WIN
CERNET              FlexDATE         0.917   0.9988 0.975  yes   0.0003  0.0183 yes   36      208  yes  42    500  KEEP+K500        WIN
GEANT               FlexDATE         0.403   0.9992 0.995  yes   0.0036  0.0296 yes   95      150  yes  185   200  K200 DB-budgeted WIN
Sprintlink [rt]     FlexDATE         0.692   0.9707 0.999  no    0.0019  0.0510 yes   380     450  yes  500   500  K500             Pareto PR<0.999
Sprintlink [acc]    FlexDATE         0.692   0.9994 0.999  yes   0.0009  0.0510 yes   1004    1330 no   1377  1400 K1400            Pareto ms>500
Germany50           internal/unseen  0.344   0.9521 0.950  yes   0.0093  0.0300 yes   250     470  yes  356   500  K500+KEEP        WIN
Vtl [rt]            internal/unseen  0.918   0.9377 0.950  no    0.0009  0.0300 yes   485     560  yes  200   200  K200 DBbud0.08   Limitation PR<0.95
Vtl [acc]           internal/unseen  0.918   0.9443 0.950  no    0.0013  0.0300 yes   610     760  no   400   400  K400 DBbud0.08   Limitation PR<0.95
Tiscali [rt]        internal seen    0.797   0.9327 0.950  no    0.0021  0.0300 yes   352     420  yes  500   500  K500             Pareto PR<0.95
Tiscali [acc]       internal seen    0.797   0.9616 0.950  yes   0.0022  0.0300 yes   702     850  no   800   800  K800             Pareto ms>500
```
**All ms values above are subject to R2 (thermal-throttle sensitivity).** PR/DB are reliable.

---

## 9. Outcome summary of calibrated results

```
FlexDATE official result is NOT 4/4.
It is 3/4 clean PR+DB+mean-runtime wins: Abilene, CERNET, GEANT.
Sprintlink is a runtime-accuracy PARETO case:
  PR>=0.999 requires K1400 with mean ~1004 ms (>500); runtime-compliant K500 gives PR ~0.971.

Internal:
  Germany50 wins internal target (PR 0.952, DB 0.009, mean ~250 ms — pending clean re-measure).
  Tiscali is Pareto (K500 fast/PR 0.933 vs K800 PR 0.962/~702 ms).
  VtlWavenet2011 is a LIMITATION: accuracy row PR 0.9443 < 0.95, and K400 runtime exceeds 500.
```

---

## 10. CRITICAL BLOCKER R1 — Deployability (most important)

The calibrated KEEP-vs-OPTIMIZE decision used:
```
if pr_of(optimal_mlu, keep_mlu) >= keep_threshold:  KEEP   else: OPTIMIZE
```
i.e. it reads the **per-cycle path-optimal MLU** to decide whether KEEP is safe. This is
**NOT deployable** — at inference the controller must not know the per-cycle optimum before
deciding (computing it defeats the purpose). Affected scripts: the `/tmp/flexdate_*` and
`/tmp/geant_sprint_sweep.py`, `/tmp/internal_calibration.py`, `/tmp/vtl_eval.py` `run()`
functions (KEEP branch). The low-DB FlexDATE wins (esp. Abilene/CERNET, where KEEP fires
often) DEPEND on this oracle-informed decision.

**Consequence:**
```
The calibrated FlexDATE wins are ORACLE-INFORMED UPPER BOUNDS unless a DEPLOYABLE
KEEP proxy (cheap features only) reproduces them.
```

The only currently-deployable policy is the trained unified DQN
(`corrected_true_ecmp_eval/unified_dqn/`, cheap features), BUT it had HIGHER DB
(e.g. Abilene 0.046, GEANT 0.112 cycle-to-cycle) and would NOT pass the FlexDATE DB
targets. So the deployable result is currently WEAKER than the calibrated upper bound.

**Forbidden inference features (if ANY appears at inference -> `deployable = false`, stop):**
```
optimal_mlu, pathopt_mlu, oracle_pr, pr_of(optimal_mlu, keep_mlu),
future TM, solving candidate actions before choosing KEEP,
any metric requiring the per-cycle optimum.
```
**Allowed deployable features:**
```
topology size (nodes/edges/OD count), current demand stats, cycle-to-cycle demand delta,
current ECMP load estimate, accepted-routing load/MLU estimate under current demand,
accepted-routing max-link-utilization estimate, GNN-LPD score percentiles,
top-score concentration, previous action, previous selected K, previous accepted DB,
previous accepted MLU estimate, predicted runtime estimate.
```

---

## 11. CRITICAL BLOCKER R2 — Runtime measurement stability

`mean_ms` / `p95_ms` are affected by thermal throttling and load-average tail on this
laptop (observed: GEANT 95 ms clean vs 1257 ms throttled; Sprintlink KEEP 34 ms vs
740–1079 ms under load). **PR/DB are throttle-independent.** Re-measure boundary runtime
cases CLEANLY in isolation (one process, cool CPU) before any `<500 ms` claim:
```
Germany50, Tiscali [rt], Vtl [rt], GEANT (if needed), Sprintlink (if used).
```

---

## 12. Zero-shot claim status

```
Current results DO NOT support true zero-shot generalization.
Germany50 and VtlWavenet2011 used per-topology K budgets chosen AFTER seeing their results.
Topology-specific K/thresholds violate a pure zero-shot claim.
```
Safe wording:
```
The calibrated deployment results include unseen topologies as internal stress/transfer
rows, but they are NOT evidence of pure zero-shot generalization.
```
True zero-shot requires: one frozen global policy, one global action space, no
topology-specific K, no topology-specific thresholds, no tuning on Germany50 / Vtl.

Possible FUTURE design (DO NOT RUN until R1 is solved):
```
Normalized global actions:
  USE_ECMP_BASELINE, KEEP_ACCEPTED_ROUTING,
  OPTIMIZE_TOP_1_PERCENT_DB_0.01, OPTIMIZE_TOP_2_PERCENT_DB_0.01,
  OPTIMIZE_TOP_5_PERCENT_DB_0.01, OPTIMIZE_TOP_10_PERCENT_DB_0.01,
  OPTIMIZE_TOP_20_PERCENT_DB_0.01
  K = ceil(percent * active_OD_count); global runtime cap; deployable features only.
```

---

## 13. Student-question history and safe answers

```
1.  Actions = semantic decisions: USE_ECMP_BASELINE, KEEP_ACCEPTED_ROUTING, OPTIMIZE_K.
2.  K = number of selected top-ranked OD pairs, NOT full-OD optimization.
3.  Current K is topology-calibrated; therefore NOT zero-shot.
4.  First TM starts from ECMP (accepted = ECMP at reset).
5.  USE_ECMP and KEEP_ACCEPTED differ after the first accepted optimization.
6.  Optimization = selected-K DB-budgeted LP (single solve).
7.  Topology-specific K violates pure zero-shot.
8.  Tiscali is NOT a FlexDATE win (no source-locked reference).
9.  Tiscali is NOT unseen (internal seen).
10. Pareto rows shown because no single policy meets both PR target and mean runtime <500.
11. Pareto = runtime/accuracy tradeoff.
12. CANNOT claim 4/4 FlexDATE PR+DB+runtime; only 3/4 + Sprintlink Pareto.
13. CANNOT claim zero-shot from current calibrated results.
```

---

## 14. What must be done next (ONLY this)

Do NOT run zero-shot. Do NOT rewrite the report. Do NOT overwrite previous results.
The next task is ONLY: **close R1 deployability.**

Create (new folder only):
```
results/gnn_lpd_dqn_selective_db_lp/corrected_true_ecmp_eval/deployability_audit/
```
Required output files:
```
deployability_feature_audit.json     <- per policy/script: which inference features are used; uses_optimal_at_inference; deployable flag
oracle_vs_deployable_summary.csv     <- the 3-rows-per-topology comparison
deployable_keep_proxy_eval.csv       <- results of a deployable KEEP-proxy candidate (cheap features only)
deployability_acceptance.json        <- pass/fail vs the criteria in Section 15
```
Required scripts saved IN REPO (not /tmp):
```
scripts/phase1_5/run_deployability_audit.py
scripts/phase1_5/run_deployable_keep_proxy_eval.py
```
The deployability audit must compare three rows per topology where available:
```
1. oracle_keep_calibrated            (current calibrated controller; uses_optimal_at_inference = true)
2. existing_deployable_unified_dqn   (corrected_true_ecmp_eval/unified_dqn; cheap features)
3. deployable_keep_proxy_candidate   (new: KEEP decision from allowed features only)
```
Across:
```
Abilene, CERNET, GEANT, Sprintlink, Germany50, Tiscali, VtlWavenet2011
```
Per row report:
```
Topology, Policy type, Uses optimal at inference? (yes/no), Deployable? (yes/no),
ECMP PR, Our PR, PR target, PR pass, Our DB, DB target, DB pass,
Mean decision_ms, P95 decision_ms, Runtime pass, Action mix, Mean K, Max K, Overall status.
```

---

## 15. Acceptance criteria for deployability

A row may be called FINAL DEPLOYABLE only if:
```
uses_optimal_at_inference = false
deployable = true
full_od_lp_used = 0
k_escalation_used = 0
final_selected_k <= action K
single_K_solve = true
hidden_full_od_fallback = false
```
FlexDATE targets (deployable):
```
Abilene:    PR >= 0.958, DB <= 0.0513, mean_ms < 500
CERNET:     PR >= 0.975, DB <= 0.0183, mean_ms < 500
GEANT:      PR >= 0.995, DB <= 0.0296, mean_ms < 500
Sprintlink: PR >= 0.999, DB <= 0.0510, mean_ms < 500
```
If Sprintlink cannot pass deployably, mark Pareto. **Do NOT force 4/4.**

Stop rule: if the deployable proxy cannot reproduce the oracle-informed calibrated wins,
DO NOT hide it. Report:
```
Final deployable claim       = existing deployable result / weaker result
Oracle-informed calibrated   = upper-bound only
```
Do NOT call oracle-informed results the "final deployable method."

---

## 16. Required final recommendation (end with exactly one)

```
A. R1 closed: deployable KEEP policy reproduces the calibrated wins.
B. R1 not closed: calibrated wins are oracle-informed upper bounds only.
C. More evidence required: specific missing files or scripts.
```

---

## 17. Explicit warnings

```
Do not call topology-calibrated results zero-shot.
Do not call Tiscali unseen.
Do not count Tiscali as FlexDATE.
Do not claim 4/4 FlexDATE wins.
Do not call oracle-informed KEEP results deployable.
Do not use optimal_mlu / pathopt_mlu / oracle_pr at inference.
Do not overwrite previous results.
Do not update the DOCX report yet.
```

---

## Next command

> **Close R1 only.** Build the deployable KEEP decision (allowed features in Section 10),
> evaluate the three policy rows (oracle_keep_calibrated, existing_deployable_unified_dqn,
> deployable_keep_proxy_candidate) across all seven topologies, and write the four output
> files into `corrected_true_ecmp_eval/deployability_audit/` plus the two scripts into
> `scripts/phase1_5/`. Do not run zero-shot training. Do not edit the report. Do not
> overwrite existing results. End with recommendation A, B, or C (Section 16).
