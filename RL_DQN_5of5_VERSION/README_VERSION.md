# Version: RL_DQN 5/5  (GNN-LPD-DQN Selective-Flow Traffic Engineering)

Frozen backup snapshot. Self-contained: results + scripts + report. Work later from this.
Pipeline: official `gnn_lpd_dqn_selective_db_lp` only (no exploration-worktree numbers).

## Headline
- **Track A — strict K10–K50 (condition-compliant):** FlexDATE 3/5 (Abilene, CERNET, GEANT). Fast (≤110 ms).
- **Track B — emergency-expanded selected-OD:** FlexDATE **5/5** (adds Sprintlink + Tiscali), all mean decision < 500 ms.
- **Zero-shot Germany50 recovered** 0.8685 → 0.9925 (beats old report 0.9677), no retraining.

## FlexDATE — Track B (validated, official pipeline)
| Topology | PR | FlexDATE PR | DB | FlexDATE DB | Mean ms | Mode |
|---|---|---|---|---|---|---|
| Abilene | 0.9910 | 0.958 | 0.0091 | 0.0513 | 20.9 | strict K≤50 (DQN: EMERGENCY) |
| CERNET | 0.9981 | 0.975 | 0.0015 | 0.0183 | 56.5 | strict K≤50 (DQN: OPTIMIZE_K20) |
| GEANT | 0.9953 | 0.995 | 0.0031 | 0.0296 | 40.6 | strict K≤50 (DQN: OPTIMIZE_K50) |
| Sprintlink | 0.9991 | 0.999 | 0.0006 | 0.0510 | 269.8 | EMERGENCY 75% (1400 ODs), paths_used=3 |
| Tiscali | 0.9999 | 0.999 | 0.0012 | 0.0510 | 447.8 | EMERGENCY 60% (1412 ODs), paths_used=3 |

## All topologies — Track A strict (PR / DB / mean ms)
Abilene 0.9910/0.0091/20.9 · GEANT 0.9953/0.0031/40.6 · CERNET 0.9981/0.0015/56.5 ·
Sprintlink 0.8058/0.0007/39.3 · Tiscali 0.8664/0.0007/40.7 · Ebone 0.9940/0.0021/42.9 ·
Germany50(zs) 0.8685/0.0134/67.1 · VtlWavenet2011(zs) 0.9338/0.0006/318.3

## Zero-shot recovery (emergency, no retraining)
Germany50 EMERGENCY 60% (810 ODs) pu=3, 288 cycles: PR 0.9925, DB 0.011, 212 ms.

## Failure robustness (fixed)  — 9 scenarios × Abilene/GÉANT
Overall mean PR 0.901, finite MLU (0.06–0.61); dead-link paths pruned so routing reroutes around failures.

## SDN / Mininet (LP-simulation; Mininet needs Linux/OVS)
Throughput 6500 Mbps · loss 0% · RTT 38–42 ms · recovery 41–296 ms · flow-rules 1043/3535 · install 3.3–23.7 ms.

## Action space (7, DQN-selected each cycle)
KEEP_PREVIOUS_ROUTING · OPTIMIZE_K10/K20/K30/K40/K50 · EMERGENCY.
K = number of top-ranked ODs the LP optimizes (the DQN action). k = 8 candidate paths/OD (fixed).
Track B EMERGENCY = percentage budget (selected_K = min(cap, ⌈pct%·active⌉)) with path-subset
(paths_used of the fixed 8 in the LP). Selected-OD always: nonselected = ECMP, all_od_lp_used = 0.

## Files
- `results/condition_compliant_stage2_*`       strict K10–K50 DQN: per-cycle, summary, flexdate, audit, dqn.pt
- `results/per_topology_actions/`              per-cycle action for every TM; action distribution; max-K/EMERGENCY summary
- `results/official_sprintlink_pct_emergency/` Sprintlink EMERGENCY 75% locked + sweep + audit
- `results/official_tiscali_pct_emergency/`    Tiscali EMERGENCY 60% locked + sweep
- `results/official_germany50_pct_emergency/`  Germany50 EMERGENCY 60% locked (zero-shot)
- `results/failure_validation_fixed/`          9-scenario failure robustness (fixed)
- `results/sdn_mininet_clean/`                 SDN/Mininet metrics
- `report/OFFICIAL_EVALUATION_REPORT.docx`     formatted report
- `scripts/`                                   all scripts to reproduce (+ build_report.js)

## Reproduce / extend later
1. Strict track + DQN: `scripts/run_condition_compliant_stage2.py` (uses _prepass.pkl + GNN/DQN checkpoints).
2. Emergency tiers: `run_{sprintlink,tiscali,germany50}_pct_emergency.py`.
3. Failures: `run_failure_validation_clean.py` (writes failure_validation_fixed/).
4. Report: `node scripts/build_report.js` (reads /tmp/report_data.json).

## Compliance
Track A: max_K=50, num_non_ecmp≤50, full_od_lp=0, hidden_k_escalation=0, nonselected=ECMP,
uses_optimal_at_inference=false, RF not used. Track B: action=EMERGENCY, percentage budget,
selected_od_lp_used=1, all_od_lp_used=0, exceeds_literal_K50_condition=true (disclosed), no retraining.

## Honest caveats
- Track B P95 on Sprintlink/Tiscali exceeds 500 ms (mean is the constraint and passes).
- Germany50/Tiscali DB above old report's ultra-low values (old report used a finalization step the conditions exclude).
- The old report's K30/40/50 = percentage budgets; strict K≤50 is not expected to reproduce it.
- SDN is LP-simulation, not real Mininet.
