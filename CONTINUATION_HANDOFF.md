# CONTINUATION HANDOFF — Clean GNN-LPD-DQN TE Project

---

## 1. Repository and Paths

- **GitHub repo:** `https://github.com/moahaimen/f_flex_network_code`
- **Clean local working directory:** `/Users/moahaimentalib/Desktop/f_flex_network_code_clean`
- **Fresh clone validation directory:** `/Users/moahaimentalib/Desktop/f_flex_network_code_test`
- **Current latest commit:** `8ffc869`
- **Current branch:** `main`
- **Current verified git status:** `working tree clean`

**Recent commits:**

```
8ffc869 Enforce ECMP background and DQN-controlled solve scope; rename path-cost features
d95e2e4 Fix clean audit artifact filename detection
ee1f335 Add required models and evaluation artifacts
f2ec9ef Initial clean project upload
```

---

## 2. Current State

Commit `8ffc869` applied three strict methodological corrections (ECMP background enforcement,
DQN scope fix, flexdate internal rename) and passed all validation checks.

**Current corrected results:**

```
N = 3976
Mean PR = 93.346%
Mean DB = 0.695%
Mean decision time = 50.9 ms
P95 decision time = 337.2 ms
Full-OD fallback = 0.0%
Audit = 13/13 PASS
Fresh-clone smoke + audit = PASS
```

**Per-topology breakdown:**

```
Abilene:       N=2016  PR=97.157%  DB=0.891%  FO=0.0%  mean=7.2ms   p95=7.8ms
CERNET:        N=200   PR=71.256%  DB=0.806%  FO=0.0%  mean=87.2ms  p95=90.4ms
GEANT:         N=672   PR=99.153%  DB=0.471%  FO=0.0%  mean=45.5ms  p95=53.7ms
Sprintlink:    N=200   PR=86.961%  DB=0.203%  FO=0.0%  mean=92.7ms  p95=114.0ms
Tiscali:       N=200   PR=71.191%  DB=0.222%  FO=0.0%  mean=120.1ms p95=130.1ms
Ebone:         N=200   PR=99.992%  DB=0.228%  FO=0.0%  mean=15.4ms  p95=16.9ms
Germany50:     N=288   PR=84.476%  DB=1.177%  FO=0.0%  mean=88.2ms  p95=101.5ms
VtlWavenet2011:N=200   PR=92.172%  DB=0.105%  FO=0.0%  mean=344.3ms p95=351.7ms
```

**This result is methodologically clean.** Every constraint is respected:
- Non-selected ODs use ECMP background.
- DQN scope is strictly enforced.
- No hidden overrides.
- Audit 13/13 PASS.

**However, the result is numerically rejected** by the professor because PR has dropped
significantly compared to the earlier (methodologically invalid) run. The target is to beat
FlexDATE on as many of the 4 reference topologies as possible (ideally 4/4 WIN BOTH),
while maintaining full methodological compliance.

**Do not revert to hidden automatic full-OD fallback. That path is permanently closed.**

---

## 3. Main Files

```
scripts/phase1_5/gnn_lpd_dqn_selective_db_lp.py
    Main eval/train script. Step function, DQN env, LP calls, per_cycle writer.
    Contains the corrected _base_splits(), _background_mode(), and step() DQN scope.

scripts/phase1_5/audit_gnn_lpd_dqn_clean_method.py
    13-block professor-compliance audit. Must pass CLEAN METHOD AUDIT PASSED.

scripts/phase1_5/windows_smoke_test.py
    Quick artifact/import check. Must pass READY FOR STUDENT RUN.

phase1_reactive/drl/gnn_selector.py
    GNNFlowSelector model + build_od_features. Internal names are now path_cost_demand_scores,
    path_cost_demand_norm, w_path_cost (all flexdate_* renamed).

scripts/phase1_5/gnn_lp_inference.py
    Wraps GNNFlowSelector inference (score_lp_gnn_cycle). Called by main eval script.

scripts/phase1_5/generate_eval_diagnostics.py
    Generates decision_time_diagnostics.csv and action_time_diagnostics.csv from per_cycle.csv.

results/gnn_lpd_dqn_selective_db_lp/final_N3976/
    Eval output directory. Contains per_cycle.csv, per_topology_summary.csv,
    overall.json, method_audit.json, decision_time_diagnostics.csv,
    action_time_diagnostics.csv.

results/gnn_lpd_dqn_selective_db_lp/models/gnn_dbbudget_selector.pt
    GNN-LPD selector checkpoint (trained from DB-budgeted oracle labels).

results/gnn_lpd_dqn_selective_db_lp/models/gnn_dbbudget_selector_meta.json
    GNN training provenance: oracle_solver=full_od_db_budgeted_lp,
    heuristic_ranking_used_for_labels=false, db_budgeted_oracle_used=true.

results/gnn_lpd_dqn_selective_db_lp/dqn_best.pt
    DQN policy checkpoint. Trained with the old 9-action space (K30/K40/K50 + FULL_OD).
    Will need retraining if action space changes.

results/gnn_lpd_dqn_selective_db_lp/labels/oracle_labels.csv
    279,360 oracle labels from DB-budgeted full-OD LP.

results/gnn_lpd_dqn_selective_db_lp/labels/label_provenance.json
    Label provenance proof for audit block 7.

results/gnn_lpd_dqn_selective_db_lp/STUDENT_PROFESSOR_FINAL_REPORT.md
    Report with corrected numbers. Must be updated after any new eval.

results/gnn_lpd_dqn_selective_db_lp/STUDENT_PROFESSOR_FINAL_REPORT.docx
    Word version of report.

results/gnn_lpd_dqn_selective_db_lp/STUDENT_PROFESSOR_FINAL_REPORT.pdf
    PDF version of report.
```

---

## 4. What Was Fixed in Commit 8ffc869

### 4.1 ECMP Background Fix

Non-selected / non-critical OD pairs always use ECMP, never previous LP routing.
`prev_splits` is only referenced as the DB-budget baseline inside the LP constraint.

The method keeps these two functions in `gnn_lpd_dqn_selective_db_lp.py`:

```python
def _base_splits(self):
    # Non-selected OD pairs always route on static ECMP.
    # prev_splits is used only as DB-budget reference inside the LP constraint.
    return self.ctx["ecmp"]

def _background_mode(self) -> str:
    return "ecmp"
```

Audit must prove (every row in per_cycle.csv):

```
noncritical_background_mode == "ecmp"
ecmp_background_used == 1
previous_background_used == 0
```

### 4.2 DQN Scope Fix

The hidden `if not accepted:` block that automatically escalated from selected-K LP to
full-OD LP was removed. The old code (now deleted) was:

```python
# OLD — FORBIDDEN — DO NOT RESTORE:
if not accepted:
    full_od_fallback_used = 1
    if fallback_reason != "solver_failed":
        fallback_reason = "pr_failed_after_k_cap"
    for fb_budget in (0.05, 1.0):
        f_splits, f_routing, f_mlu, f_pr, f_status = self._run_lp(
            tm, active, base_splits, fb_budget, 1e-6)
        ...
```

The replacement (now in place):

```python
if not accepted:
    # DQN selected a selected-K action; full-OD override is forbidden.
    selected_od_lp_used = 1
    if fallback_reason not in ("solver_failed",):
        fallback_reason = "selected_k_pr_failed_no_full_override"
    if splits is None:
        splits = clone_splits(base_splits)
        routing = apply_routing(tm, splits, ctx["pl"], ctx["caps"])
        mlu = float(routing.mlu)
        pr = float(min(1.0, ref / mlu)) if (mlu > 0 and ref == ref) else 0.0
    # full_od_fallback_used, full_od_lp_used stay 0 — DQN did not choose full-OD.
```

Full-OD LP is allowed **only** if DQN explicitly selects one of:

```
FULL_OD_FALLBACK_PR_SAFE  (action index 7)
FULL_OD_FALLBACK_LOW_MLU  (action index 8)
```

When selected-K LP fails PR guard and DQN did not choose full-OD, the per-cycle row must record:

```
fallback_reason = selected_k_pr_failed_no_full_override
full_od_fallback_used = 0
full_od_lp_used = 0
selected_od_lp_used = 1
```

Audit checks that enforce this (Blocks 11 and 12):
- `pr_failed_after_k_cap` must never appear in `fallback_reason`.
- `full_od_lp_used == 1` implies `dqn_action` is `FULL_OD_FALLBACK_PR_SAFE` or `FULL_OD_FALLBACK_LOW_MLU`.

### 4.3 FlexDATE Helper Rename

Internal `flexdate_score` naming was removed from `phase1_reactive/drl/gnn_selector.py`.

Renamed in code:

```
flexdate_scores       → path_cost_demand_scores
flexdate_norm         → path_cost_demand_norm
flexdate (local var)  → path_cost_scores
flex_norm             → path_cost_norm
w_flex                → w_path_cost
"w_flexdate" (key)    → "w_path_cost"
```

The all-caps constant `FLEXDATE` used for external baseline comparison data (in
`gnn_lpd_dqn_selective_db_lp.py` lines 164, 932–933) is NOT a variable name and must
be preserved — it refers to the external FlexDATE baseline method, not an internal feature.

Audit Block 13 checks that `flexdate_scores`, `flexdate_norm`, and `w_flexdate` are absent
from both `gnn_lpd_dqn_selective_db_lp.py` and `gnn_selector.py`.

---

## 5. Why the Old High PR Result Was Invalid

The old eval (pre-8ffc869) reported PR ≈ 99.743% across N=3976. This was inflated by a
hidden automatic full-OD override that activated whenever selected-K LP failed the PR guard.

**Evidence from Tiscali (most extreme case):**

```
rows = 200
active_od_count = 2352 (large dense topology)
full_od_fallback_used = 163 / 200 = 81.5%
selected_od_lp_used = 37 / 200 = 18.5%
mean decision_ms = 4,340.39 ms
P95 decision_ms = 9,255.55 ms
```

**DQN actions chosen for Tiscali (old eval):**

```
OPTIMIZE_K40_DB_0.03    153 cycles
OPTIMIZE_K30_DB_0.01     30 cycles
OPTIMIZE_K30_DB_0.03     16 cycles
OPTIMIZE_K40_DB_0.01      1 cycle
Full-OD action (7 or 8)   0 cycles
```

**The contradiction:**
- DQN selected a selected-K action in 200/200 cycles.
- DQN selected a full-OD action in 0/200 cycles.
- But the code ran full-OD LP in 163/200 cycles anyway.

The hidden `if not accepted:` block overrode the DQN's own decision 163 times. This is
methodologically invalid: the DQN did not "choose" full-OD, but the solver secretly used
full-OD without authorization. The inflated PR (≈100% for Tiscali, ≈99.9% pooled) was
entirely a result of this unauthorized override.

The same pattern appeared on CERNET and Sprintlink (k_escalation_rate=100%, K-cap binding
every cycle, hidden FO override executing silently).

After removing the override (commit 8ffc869), honest PR is 71–87% on dense topologies.

---

## 6. Professor Rejection Problem

The professor rejected the corrected result because the PR is too low to be useful.
The task is to recover PR while keeping the method strictly clean.

**Target:**
- Beat FlexDATE on 4/5 topologies (WIN BOTH PR and DB) if possible.
- Recover PR.
- Keep DB low (below FlexDATE DB on all reference topologies).
- Keep decision time reasonable (mean < 500 ms on most topologies).
- Maintain full audit compliance.

**FlexDATE reference targets (source-locked):**

```
Abilene:    PR ≥ 0.958,  DB ≤ 0.0513   (current: PR=97.157% WIN, DB=0.891% WIN)
CERNET:     PR ≥ 0.975,  DB ≤ 0.0183   (current: PR=71.256% LOSS, DB=0.806% WIN)
GEANT:      PR ≥ 0.995,  DB ≤ 0.0296   (current: PR=99.153% LOSS by 0.347pp, DB=0.471% WIN)
Sprintlink: PR ≥ 0.999,  DB ≤ 0.0510   (current: PR=86.961% LOSS, DB=0.203% WIN)
Tiscali:    No source-locked FlexDATE reference. Do not fabricate a win/loss claim.
```

**Current win/loss summary:**
- Abilene: WIN BOTH (already passing)
- GEANT: Very close — PR=99.153% vs target 99.500%. Gap = 0.347 percentage points.
- CERNET: Large gap — PR=71.256% vs target 97.500%. Gap = 26.2pp.
- Sprintlink: Large gap — PR=86.961% vs target 99.900%. Gap = 12.9pp.

GEANT is the easiest to recover (tiny gap). CERNET and Sprintlink need dramatically higher
selected-K coverage to fix.

---

## 7. Correct Next Strategy

**Do not reintroduce hidden fallback.**

The root cause of PR shortfall on dense topologies is that K=30/40/50 is too small to cover
critical ODs on topologies with 1,640–2,352 OD pairs. With DQN-scope enforcement, the K-cap
is binding on every cycle, and the LP cannot reach PR target with only 30–50 selected ODs.

**Recommended strategy: Option C — Expand explicit DQN action space**

Add larger selected-K actions as explicit DQN choices. The DQN will then learn to choose
the right K budget for each topology. Full-OD remains available as an explicit DQN choice
(actions 7 and 8) for cases where even K=160 is insufficient.

**Proposed expanded action space (example):**

```
Action 0:  KEEP_PREVIOUS_ROUTING
Action 1:  OPTIMIZE_K30_DB_0.01
Action 2:  OPTIMIZE_K30_DB_0.03
Action 3:  OPTIMIZE_K50_DB_0.01
Action 4:  OPTIMIZE_K50_DB_0.03
Action 5:  OPTIMIZE_K80_DB_0.01
Action 6:  OPTIMIZE_K80_DB_0.03
Action 7:  OPTIMIZE_K120_DB_0.01
Action 8:  OPTIMIZE_K120_DB_0.03
Action 9:  FULL_OD_FALLBACK_PR_SAFE
Action 10: FULL_OD_FALLBACK_LOW_MLU
```

Or with K160 for very dense topologies:

```
Action 11: OPTIMIZE_K160_DB_0.01
Action 12: OPTIMIZE_K160_DB_0.03
```

**Important constraints:**
- Larger-K actions must be explicit DQN actions (not automatic escalation).
- Full-OD (action 9 or 10) still requires explicit DQN selection.
- Non-selected ODs still always use ECMP.
- No hidden override of any kind.
- If the DQN network dimension changes (new action count), the DQN must be retrained.

**Alternative if action space change is risky:**
Remap the existing K30/K40/K50 action integers to larger K values (e.g., K50→K80,
K40→K120) without changing the total action count. This avoids retraining the DQN network
architecture but changes what the existing actions mean. Document this clearly in the report.

---

## 8. Required Diagnosis Before Changing Action Space

Run this diagnostic before making any changes:

```bash
cd /Users/moahaimentalib/Desktop/f_flex_network_code_clean
source .venv/bin/activate || true

python3 - <<'PY'
import pandas as pd
import numpy as np

p = "results/gnn_lpd_dqn_selective_db_lp/final_N3976/per_cycle.csv"
df = pd.read_csv(p)

print("Columns:")
print(df.columns.tolist())

print("\nPer-topology corrected results:")
print(df.groupby("topology").agg(
    n=("topology", "size"),
    pr=("feat_PR", "mean"),
    db=("chosen_disturbance", "mean"),
    min_pr=("feat_PR", "min"),
    pr95=("feat_PR", lambda x: (x >= .95).mean()),
    mean_ms=("decision_ms", "mean"),
    p50_ms=("decision_ms", lambda x: x.quantile(.50)),
    p95_ms=("decision_ms", lambda x: x.quantile(.95)),
    max_ms=("decision_ms", "max"),
    full_od=("full_od_lp_used", "mean"),
    selected=("selected_od_lp_used", "mean"),
).to_string())

print("\nAction distribution:")
if "dqn_action" in df.columns:
    print(df.groupby(["topology", "dqn_action"]).size().to_string())
elif "action_name" in df.columns:
    print(df.groupby(["topology", "action_name"]).size().to_string())

print("\nFallback reason distribution:")
if "fallback_reason" in df.columns:
    print(df.groupby(["topology", "fallback_reason"]).size().to_string())

print("\nK-escalation rate by topology:")
if "k_escalation_used" in df.columns:
    print(df.groupby("topology")["k_escalation_used"].mean().to_string())
PY
```

From this output, identify:
- Which topologies have `k_escalation_rate = 1.0` (K-cap binding every cycle).
- What `final_selected_k` is relative to `active_od_count` on those topologies.
- Whether the PR shortfall is entirely due to K-cap (expected: yes on CERNET, Sprintlink, Tiscali).

---

## 9. Required Implementation Direction

**File to modify:**

```
scripts/phase1_5/gnn_lpd_dqn_selective_db_lp.py
```

**Tasks (in order):**

1. **Add larger selected-K explicit actions.** Update `ACTION_CONFIG` and `ACTION_NAMES`
   dicts at the top of the file. Add entries for K80, K120 (and optionally K160).
   Example structure for ACTION_CONFIG:
   ```python
   ACTION_CONFIG = {
       0: ("keep", None, None),
       1: ("selected", 30, 0.01),
       2: ("selected", 30, 0.03),
       3: ("selected", 50, 0.01),
       4: ("selected", 50, 0.03),
       5: ("selected", 80, 0.01),
       6: ("selected", 80, 0.03),
       7: ("selected", 120, 0.01),
       8: ("selected", 120, 0.03),
       9: ("full",   None, 0.05),   # FULL_OD_FALLBACK_PR_SAFE
       10: ("full",  None, 1.00),   # FULL_OD_FALLBACK_LOW_MLU
   }
   ```

2. **Update DQN network output dimension.** If action count changes from 9 to 11 (or more),
   update `n_actions` in the DQN network construction and training loop. The DQN linear
   output layer must match.

3. **Retrain DQN checkpoint.** With a new action space, `dqn_best.pt` is invalid.
   Run training mode:
   ```bash
   python scripts/phase1_5/gnn_lpd_dqn_selective_db_lp.py --mode train
   ```
   This will produce a new `dqn_best.pt`. Training topology order matches the current
   config (Abilene, CERNET, GEANT, Ebone).

4. **Re-run N=3976 evaluation:**
   ```bash
   python scripts/phase1_5/gnn_lpd_dqn_selective_db_lp.py --mode eval --tag final_N3976
   ```

5. **Keep strict no-hidden-fallback logic** (the corrected `step()` function from commit
   8ffc869 must not be reverted).

6. **Generate diagnostics:**
   ```bash
   python scripts/phase1_5/generate_eval_diagnostics.py \
       --eval_dir results/gnn_lpd_dqn_selective_db_lp/final_N3976
   ```

7. **Update audit for new actions** in `audit_gnn_lpd_dqn_clean_method.py`:
   - Update `FULL_OD_ACTIONS` set in Block 12 if action names change.
   - Keep all 13 blocks intact.

8. **Update report** `results/gnn_lpd_dqn_selective_db_lp/STUDENT_PROFESSOR_FINAL_REPORT.md`
   with new numbers from eval. Update Tables 2, 3, 4, 8, and Final Safe Claims.

---

## 10. Required Final Output Files

Final regenerated folder:

```
results/gnn_lpd_dqn_selective_db_lp/final_N3976/
```

Must include:

```
per_cycle.csv
per_topology_summary.csv
overall.json
method_audit.json
decision_time_diagnostics.csv
action_time_diagnostics.csv
```

`per_cycle.csv` must include at minimum these columns:

```
topology
timestep
feat_PR
chosen_disturbance
decision_ms
action
dqn_action
initial_selected_k
final_selected_k
selected_k
selected_od_count
active_od_count
full_od_lp_used
full_od_fallback_used
selected_od_lp_used
fallback_reason
noncritical_background_mode
ecmp_background_used
previous_background_used
criticality_backend
heuristic_used
random_forest_gate_used
sticky_gate_used
stage2_used
disturbance_finalization_used
```

---

## 11. Required Audit Guarantees

The final audit (`audit_gnn_lpd_dqn_clean_method.py`) must prove all 13 blocks:

```
Block 1:  Required output files present
Block 2:  Per-cycle CSV non-empty
Block 3:  Per-row component flags:
              gnn_used == 1
              lpd_used == 1
              dqn_used == 1
              stage2_used == 0
              disturbance_finalization_used == 0
              random_forest_gate_used == 0
              sticky_gate_used == 0
              heuristic_used == 0
              ecmp_background_used == 1
              previous_background_used == 0
Block 4:  criticality_backend == "gnn_lpd" in every row
Block 5:  method_audit.json flags correct
Block 6:  GNN checkpoint meta — DB-budgeted oracle provenance
Block 7:  Label provenance JSON correct
Block 8:  Static grep for forbidden tokens in method script
Block 9:  Hardcoded-result check
Block 10: FlexDATE comparison (informational, not hard fail)
Block 11: ECMP background mode + dqn_action column present
Block 12: DQN scope enforcement:
              fallback_reason "pr_failed_after_k_cap" absent
              full_od_lp_used==1 only for explicit full-OD DQN actions
Block 13: flexdate internal naming absent from method and selector scripts
```

---

## 12. Validation Commands

Run these before every final commit:

```bash
cd /Users/moahaimentalib/Desktop/f_flex_network_code_clean

# Static scans
grep -RInE "worktree|worktrees|/Users/|contributor" . \
  --include="*.py" --include="*.md" --include="*.json" \
  --exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ || true

grep -RIn "flexdate_scores\|flexdate_norm\|w_flexdate" . \
  --include="*.py" \
  --exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ \
  | grep -v "audit_gnn_lpd" || echo "(none)"

# Smoke test
python scripts/phase1_5/windows_smoke_test.py

# Full audit
python scripts/phase1_5/audit_gnn_lpd_dqn_clean_method.py \
    --eval_dir results/gnn_lpd_dqn_selective_db_lp/final_N3976
```

**Expected output:**

```
READY FOR STUDENT RUN
CLEAN METHOD AUDIT PASSED
```

---

## 13. Commit and Push

After final regeneration:

```bash
cd /Users/moahaimentalib/Desktop/f_flex_network_code_clean

git add \
    scripts/phase1_5/gnn_lpd_dqn_selective_db_lp.py \
    scripts/phase1_5/audit_gnn_lpd_dqn_clean_method.py \
    scripts/phase1_5/gnn_selector.py \
    scripts/phase1_5/gnn_lp_inference.py \
    phase1_reactive/drl/gnn_selector.py \
    README.md README_WINDOWS_RUN.md REPRODUCIBILITY.md

git add -f \
    results/gnn_lpd_dqn_selective_db_lp/final_N3976/per_cycle.csv \
    results/gnn_lpd_dqn_selective_db_lp/final_N3976/per_topology_summary.csv \
    results/gnn_lpd_dqn_selective_db_lp/final_N3976/overall.json \
    results/gnn_lpd_dqn_selective_db_lp/final_N3976/method_audit.json \
    results/gnn_lpd_dqn_selective_db_lp/final_N3976/decision_time_diagnostics.csv \
    results/gnn_lpd_dqn_selective_db_lp/final_N3976/action_time_diagnostics.csv \
    results/gnn_lpd_dqn_selective_db_lp/dqn_best.pt \
    results/gnn_lpd_dqn_selective_db_lp/STUDENT_PROFESSOR_FINAL_REPORT.md

git commit -m "Recover PR with explicit DQN-controlled larger-scope actions"
git push origin main
```

Note: `results/*` is in `.gitignore` (line 16). The final_N3976/ files and other result
artifacts must be staged with `git add -f`. This is the established pattern in this repo.

---

## 14. Mandatory Performance Target — Must Compete with FlexDATE in PR and DB

The corrected strict method at commit `8ffc869` is methodologically valid but numerically
rejected:

```
N = 3976
Mean PR = 93.346%
Mean DB = 0.695%
Mean decision time = 50.9 ms
P95 decision time = 337.2 ms
Full-OD fallback = 0.0%
Audit = 13/13 PASS
```

This result is not acceptable because PR is too low and the method no longer beats FlexDATE
broadly.

**Primary performance target:**

```
Beat FlexDATE in both PR and DB on the comparison topologies.
```

**Known FlexDATE reference targets (source-locked):**

```
Abilene:    FlexDATE PR = 0.958,  DB = 0.0513
CERNET:     FlexDATE PR = 0.975,  DB = 0.0183
GEANT:      FlexDATE PR = 0.995,  DB = 0.0296
Sprintlink: FlexDATE PR = 0.999,  DB = 0.0510
Tiscali:    Source-locked FlexDATE reference is unclear — do not fabricate a FlexDATE value.
```

**Required target logic:**

1. If Tiscali has no valid source-locked FlexDATE reference, evaluate the main FlexDATE win
   claim on the 4 source-locked topologies: Abilene, CERNET, GEANT, Sprintlink.
2. If a valid Tiscali FlexDATE reference exists in the repository/report data, include Tiscali
   and target 4/5 or better.
3. For each topology, report separately:
   ```
   our_PR
   flexdate_PR
   PR_win = our_PR >= flexdate_PR

   our_DB
   flexdate_DB
   DB_win = our_DB <= flexdate_DB
   ```
4. The desired final result is: PR win AND DB win on as many FlexDATE topologies as possible,
   targeting 4/4 source-locked topologies or 4/5 if Tiscali is valid.
5. Do not stop after one low-PR corrected run. Search the compliant design space.

**Recommended search direction:**

- Expand explicit DQN selected-K actions: K80, K120, K160, maybe K200 if runtime permits.
- Retrain DQN with stronger PR-failure penalty.
- Allow explicit full-OD actions only when chosen by DQN.
- Optionally train a PR-risk-aware DQN state so it learns when larger K or full-OD is needed.
- Keep DB penalty in the reward so DB remains below FlexDATE.
- Keep decision time reasonable, but PR and DB wins are the professor's priority.

**Acceptable way to improve:**

```
The DQN explicitly chooses a larger selected-K action or an explicit full-OD action.
```

**Unacceptable way to improve:**

```
The code silently overrides a selected-K action and runs full-OD after PR failure.
```

**Required final comparison output:**

Produce a final comparison table in this format:

```
topology,n,our_PR,flexdate_PR,PR_win,our_DB,flexdate_DB,DB_win,mean_ms,p95_ms,dominant_actions
```

and must clearly state whether the final run beats FlexDATE in:

```
PR only
DB only
both PR and DB
neither
```

for each comparison topology.

**The final report must use only regenerated results from the corrected method.** Do not reuse
old 99% PR numbers unless they are reproduced under the strict DQN-controlled method.

---

## 15. Fresh Clone Validation

After every push:

```bash
cd /Users/moahaimentalib/Desktop
rm -rf f_flex_network_code_test
git clone https://github.com/moahaimen/f_flex_network_code.git f_flex_network_code_test
cd f_flex_network_code_test

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/phase1_5/windows_smoke_test.py
python scripts/phase1_5/audit_gnn_lpd_dqn_clean_method.py \
    --eval_dir results/gnn_lpd_dqn_selective_db_lp/final_N3976
```

**Expected:**

```
READY FOR STUDENT RUN
CLEAN METHOD AUDIT PASSED
```

---

## 16. Absolute Forbidden Shortcuts

Do not do any of the following under any circumstances:

```
Do not restore the hidden selected-K -> full-OD automatic fallback.
Do not reintroduce: if not accepted: ... self._run_lp(tm, active, base_splits, ...) ...
Do not use previous-routing background for non-selected ODs.
Do not return self.prev_splits from _base_splits().
Do not fake or hardcode metrics.
Do not use the old 99% PR result unless it is regenerated under the strict method.
Do not add RandomForest.
Do not add heuristic criticality.
Do not add sticky reuse / sticky gate.
Do not add Stage-2 LP.
Do not add disturbance finalization LP.
Do not use solve_selected_path_lp_min_db.
Do not use internal flexdate_score, flexdate_norm, w_flexdate naming.
Do not add "pr_failed_after_k_cap" as a valid fallback_reason in new code.
Do not add any automated tool, vendor, or system name as contributor or author.
Do not commit local paths (/Users/...) into source files.
Do not commit .venv, __pycache__, or temporary scratch files.
```

---

## 17. Authorship / Metadata Rules (Permanent)

These rules apply to every commit and file in this repo:

1. Do not add any automated tool, vendor, or system name as contributor, author,
   co-author, maintainer, or in any acknowledgement section.
2. Do not add automated-generation attribution comments, tool-brand boilerplate,
   or similar generated-content labels.
3. Do not change the academic authorship of the project.
4. Do not falsify authorship. Use the existing git configuration / repository owner identity.
   If git identity is missing, stop and ask.
5. Keep comments technical only.
6. Do not include hidden metadata, model names, chat transcripts, or prompt files.
7. Do not commit temporary scratch files, failed runs, caches, or private machine-specific files.
8. Do not commit large unnecessary artifacts unless required for reproducibility.

---

## 18. Final Note

The correct solution is to recover PR through **explicit DQN-controlled larger-scope actions**,
or through retraining the DQN policy to learn when to select full-OD actions — not through
hidden solver override.

The key insight: the DQN was trained under a regime where the solver automatically escalated
beyond the DQN's K budget. The DQN never needed to learn "choose full-OD when K=50 is
insufficient" because the solver did it automatically. With scope enforcement, the DQN now
needs either a richer action space (larger K options) or retraining to use explicit full-OD
actions when needed.

GEANT is very close (0.347 pp gap) and may recover with only slightly larger K (K=50 or K=60).
CERNET and Sprintlink have large gaps (26 pp and 13 pp) and likely need K=80–120 or explicit
DQN selection of full-OD actions to recover.

The DB metric is already excellent across all topologies (0.1–1.2%), well below FlexDATE DB
thresholds. Any PR recovery strategy that keeps the LP DB-budget constraint will preserve this.

---

*End of handoff.*
