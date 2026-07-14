# Final SDN Execution Config (Section-13 selected rows)

This records the exact routing/install implementation and environment flags used to produce each
**selected** Section-13 row. It exists because the optimized routing and install paths are **env-gated and
OFF by default** - they are NOT the code default and must be explicitly enabled.

## Code defaults (runner `run_sdn_mininet_clean.py`)
| Flag | Default | Meaning |
| --- | --- | --- |
| `TARGET2_SWITCH_FAIL_MODE` | **`secure`** | Target-2 fix (drop-on-miss, no NORMAL mesh flood). ON by default. |
| `TARGET2_CAPACITY_INPLACE` | **`1`** | In-place `tc class change` capacity update. ON by default. |
| `TARGET5_USE_OPT_ROUTING` | **`0` (OFF)** | Optimized CBC-MPS routing solver B. **Not default.** |
| `TARGET6_OPT_INSTALL` | **`0` (OFF)** | Optimized atomic `--bundle` per-switch install B. **Not default.** |
| `TARGET6_INSTALL_PARALLEL` | `32` | Parallel switches for the optimized install (only used when `TARGET6_OPT_INSTALL=1`). |

The trusted routing solver (`te.lp_solver.solve_selected_path_lp_dbbudget`) and the trusted incremental
install remain the **default** path. Optimized routing B and optimized install B are opt-in.

## Selected-row execution config
| Final selected scenario | Routing implementation | Install implementation | Required environment/config flags |
| --- | --- | --- | --- |
| abilene normal (retained trusted) | trusted LP solver A | trusted incremental | none (code defaults) |
| abilene single_link_failure (retained trusted) | trusted LP solver A | trusted incremental | none (code defaults) |
| abilene two_link_failure (Target 1) | trusted LP solver A | trusted incremental | none (code defaults; `secure`/in-place are defaults) |
| abilene capacity_degradation_50 (Target 2) | trusted LP solver A | trusted incremental | none (code defaults; `TARGET2_CAPACITY_INPLACE=1` and `secure` are defaults) |
| abilene spike_x3 (retained trusted) | trusted LP solver A | trusted incremental | none (code defaults) |
| abilene mixed_spike_failure (Target 3) | trusted LP solver A | trusted incremental | none (code defaults) |
| geant normal (Target 5) | **optimized routing B** | trusted incremental | **`TARGET5_USE_OPT_ROUTING=1`** |
| geant single_link_failure (Target 4) | trusted LP solver A | trusted incremental | none (code defaults) |
| geant two_link_failure (retained trusted) | trusted LP solver A | trusted incremental | none (code defaults) |
| geant capacity_degradation_50 (retained trusted) | trusted LP solver A | trusted incremental | none (code defaults) |
| geant spike_x3 (Target 6) | **optimized routing B** | **optimized bundle+parallel install B** | **`TARGET5_USE_OPT_ROUTING=1` + `TARGET6_OPT_INSTALL=1` + `TARGET6_INSTALL_PARALLEL=32`** |
| geant mixed_spike_failure (retained trusted) | trusted LP solver A | trusted incremental | none (code defaults) |

## Reproducibility notes
- `db_weight=1e-6` is the trusted live semantics.
- Optimized routing B: `te/lp_solver_opt.py`; gate: `scripts/phase1_5/target5_ab_gate.py`.
- Optimized install B: `_ovs_bundle_apply` + parallel path in the runner; gate: `scripts/phase1_5/target6_flow_install_ab.py`.
- **The optimized paths are OFF by default.** These final rows are not the default trusted path; they require the flags above.
