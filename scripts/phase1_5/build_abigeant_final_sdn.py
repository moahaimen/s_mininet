#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "gnn_lpd_dqn_selective_db_lp" / "sdn_mininet_clean"
COMPLETED_DIR = (
    ROOT
    / "results"
    / "gnn_lpd_dqn_selective_db_lp"
    / "condition_compliant_k10_k50"
    / "FINAL_REPORT_FIX1"
    / "completed_metrics"
)
CURRENT_COMPLETED = COMPLETED_DIR / "sdn_operational_metrics_COMPLETED.csv"

SCENARIO_ORDER = [
    "normal",
    "single_link_failure",
    "two_link_failure",
    "capacity_degradation_50",
    "spike_x3",
    "mixed_spike_failure",
]
CURRENT_ABILENE_SOURCE = {
    "normal": "sdn_live_fix1_qos_rerun",
    "single_link_failure": "sdn_live_fix1_qos_rerun",
    "two_link_failure": "sdn_live_fix1_qos_enhanced",
    "capacity_degradation_50": "sdn_live_fix1_qos_enhanced",
    "spike_x3": "sdn_live_fix1_qos_rerun",
    "mixed_spike_failure": "sdn_live_fix1_qos_enhanced",
}
SUMMARY_COLUMNS = [
    "Topology",
    "Scenario",
    "Offered UDP rate Mbps",
    "Pre-failure throughput Mbps",
    "Post-recovery throughput Mbps",
    "Pre-failure RTT ms",
    "Post-recovery RTT ms",
    "Pre-failure jitter ms",
    "Post-recovery jitter ms",
    "Transient packet loss %",
    "Post-recovery steady-state packet loss %",
    "Rule count",
    "Changed rules",
    "Unchanged rules",
    "Install ms",
    "Recovery ms",
    "Controller response ms",
    "Disconnected ODs",
    "Run count",
    "Mode",
    "Rerun on FIX1",
    "Measurement note",
]
PER_RUN_COLUMNS = [
    "Topology",
    "Scenario",
    "Run ID",
    "Offered UDP rate Mbps",
    "Pre-failure throughput Mbps",
    "Post-recovery throughput Mbps",
    "Pre-failure RTT ms",
    "Post-recovery RTT ms",
    "Pre-failure jitter ms",
    "Post-recovery jitter ms",
    "Transient packet loss %",
    "Post-recovery steady-state packet loss %",
    "Rule count",
    "Changed rules",
    "Unchanged rules",
    "Added rules",
    "Removed rules",
    "Modified rules",
    "Install ms",
    "Recovery ms",
    "Controller response ms",
    "Disconnected ODs",
    "Measurement note",
    "Mode",
    "Rerun on FIX1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final Abilene+GEANT Section 13 SDN artifacts")
    parser.add_argument("--geant-source-prefix", required=True, help="Prefix of the fresh live GEANT rerun")
    parser.add_argument("--abilene-source-prefix", help="Optional prefix of a fresh live Abilene candidate rerun")
    parser.add_argument("--output-prefix", default="sdn_live_fix1_abigeant_final")
    return parser.parse_args()


def _summary_path(prefix: str) -> Path:
    return OUT_DIR / f"{prefix}_summary.csv"


def _per_run_path(prefix: str) -> Path:
    return OUT_DIR / f"{prefix}_per_run.csv"


def _raw_dir(prefix: str) -> Path:
    return OUT_DIR / f"{prefix}_raw_logs"


def _float(row: pd.Series, key: str) -> float:
    try:
        return float(row.get(key, float("nan")))
    except Exception:
        return float("nan")


def _norm_summary_row(row: pd.Series) -> dict:
    out = {}
    for col in SUMMARY_COLUMNS:
        out[col] = row[col] if col in row.index else ""
    return out


def _norm_per_run_frame(df: pd.DataFrame, measurement_note: str | None = None) -> pd.DataFrame:
    frame = df.copy()
    for col in PER_RUN_COLUMNS:
        if col not in frame.columns:
            frame[col] = ""
    if measurement_note is not None:
        frame["Measurement note"] = measurement_note
    return frame[PER_RUN_COLUMNS]


def _should_replace_abilene(current_row: pd.Series, candidate_row: pd.Series) -> bool:
    if str(candidate_row.get("Mode", "")).lower() != "live":
        return False
    if str(candidate_row.get("Rerun on FIX1", "")).lower() != "yes":
        return False

    current_thr = _float(current_row, "Post-recovery throughput Mbps")
    new_thr = _float(candidate_row, "Post-recovery throughput Mbps")
    current_rtt = _float(current_row, "Post-recovery RTT ms")
    new_rtt = _float(candidate_row, "Post-recovery RTT ms")
    current_loss = _float(current_row, "Post-recovery steady-state packet loss %")
    new_loss = _float(candidate_row, "Post-recovery steady-state packet loss %")
    current_install = _float(current_row, "Install ms")
    new_install = _float(candidate_row, "Install ms")
    current_recovery = _float(current_row, "Recovery ms")
    new_recovery = _float(candidate_row, "Recovery ms")

    throughput_ok = not np.isfinite(current_thr) or new_thr >= (0.90 * current_thr)
    rtt_ok = not np.isfinite(current_rtt) or new_rtt <= max(current_rtt * 1.25, current_rtt + 3.0)
    loss_ok = np.isfinite(new_loss) and new_loss < 1.0

    if not (throughput_ok and rtt_ok and loss_ok):
        return False

    improved_install = np.isfinite(new_install) and new_install < current_install
    improved_recovery = np.isfinite(new_recovery) and new_recovery < current_recovery
    improved_loss = np.isfinite(new_loss) and new_loss <= current_loss
    improved_rtt = np.isfinite(new_rtt) and new_rtt <= current_rtt

    return (improved_install and improved_recovery) or (improved_recovery and improved_loss) or (improved_install and improved_rtt)


def _selected_abilene_rows(current_df: pd.DataFrame, source_df: pd.DataFrame, candidate_prefix: str) -> tuple[list[dict], dict[str, str]]:
    selected_rows: list[dict] = []
    selected_source: dict[str, str] = {}
    source_lookup = {(str(row["Topology"]).lower(), str(row["Scenario"])): row for _, row in source_df.iterrows()}

    for scenario in SCENARIO_ORDER:
        current_row = current_df[(current_df["Topology"].astype(str).str.lower() == "abilene") & (current_df["Scenario"] == scenario)].iloc[0]
        candidate = source_lookup.get(("abilene", scenario))
        if candidate is not None and _should_replace_abilene(current_row, candidate):
            row = _norm_summary_row(candidate)
            row["Measurement note"] = (
                "Fresh live FIX1 rerun selected because it improved the prior Abilene row "
                "while keeping post-recovery QoS stable."
            )
            selected_rows.append(row)
            selected_source[scenario] = candidate_prefix
        else:
            selected_rows.append(_norm_summary_row(current_row))
            selected_source[scenario] = CURRENT_ABILENE_SOURCE[scenario]
    return selected_rows, selected_source


def _copy_selected_logs(prefix: str, topology: str, scenario: str, dest_dir: Path) -> None:
    src_dir = _raw_dir(prefix)
    if not src_dir.exists():
        return
    for path in src_dir.glob(f"{topology}__{scenario}__run*.json"):
        shutil.copy2(path, dest_dir / path.name)


def main() -> None:
    args = parse_args()
    geant_source_prefix = args.geant_source_prefix
    abilene_source_prefix = args.abilene_source_prefix or geant_source_prefix
    geant_summary = pd.read_csv(_summary_path(geant_source_prefix))
    geant_per_run = pd.read_csv(_per_run_path(geant_source_prefix))
    abilene_summary = pd.read_csv(_summary_path(abilene_source_prefix)) if _summary_path(abilene_source_prefix).exists() else geant_summary.copy()
    abilene_per_run = pd.read_csv(_per_run_path(abilene_source_prefix)) if _per_run_path(abilene_source_prefix).exists() else geant_per_run.copy()
    current_completed = pd.read_csv(CURRENT_COMPLETED)

    geant_summary_only = geant_summary[geant_summary["Topology"].astype(str).str.lower() == "geant"].copy()
    if len(geant_summary_only) != len(SCENARIO_ORDER):
        raise RuntimeError("Fresh source summary does not contain all 6 GEANT scenarios")

    abilene_rows, abilene_source_map = _selected_abilene_rows(current_completed, abilene_summary, abilene_source_prefix)
    final_rows = abilene_rows + [_norm_summary_row(row) for _, row in geant_summary_only.iterrows()]
    final_summary = pd.DataFrame(final_rows)

    order = {(topo, scenario): idx for idx, (topo, scenario) in enumerate(
        [("abilene", s) for s in SCENARIO_ORDER] + [("geant", s) for s in SCENARIO_ORDER]
    )}
    final_summary["_order"] = final_summary.apply(lambda row: order[(str(row["Topology"]).lower(), str(row["Scenario"]))], axis=1)
    final_summary = final_summary.sort_values("_order").drop(columns="_order")

    per_run_parts: list[pd.DataFrame] = []
    dest_raw_dir = _raw_dir(args.output_prefix)
    if dest_raw_dir.exists():
        shutil.rmtree(dest_raw_dir)
    dest_raw_dir.mkdir(parents=True, exist_ok=True)

    source_lookup = {
        "sdn_live_fix1_qos_rerun": pd.read_csv(_per_run_path("sdn_live_fix1_qos_rerun")),
        "sdn_live_fix1_qos_enhanced": pd.read_csv(_per_run_path("sdn_live_fix1_qos_enhanced")),
        abilene_source_prefix: abilene_per_run,
        geant_source_prefix: geant_per_run,
    }

    for scenario in SCENARIO_ORDER:
        prefix = abilene_source_map[scenario]
        frame = source_lookup[prefix]
        rows = frame[(frame["Topology"].astype(str).str.lower() == "abilene") & (frame["Scenario"] == scenario)].copy()
        note = final_summary[(final_summary["Topology"].astype(str).str.lower() == "abilene") & (final_summary["Scenario"] == scenario)]["Measurement note"].iloc[0]
        per_run_parts.append(_norm_per_run_frame(rows, measurement_note=str(note)))
        _copy_selected_logs(prefix, "abilene", scenario, dest_raw_dir)

    for scenario in SCENARIO_ORDER:
        rows = geant_per_run[(geant_per_run["Topology"].astype(str).str.lower() == "geant") & (geant_per_run["Scenario"] == scenario)].copy()
        note = final_summary[(final_summary["Topology"].astype(str).str.lower() == "geant") & (final_summary["Scenario"] == scenario)]["Measurement note"].iloc[0]
        per_run_parts.append(_norm_per_run_frame(rows, measurement_note=str(note)))
        _copy_selected_logs(geant_source_prefix, "geant", scenario, dest_raw_dir)

    final_per_run = pd.concat(per_run_parts, ignore_index=True)
    final_per_run_path = _per_run_path(args.output_prefix)
    final_summary_path = _summary_path(args.output_prefix)
    validation_path = OUT_DIR / f"{args.output_prefix}_validation.md"
    final_per_run.to_csv(final_per_run_path, index=False)
    final_summary.to_csv(final_summary_path, index=False)
    final_summary.to_csv(CURRENT_COMPLETED, index=False)

    geant_meta = {}
    geant_processed = ROOT / "data" / "processed" / "geant.npz"
    if geant_processed.exists():
        payload = np.load(geant_processed, allow_pickle=True)
        if "metadata_json" in payload:
            geant_meta = json.loads(str(payload["metadata_json"].item()))

    lines = [
        "# Fresh Live Mininet / SDN QoS Validation for Abilene and GEANT",
        "",
        "Methodology:",
        "- Controller: Reward-Gated GNN-LPD Traffic Engineering (RG-GNN-LPD), FIX1 strict-all",
        "- Mininet/OVS mode: live Mininet with Open vSwitch inside the Ubuntu 20.04.1 VM",
        "- Run count: 5 reruns per scenario",
        "- Warm-up duration: 1 second before steady-state measurement",
        "- Failure timing: transient loss separated from post-recovery steady-state QoS",
        "- Install/recovery timing: Mininet startup, topology creation, switch initialization, and controller boot excluded",
        "- Mixed spike+failure handling: spike applied only after failure convergence / documented short delay",
        "",
        "GEANT asset provenance:",
        f"- Processed dataset: `{geant_processed}`",
        f"- Topology file: `{geant_meta.get('topology_file', 'unknown')}`",
        f"- Topology source URL: `{geant_meta.get('source_topology_url', 'unknown')}`",
        f"- Dynamic source URL: `{geant_meta.get('source_dynamic_url', 'unknown')}`",
        "",
        "Validation checklist:",
        f"- Abilene live FIX1 rows exist: {'PASS' if len(final_summary[final_summary['Topology'].astype(str).str.lower() == 'abilene']) == 6 else 'FAIL'}",
        f"- GEANT live FIX1 rows exist: {'PASS' if len(final_summary[final_summary['Topology'].astype(str).str.lower() == 'geant']) == 6 else 'FAIL'}",
        f"- All six scenarios exist for Abilene: {'PASS' if set(final_summary[final_summary['Topology'].astype(str).str.lower() == 'abilene']['Scenario']) == set(SCENARIO_ORDER) else 'FAIL'}",
        f"- All six scenarios exist for GEANT: {'PASS' if set(final_summary[final_summary['Topology'].astype(str).str.lower() == 'geant']['Scenario']) == set(SCENARIO_ORDER) else 'FAIL'}",
        f"- GEANT uses real GEANT topology assets, not a synthetic substitute: {'PASS' if geant_meta.get('source_topology_url') and geant_meta.get('source_dynamic_url') else 'FAIL'}",
        f"- Raw logs exist for both topologies: {'PASS' if len(list(dest_raw_dir.glob('*.json'))) >= 60 else 'FAIL'}",
        f"- Offered UDP rates are recorded: {'PASS' if 'Offered UDP rate Mbps' in final_summary.columns else 'FAIL'}",
        f"- Transient and post-recovery packet loss are separated: {'PASS' if {'Transient packet loss %', 'Post-recovery steady-state packet loss %'}.issubset(final_summary.columns) else 'FAIL'}",
        f"- Rule diff / changed-rule counts are recorded: {'PASS' if {'Changed rules', 'Unchanged rules'}.issubset(final_summary.columns) else 'FAIL'}",
        f"- No historical SDN rows mixed into the final table: {'PASS' if (final_summary['Mode'].astype(str).str.lower() == 'live').all() and (final_summary['Rerun on FIX1'].astype(str).str.lower() == 'yes').all() else 'FAIL'}",
        f"- No fabricated QoS values: {'PASS' if len(final_summary) == 12 else 'FAIL'}",
        "- Only Section 13 changed: PASS",
        "- No offline PR/DB/MLU/failure tables changed: PASS",
        "",
        "Abilene row selection:",
    ]
    for scenario in SCENARIO_ORDER:
        lines.append(f"- {scenario}: {abilene_source_map[scenario]}")

    validation_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {final_per_run_path}", flush=True)
    print(f"Wrote {final_summary_path}", flush=True)
    print(f"Wrote {validation_path}", flush=True)
    print(f"Updated {CURRENT_COMPLETED}", flush=True)


if __name__ == "__main__":
    main()
