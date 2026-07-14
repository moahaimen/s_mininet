#!/usr/bin/env python3
"""Reconstruct the HARDFIX aggregated CSV artifacts from raw JSON logs.

This is a deterministic re-implementation of the (missing) original HARDFIX
aggregator. The PRIMARY METRIC SOURCE is the raw per-run JSON logs; no metric
value is hard-coded and the trusted CSVs are NOT read as inputs. Aggregation
semantics are documented in HARDFIX_AGGREGATION_SEMANTICS.md and were taken
from the authoritative runner (run_sdn_mininet_clean.py).

Raw-evidence gap: "Logging ms" is stored as 0.0 inside the raw JSON's
timing_breakdown_ms (the real logging duration was written only to a flat field
that the JSON does not persist). It is emitted as 0.0 from the raw logs and not
fabricated. All experiment-meaningful metrics reconstruct exactly.

Usage:
    python build_final_hardfix_sdn_artifacts.py \
        --abilene-raw <dir> --geant-raw <dir> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # silence all-NaN slice RuntimeWarnings

NON_FAILURE = {"normal", "spike_x3"}
SCENARIO_ORDER = [
    "normal",
    "single_link_failure",
    "two_link_failure",
    "capacity_degradation_50",
    "spike_x3",
    "mixed_spike_failure",
]
TOPOLOGY_ORDER = ["abilene", "geant"]

TIMING_FIELDS = [
    ("event_detection_ms", "Event detection ms"),
    ("state_snapshot_ms", "State snapshot ms"),
    ("ddqn_inference_ms", "DDQN inference ms"),
    ("scorer_ranking_ms", "Scorer ranking ms"),
    ("routing_compute_ms", "Routing compute ms"),
    ("rule_diff_build_ms", "Rule diff build ms"),
    ("flow_mod_send_ms", "Flow-mod send ms"),
    ("barrier_wait_ms", "Barrier wait ms"),
    ("probe_wait_ms", "Probe wait ms"),
    ("logging_ms", "Logging ms"),
]

PER_RUN_LOWER = [
    "topology", "scenario", "run_id", "offered_udp_rate_mbps",
    "pre_failure_throughput_mbps", "post_recovery_throughput_mbps",
    "pre_failure_rtt_ms", "post_recovery_rtt_ms",
    "pre_failure_jitter_ms", "post_recovery_jitter_ms",
    "transient_packet_loss_pct", "post_recovery_packet_loss_pct",
    "affected_ods", "rule_count", "changed_rules", "unchanged_rules",
    "added_rules", "removed_rules", "modified_rules",
    "install_ms", "initial_install_ms", "incremental_install_ms", "recovery_ms",
    "controller_pipeline_ms", "controller_decision_ms",
    "legacy_controller_response_ms", "controller_response_ms",
    "disconnected_ods", "candidate_path_exhausted_ods",
    "event_detection_ms", "state_snapshot_ms", "ddqn_inference_ms",
    "scorer_ranking_ms", "routing_compute_ms", "rule_diff_build_ms",
    "flow_mod_send_ms", "barrier_wait_ms", "probe_wait_ms", "logging_ms",
    "controller_timing_note", "precomputed_backup_plan",
    "post_failure_bottleneck_capacity_mbps", "rate_cap_used",
    "measurement_note", "mode",
]
PER_RUN_RENAME = {
    "topology": "Topology", "scenario": "Scenario", "run_id": "Run ID",
    "offered_udp_rate_mbps": "Offered UDP rate Mbps",
    "pre_failure_throughput_mbps": "Pre-failure throughput Mbps",
    "post_recovery_throughput_mbps": "Post-recovery throughput Mbps",
    "pre_failure_rtt_ms": "Pre-failure RTT ms",
    "post_recovery_rtt_ms": "Post-recovery RTT ms",
    "pre_failure_jitter_ms": "Pre-failure jitter ms",
    "post_recovery_jitter_ms": "Post-recovery jitter ms",
    "transient_packet_loss_pct": "Transient packet loss %",
    "post_recovery_packet_loss_pct": "Post-recovery steady-state packet loss %",
    "affected_ods": "Affected ODs", "rule_count": "Rule count",
    "changed_rules": "Changed rules", "unchanged_rules": "Unchanged rules",
    "added_rules": "Added rules", "removed_rules": "Removed rules",
    "modified_rules": "Modified rules", "install_ms": "Install ms",
    "initial_install_ms": "Initial install ms",
    "incremental_install_ms": "Incremental install ms",
    "recovery_ms": "Recovery ms",
    "controller_pipeline_ms": "Controller pipeline ms",
    "controller_decision_ms": "Controller decision ms",
    "legacy_controller_response_ms": "Legacy controller response ms",
    "controller_response_ms": "Controller response ms",
    "disconnected_ods": "Disconnected ODs",
    "candidate_path_exhausted_ods": "Candidate-path exhausted ODs",
    "event_detection_ms": "Event detection ms",
    "state_snapshot_ms": "State snapshot ms",
    "ddqn_inference_ms": "DDQN inference ms",
    "scorer_ranking_ms": "Scorer ranking ms",
    "routing_compute_ms": "Routing compute ms",
    "rule_diff_build_ms": "Rule diff build ms",
    "flow_mod_send_ms": "Flow-mod send ms",
    "barrier_wait_ms": "Barrier wait ms", "probe_wait_ms": "Probe wait ms",
    "logging_ms": "Logging ms", "controller_timing_note": "Controller timing note",
    "precomputed_backup_plan": "Precomputed backup plan",
    "post_failure_bottleneck_capacity_mbps": "Post-failure bottleneck capacity Mbps",
    "rate_cap_used": "Rate cap used", "measurement_note": "Measurement note",
    "mode": "Mode",
}
INT_COLS = {
    "Run ID", "Affected ODs", "Rule count", "Changed rules", "Unchanged rules",
    "Added rules", "Removed rules", "Modified rules", "Disconnected ODs",
    "Candidate-path exhausted ODs",
}


def write_csv(df: pd.DataFrame, path: Path) -> None:
    Path(path).write_bytes(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def _f(x):
    return np.nan if x is None else float(x)


def _nansum(vals):
    return float(np.nansum([_f(v) for v in vals])) if vals else 0.0


def _nanmean(vals):
    arr = [_f(v) for v in vals]
    return float(np.nanmean(arr)) if arr else float("nan")


def _pair_mean_over_ods(lst, a, b):
    return _nanmean([np.nanmean([_f(e.get(a)), _f(e.get(b))]) for e in lst]) if lst else float("nan")


def record_from_json(d: dict) -> dict:
    pre = d.get("pre_failure") or []
    post = d.get("post_recovery") or []
    tr = d.get("transient_recovery") or []
    tb = d.get("timing_breakdown_ms") or {}
    scen = d["scenario"]
    rec = {
        "topology": d["topology"],
        "scenario": scen,
        "run_id": int(d["run_id"]),
        "offered_udp_rate_mbps": round(_f(d.get("offered_udp_rate_mbps")), 3),
        "pre_failure_throughput_mbps": round(_nansum([e.get("throughput_mbps") for e in pre]), 3),
        "post_recovery_throughput_mbps": round(_nansum([e.get("throughput_mbps") for e in post]), 3),
        "pre_failure_rtt_ms": round(_nanmean([e.get("ping_rtt_ms") for e in pre]), 3),
        "post_recovery_rtt_ms": round(_nanmean([e.get("ping_rtt_ms") for e in post]), 3),
        "pre_failure_jitter_ms": round(_pair_mean_over_ods(pre, "iperf_jitter_ms", "ping_jitter_ms"), 3),
        "post_recovery_jitter_ms": round(_pair_mean_over_ods(post, "iperf_jitter_ms", "ping_jitter_ms"), 3),
        "transient_packet_loss_pct": round(_nanmean([e.get("transient_packet_loss_pct") for e in tr]), 4) if tr else 0.0,
        "post_recovery_packet_loss_pct": round(_pair_mean_over_ods(post, "iperf_loss_pct", "ping_loss_pct"), 4),
        "affected_ods": int(d.get("affected_ods", 0)),
        "rule_count": int(d.get("rule_count", 0)),
        "changed_rules": int(d.get("changed_rules", 0)),
        "unchanged_rules": int(d.get("unchanged_rules", 0)),
        "added_rules": int(d.get("added_rules", 0)),
        "removed_rules": int(d.get("removed_rules", 0)),
        "modified_rules": int(d.get("modified_rules", 0)),
        "install_ms": round(_f(d.get("install_ms")), 3),
        "initial_install_ms": round(_f(d.get("initial_install_ms")), 3),
        "incremental_install_ms": round(_f(d.get("incremental_install_ms")), 3),
        "recovery_ms": round(_f(d.get("recovery_ms")), 3),
        "controller_pipeline_ms": round(_f(d.get("controller_pipeline_ms")), 3),
        "controller_decision_ms": round(_f(d.get("controller_decision_ms")), 3),
        "legacy_controller_response_ms": round(_f(d.get("legacy_controller_response_ms")), 3),
        "controller_response_ms": round(_f(d.get("controller_response_ms")), 3),
        "disconnected_ods": int(d.get("disconnected_ods", 0)),
        "candidate_path_exhausted_ods": int(d.get("candidate_path_exhausted_ods", 0)),
        "controller_timing_note": d.get("controller_timing_note", ""),
        "precomputed_backup_plan": "yes" if scen not in NON_FAILURE else "no",
        "post_failure_bottleneck_capacity_mbps": round(_f(d.get("post_failure_bottleneck_capacity_mbps")), 3),
        "rate_cap_used": round(_f(d.get("rate_cap_used")), 3),
        "measurement_note": d.get("measurement_note", ""),
        "mode": d.get("mode", "live"),
    }
    for key, _label in TIMING_FIELDS:
        rec[key] = round(_f(tb.get(key)), 3)
    return rec


def load_records(*dirs) -> pd.DataFrame:
    recs = []
    for dpath in dirs:
        for p in sorted(glob.glob(os.path.join(dpath, "*.json"))):
            recs.append(record_from_json(json.load(open(p, encoding="utf-8"))))
    df = pd.DataFrame(recs)
    df["_t"] = df["topology"].map({t: i for i, t in enumerate(TOPOLOGY_ORDER)})
    df["_s"] = df["scenario"].map({s: i for i, s in enumerate(SCENARIO_ORDER)})
    df = df.sort_values(["_t", "_s", "run_id"]).drop(columns=["_t", "_s"]).reset_index(drop=True)
    return df


def build_per_run(df: pd.DataFrame) -> pd.DataFrame:
    out = df[PER_RUN_LOWER].rename(columns=PER_RUN_RENAME).copy()
    out["Rerun on FIX1"] = "yes"
    for c in INT_COLS:
        out[c] = out[c].astype(int)
    return out


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (topo, scenario), g in df.groupby(["topology", "scenario"], sort=False):
        notes = [n for n in g["measurement_note"].dropna().astype(str).unique().tolist() if n.strip()]
        tnotes = [n for n in g["controller_timing_note"].dropna().astype(str).unique().tolist() if n.strip()]
        rows.append({
            "Topology": topo, "Scenario": scenario,
            "Offered UDP rate Mbps": round(float(g["offered_udp_rate_mbps"].mean()), 3),
            "Pre-failure throughput Mbps": round(float(g["pre_failure_throughput_mbps"].mean()), 3),
            "Post-recovery throughput Mbps": round(float(g["post_recovery_throughput_mbps"].mean()), 3),
            "Pre-failure RTT ms": round(float(g["pre_failure_rtt_ms"].mean()), 3),
            "Post-recovery RTT ms": round(float(g["post_recovery_rtt_ms"].mean()), 3),
            "Pre-failure jitter ms": round(float(g["pre_failure_jitter_ms"].mean()), 3),
            "Post-recovery jitter ms": round(float(g["post_recovery_jitter_ms"].mean()), 3),
            "Transient packet loss %": round(float(g["transient_packet_loss_pct"].mean()), 4),
            "Post-recovery steady-state packet loss %": round(float(g["post_recovery_packet_loss_pct"].mean()), 4),
            "Affected ODs": round(float(g["affected_ods"].mean()), 3),
            "Rule count": round(float(g["rule_count"].mean()), 3),
            "Changed rules": round(float(g["changed_rules"].mean()), 3),
            "Unchanged rules": round(float(g["unchanged_rules"].mean()), 3),
            "Added rules": round(float(g["added_rules"].mean()), 3),
            "Removed rules": round(float(g["removed_rules"].mean()), 3),
            "Modified rules": round(float(g["modified_rules"].mean()), 3),
            "Install ms": round(float(g["install_ms"].mean()), 3),
            "Initial install ms": round(float(g["initial_install_ms"].mean()), 3),
            "Incremental install ms": round(float(g["incremental_install_ms"].mean()), 3),
            "Recovery ms": round(float(g["recovery_ms"].mean()), 3),
            "Controller pipeline ms": round(float(g["controller_pipeline_ms"].mean()), 3),
            "Controller decision ms": round(float(g["controller_decision_ms"].mean()), 3),
            "Legacy controller response ms": round(float(g["legacy_controller_response_ms"].mean()), 3),
            "Disconnected ODs": int(g["disconnected_ods"].max()),
            "Run count": int(len(g)),
            "Mode": "live" if (g["mode"].astype(str).str.lower() == "live").all() else "mixed",
            "Rerun on FIX1": "yes" if (g["mode"].astype(str).str.lower() == "live").all() else "no",
            "Measurement note": " | ".join(notes),
            "Controller timing note": " | ".join(tnotes),
            "Post-failure bottleneck capacity Mbps": round(float(g["post_failure_bottleneck_capacity_mbps"].mean()), 3),
            "Rate cap used": round(float(g["rate_cap_used"].mean()), 3),
            "P95 Offered UDP rate Mbps": round(float(g["offered_udp_rate_mbps"].quantile(0.95)), 3),
            "P95 Pre-failure throughput Mbps": round(float(g["pre_failure_throughput_mbps"].quantile(0.95)), 3),
            "P95 Post-recovery throughput Mbps": round(float(g["post_recovery_throughput_mbps"].quantile(0.95)), 3),
            "P95 Pre-failure RTT ms": round(float(g["pre_failure_rtt_ms"].quantile(0.95)), 3),
            "P95 Post-recovery RTT ms": round(float(g["post_recovery_rtt_ms"].quantile(0.95)), 3),
            "P95 Pre-failure jitter ms": round(float(g["pre_failure_jitter_ms"].quantile(0.95)), 3),
            "P95 Post-recovery jitter ms": round(float(g["post_recovery_jitter_ms"].quantile(0.95)), 3),
            "P95 Transient packet loss %": round(float(g["transient_packet_loss_pct"].quantile(0.95)), 4),
            "P95 Post-recovery steady-state packet loss %": round(float(g["post_recovery_packet_loss_pct"].quantile(0.95)), 4),
            "P95 Changed rules": round(float(g["changed_rules"].quantile(0.95)), 3),
            "P95 Unchanged rules": round(float(g["unchanged_rules"].quantile(0.95)), 3),
            "P95 Added rules": round(float(g["added_rules"].quantile(0.95)), 3),
            "P95 Removed rules": round(float(g["removed_rules"].quantile(0.95)), 3),
            "P95 Modified rules": round(float(g["modified_rules"].quantile(0.95)), 3),
            "P95 Install ms": round(float(g["install_ms"].quantile(0.95)), 3),
            "P95 Initial install ms": round(float(g["initial_install_ms"].quantile(0.95)), 3),
            "P95 Incremental install ms": round(float(g["incremental_install_ms"].quantile(0.95)), 3),
            "P95 Recovery ms": round(float(g["recovery_ms"].quantile(0.95)), 3),
            "P95 Controller pipeline ms": round(float(g["controller_pipeline_ms"].quantile(0.95)), 3),
            "P95 Controller decision ms": round(float(g["controller_decision_ms"].quantile(0.95)), 3),
            "P95 Legacy controller response ms": round(float(g["legacy_controller_response_ms"].quantile(0.95)), 3),
        })
    return pd.DataFrame(rows)


def build_timing_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    cols = TIMING_FIELDS + [
        ("controller_decision_ms", "Controller decision ms"),
        ("controller_pipeline_ms", "Controller pipeline ms"),
        ("legacy_controller_response_ms", "Legacy controller response ms"),
        ("initial_install_ms", "Initial install ms"),
        ("incremental_install_ms", "Incremental install ms"),
        ("recovery_ms", "Recovery ms"),
    ]
    rows = []
    for (topo, scenario), g in df.groupby(["topology", "scenario"], sort=False):
        row = {"Topology": topo, "Scenario": scenario, "Run count": int(len(g))}
        for key, label in cols:
            row[f"Mean {label}"] = round(float(g[key].mean()), 3)
            row[f"P95 {label}"] = round(float(g[key].quantile(0.95)), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    here = Path(__file__).resolve().parent
    default_ev = here.parents[1] / "hardfix_handoff_evidence"
    ap = argparse.ArgumentParser()
    ap.add_argument("--abilene-raw", default=str(default_ev / "abilene_single_raw_logs"))
    ap.add_argument("--geant-raw", default=str(default_ev / "geant_raw_logs"))
    ap.add_argument("--out-dir", default=str(here.parents[1] / "HARDFIX_AGGREGATOR_REGEN_AUDIT"))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_records(args.abilene_raw, args.geant_raw)

    per_run = build_per_run(df)
    combined_per_run = per_run.sort_values(["Topology", "Scenario", "Run ID"]).reset_index(drop=True)
    write_csv(combined_per_run, out / "sdn_live_fix1_final_hardfix_per_run.csv")

    summary = build_summary(df)
    write_csv(summary, out / "sdn_live_fix1_final_hardfix_summary.csv")

    write_csv(build_timing_breakdown(df), out / "sdn_live_fix1_final_hardfix_timing_breakdown.csv")

    gdf = df[df["topology"] == "geant"].reset_index(drop=True)
    write_csv(build_per_run(gdf), out / "sdn_live_fix1_final_hardfix_geant_per_run.csv")
    write_csv(build_summary(gdf), out / "sdn_live_fix1_final_hardfix_geant_summary.csv")

    adf = df[(df["topology"] == "abilene") & (df["scenario"] == "single_link_failure")].reset_index(drop=True)
    write_csv(build_per_run(adf), out / "sdn_live_fix1_final_hardfix_abilene_single_per_run.csv")
    write_csv(build_summary(adf), out / "sdn_live_fix1_final_hardfix_abilene_single_summary.csv")

    print(f"Wrote 7 reconstructed CSVs to {out}")


if __name__ == "__main__":
    main()
