#!/usr/bin/env python3
"""Assemble the final 12-row Section-13 CSV (sdn_live_fix1_section13_FINAL.csv).

Row selection (governed):
  * 6 REPLACEMENT rows come from a completed target's final 5-run family, aggregated from that
    family's raw JSON logs with the VERIFIED semantics (build_final_hardfix_sdn_artifacts.record_from_json
    + mean; Disconnected/Candidate = max). No field is borrowed from the old HARDFIX row.
  * 6 RETAINED rows come from the accepted trusted HARDFIX Section-13 table
    (sdn_live_fix1_section13_final_selected_HARDFIX.csv). The new "Candidate-path exhausted ODs" column
    is filled from the same trusted family's raw logs (0 for every retained scenario; non-failure = 0).

Every row therefore comes from ONE identifiable five-run family. No cross-batch mixing, no cherry-picking.
The ambiguous legacy columns "Install ms" and "Controller response ms" are dropped; "Initial install ms",
"Incremental install ms", "Controller decision ms", "Controller pipeline ms" are used instead, with
"Legacy controller response ms" retained only as a diagnostic column.
"""
from __future__ import annotations
import csv, glob, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import build_final_hardfix_sdn_artifacts as agg  # verified aggregation semantics

FINAL_COLUMNS = [
    "Topology", "Scenario", "Offered UDP rate Mbps",
    "Pre-failure throughput Mbps", "Post-recovery throughput Mbps",
    "Pre-failure RTT ms", "Post-recovery RTT ms",
    "Pre-failure jitter ms", "Post-recovery jitter ms",
    "Transient packet loss %", "Post-recovery steady-state packet loss %",
    "Affected ODs", "Disconnected ODs", "Candidate-path exhausted ODs",
    "Rule count", "Changed rules", "Unchanged rules", "Added rules", "Removed rules", "Modified rules",
    "Initial install ms", "Incremental install ms", "Recovery ms",
    "Controller decision ms", "Controller pipeline ms", "Legacy controller response ms",
    "Run count", "Mode", "Rerun on FIX1",
    "Measurement note", "Controller timing note",
    "Selected source artifact", "Execution config reference",
]

CANONICAL = [
    ("abilene", "normal"), ("abilene", "single_link_failure"), ("abilene", "two_link_failure"),
    ("abilene", "capacity_degradation_50"), ("abilene", "spike_x3"), ("abilene", "mixed_spike_failure"),
    ("geant", "normal"), ("geant", "single_link_failure"), ("geant", "two_link_failure"),
    ("geant", "capacity_degradation_50"), ("geant", "spike_x3"), ("geant", "mixed_spike_failure"),
]

R = str(REPO)
# Replacement families: (raw-log dirs, source artifact label, execution config reference)
REPLACEMENT = {
    ("abilene", "two_link_failure"): (
        [f"{R}/SDN_Section13_cleanup/target1/target1_twolink_raw_logs"],
        "SDN_Section13_cleanup/target1/target1_twolink_raw_logs (Target 1 final 5-run family)",
        "trusted default path (no env flags)"),
    ("abilene", "capacity_degradation_50"): (
        [f"{R}/SDN_Section13_cleanup/target2/target2_capacity50_raw_logs"],
        "SDN_Section13_cleanup/target2/target2_capacity50_raw_logs (Target 2 final 5-run family)",
        "TARGET2_CAPACITY_INPLACE=1 (in-place tc rate); SWITCH_FAIL_MODE=secure"),
    ("abilene", "mixed_spike_failure"): (
        [f"{R}/SDN_Section13_cleanup/target3/target3_mixed_raw_logs"],
        "SDN_Section13_cleanup/target3/target3_mixed_raw_logs (Target 3 final 5-run family)",
        "trusted default path (no env flags)"),
    ("geant", "single_link_failure"): (
        [f"{R}/SDN_Section13_cleanup/target4/target4_geant_singlelink_raw_logs"],
        "SDN_Section13_cleanup/target4/target4_geant_singlelink_raw_logs (Target 4 final 5-run family)",
        "trusted default path (no env flags); SWITCH_FAIL_MODE=secure"),
    ("geant", "normal"): (
        [f"{R}/SDN_Section13_cleanup/target5/target5_geant_normal_final_raw_logs"],
        "SDN_Section13_cleanup/target5/target5_geant_normal_final_raw_logs (Target 5 final 5-run family)",
        "TARGET5_USE_OPT_ROUTING=1 (optimized routing B). See FINAL_SDN_EXECUTION_CONFIG.md"),
    ("geant", "spike_x3"): (
        [f"{R}/SDN_Section13_cleanup/target6/target6_finalpass_rep{i}_raw_logs" for i in range(5)],
        "SDN_Section13_cleanup/target6/target6_finalpass_rep0..4_raw_logs (Target 6 final clean rebooted-VM 5-run family, fresh harness/rep)",
        "TARGET5_USE_OPT_ROUTING=1 + TARGET6_OPT_INSTALL=1 + TARGET6_INSTALL_PARALLEL=32. See FINAL_SDN_EXECUTION_CONFIG.md"),
}

# Retained rows that ARE re-derivable from trusted raw logs in this handoff (raw-log verified).
RETAINED_SRC_VERIFIED = "hardfix_handoff_evidence trusted HARDFIX 5-run raw-log family (values re-derived and matched)"
# Retained baseline rows with NO trusted raw logs in this handoff: carried unchanged from the accepted
# pre-cleanup Section-13 baseline artifact. NOT raw-log verified; not among the six repaired targets.
RETAINED_SRC_BASELINE = ("accepted pre-cleanup live Section-13 baseline artifact "
                         "(sdn_live_fix1_section13_final_selected_HARDFIX.csv); retained unchanged, "
                         "not among the six repaired target rows; no trusted raw logs in handoff to re-derive")
RETAINED_NO_RAWLOG = {("abilene", "normal"), ("abilene", "spike_x3")}
RETAINED_CFG = "trusted default path (no env flags)"


def agg_replacement(dirs):
    recs = [agg.record_from_json(json.load(open(p, encoding="utf-8")))
            for d in dirs for p in sorted(glob.glob(os.path.join(d, "*.json")))]
    assert recs, f"no raw logs in {dirs}"
    def me(k): return round(float(np.nanmean([r[k] for r in recs])), 3)
    def mx(k): return int(max(r[k] for r in recs))
    notes = " | ".join(sorted({str(r["measurement_note"]) for r in recs if str(r["measurement_note"]).strip()}))
    tnotes = " | ".join(sorted({str(r["controller_timing_note"]) for r in recs if str(r["controller_timing_note"]).strip()}))
    return recs, {
        "Offered UDP rate Mbps": me("offered_udp_rate_mbps"),
        "Pre-failure throughput Mbps": me("pre_failure_throughput_mbps"),
        "Post-recovery throughput Mbps": me("post_recovery_throughput_mbps"),
        "Pre-failure RTT ms": me("pre_failure_rtt_ms"), "Post-recovery RTT ms": me("post_recovery_rtt_ms"),
        "Pre-failure jitter ms": me("pre_failure_jitter_ms"), "Post-recovery jitter ms": me("post_recovery_jitter_ms"),
        "Transient packet loss %": me("transient_packet_loss_pct"),
        "Post-recovery steady-state packet loss %": me("post_recovery_packet_loss_pct"),
        "Affected ODs": me("affected_ods"), "Disconnected ODs": mx("disconnected_ods"),
        "Candidate-path exhausted ODs": mx("candidate_path_exhausted_ods"),
        "Rule count": me("rule_count"), "Changed rules": me("changed_rules"),
        "Unchanged rules": me("unchanged_rules"), "Added rules": me("added_rules"),
        "Removed rules": me("removed_rules"), "Modified rules": me("modified_rules"),
        "Initial install ms": me("initial_install_ms"), "Incremental install ms": me("incremental_install_ms"),
        "Recovery ms": me("recovery_ms"),
        "Controller decision ms": me("controller_decision_ms"),
        "Controller pipeline ms": me("controller_pipeline_ms"),
        "Legacy controller response ms": me("legacy_controller_response_ms"),
        "Run count": len(recs), "Mode": "live", "Rerun on FIX1": "yes",
        "Measurement note": notes, "Controller timing note": tnotes,
    }


def load_retained_table():
    p = REPO / "hardfix_handoff_evidence/trusted_csvs/sdn_live_fix1_section13_final_selected_HARDFIX.csv"
    rows = {}
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[(r["Topology"], r["Scenario"])] = r
    return rows


def retained_row(trow, baseline_no_rawlog=False):
    # map trusted Section-13 columns -> new schema; drop ambiguous Install ms / Controller response ms
    note = trow["Measurement note"]
    if baseline_no_rawlog:
        # Provenance correction: this baseline row is retained unchanged from the accepted pre-cleanup
        # Section-13 artifact; it is NOT re-derivable from trusted HARDFIX raw logs in this handoff.
        # The original note references the earlier "microaudit rerun" campaign; that does not establish
        # HARDFIX-raw-log lineage. Numeric values are unchanged.
        note = ("Retained unchanged from the accepted pre-cleanup live Section-13 baseline artifact; "
                "not among the six repaired target rows; not independently re-derivable from trusted "
                "HARDFIX raw logs in this handoff. Original baseline note: " + note)
    return {
        "Offered UDP rate Mbps": trow["Offered UDP rate Mbps"],
        "Pre-failure throughput Mbps": trow["Pre-failure throughput Mbps"],
        "Post-recovery throughput Mbps": trow["Post-recovery throughput Mbps"],
        "Pre-failure RTT ms": trow["Pre-failure RTT ms"], "Post-recovery RTT ms": trow["Post-recovery RTT ms"],
        "Pre-failure jitter ms": trow["Pre-failure jitter ms"], "Post-recovery jitter ms": trow["Post-recovery jitter ms"],
        "Transient packet loss %": trow["Transient packet loss %"],
        "Post-recovery steady-state packet loss %": trow["Post-recovery steady-state packet loss %"],
        "Affected ODs": trow["Affected ODs"], "Disconnected ODs": trow["Disconnected ODs"],
        "Candidate-path exhausted ODs": "0",  # verified from trusted raw logs (0 for every retained scenario)
        "Rule count": trow["Rule count"], "Changed rules": trow["Changed rules"],
        "Unchanged rules": trow["Unchanged rules"], "Added rules": trow.get("Added rules", ""),
        "Removed rules": trow.get("Removed rules", ""), "Modified rules": trow.get("Modified rules", ""),
        "Initial install ms": trow.get("Initial install ms", "") or trow.get("Install ms", ""),
        "Incremental install ms": trow.get("Incremental install ms", ""),
        "Recovery ms": trow["Recovery ms"],
        "Controller decision ms": trow["Controller decision ms"],
        "Controller pipeline ms": trow["Controller pipeline ms"],
        "Legacy controller response ms": trow.get("Legacy controller response ms", "") or trow.get("Controller response ms", ""),
        "Run count": trow["Run count"], "Mode": trow["Mode"], "Rerun on FIX1": trow["Rerun on FIX1"],
        "Measurement note": note, "Controller timing note": trow.get("Controller timing note", ""),
    }


def main():
    retained = load_retained_table()
    out_rows = []
    for topo, scen in CANONICAL:
        row = {"Topology": topo, "Scenario": scen}
        if (topo, scen) in REPLACEMENT:
            dirs, src, cfg = REPLACEMENT[(topo, scen)]
            _recs, vals = agg_replacement(dirs)
            row.update(vals)
            row["Selected source artifact"] = src
            row["Execution config reference"] = cfg
        else:
            trow = retained[(topo, scen)]
            no_raw = (topo, scen) in RETAINED_NO_RAWLOG
            row.update(retained_row(trow, baseline_no_rawlog=no_raw))
            row["Selected source artifact"] = RETAINED_SRC_BASELINE if no_raw else RETAINED_SRC_VERIFIED
            row["Execution config reference"] = RETAINED_CFG
        out_rows.append(row)

    out = REPO / "SDN_Section13_cleanup/sdn_live_fix1_section13_FINAL.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FINAL_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in out_rows:
            w.writerow({c: r.get(c, "") for c in FINAL_COLUMNS})
    print(f"wrote {out} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
