#!/usr/bin/env python3
"""Aggregate the 5 Target-2 (abilene/capacity_degradation_50) live raw logs into the
required per-run and summary CSVs, with explicit transient-measurement validity and
capacity-event evidence. Primary source = raw JSON logs. No metric value hard-coded.
"""
from __future__ import annotations
import argparse, glob, json, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

def _f(x): return np.nan if x is None else float(x)
def _nansum(v): return float(np.nansum([_f(x) for x in v])) if v else 0.0
def _nanmean(v):
    a=[_f(x) for x in v]; return float(np.nanmean(a)) if a else float("nan")
def _pair(lst,a,b):
    return _nanmean([np.nanmean([_f(e.get(a)),_f(e.get(b))]) for e in lst]) if lst else float("nan")

PER_RUN_COLS=["run_id","topology","scenario","mode","offered_udp_rate_mbps",
 "pre_failure_throughput_mbps","post_recovery_throughput_mbps","pre_failure_rtt_ms",
 "post_recovery_rtt_ms","pre_failure_jitter_ms","post_recovery_jitter_ms",
 "transient_measurement_valid","transient_measurement_invalid_reason","transient_packet_loss_pct",
 "post_recovery_packet_loss_pct","affected_ods","disconnected_ods","candidate_path_exhausted_ods",
 "degraded_edge_keys","capacity_before","capacity_after","capacity_event_link_remained_up",
 "capacity_application_method","rule_count","changed_rules","unchanged_rules","initial_install_ms",
 "incremental_install_ms","recovery_ms","controller_decision_ms","controller_pipeline_ms",
 "legacy_controller_response_ms","raw_log_path"]

def row_from_json(path):
    d=json.load(open(path))
    pre=d.get("pre_failure") or []; post=d.get("post_recovery") or []
    ce=d.get("capacity_event") or {}
    before=ce.get("capacity_before_mbps") or {}; after=ce.get("capacity_after_mbps") or {}
    cb=sorted(set(before.values())); ca=sorted(set(after.values()))
    return {
        "run_id":int(d["run_id"]),"topology":d["topology"],"scenario":d["scenario"],"mode":d.get("mode"),
        "offered_udp_rate_mbps":round(_f(d.get("offered_udp_rate_mbps")),3),
        "pre_failure_throughput_mbps":round(_nansum([e.get("throughput_mbps") for e in pre]),3),
        "post_recovery_throughput_mbps":round(_nansum([e.get("throughput_mbps") for e in post]),3),
        "pre_failure_rtt_ms":round(_nanmean([e.get("ping_rtt_ms") for e in pre]),3),
        "post_recovery_rtt_ms":round(_nanmean([e.get("ping_rtt_ms") for e in post]),3),
        "pre_failure_jitter_ms":round(_pair(pre,"iperf_jitter_ms","ping_jitter_ms"),3),
        "post_recovery_jitter_ms":round(_pair(post,"iperf_jitter_ms","ping_jitter_ms"),3),
        "transient_measurement_valid":bool(d.get("transient_measurement_valid")),
        "transient_measurement_invalid_reason":d.get("transient_measurement_invalid_reason") or "",
        "transient_packet_loss_pct":d.get("transient_packet_loss_pct"),
        "post_recovery_packet_loss_pct":d.get("post_recovery_packet_loss_pct"),
        "affected_ods":d.get("affected_ods"),"disconnected_ods":d.get("disconnected_ods"),
        "candidate_path_exhausted_ods":d.get("candidate_path_exhausted_ods"),
        "degraded_edge_keys":";".join(ce.get("degraded_edge_keys",[])),
        "capacity_before":(cb[0] if len(cb)==1 else str(cb)),
        "capacity_after":(ca[0] if len(ca)==1 else str(ca)),
        "capacity_event_link_remained_up":bool(ce.get("capacity_event_link_remained_up")),
        "capacity_application_method":ce.get("capacity_application_method"),
        "rule_count":d.get("rule_count"),"changed_rules":d.get("changed_rules"),
        "unchanged_rules":d.get("unchanged_rules"),
        "initial_install_ms":d.get("initial_install_ms"),"incremental_install_ms":d.get("incremental_install_ms"),
        "recovery_ms":d.get("recovery_ms"),"controller_decision_ms":d.get("controller_decision_ms"),
        "controller_pipeline_ms":d.get("controller_pipeline_ms"),
        "legacy_controller_response_ms":d.get("legacy_controller_response_ms"),
        "raw_log_path":os.path.basename(path),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw-dir",required=True)
    ap.add_argument("--out-dir",required=True)
    a=ap.parse_args()
    os.makedirs(a.out_dir,exist_ok=True)
    files=sorted(glob.glob(os.path.join(a.raw_dir,"abilene__capacity_degradation_50__run*.json")))
    rows=[row_from_json(p) for p in files]
    df=pd.DataFrame(rows)[PER_RUN_COLS].sort_values("run_id")
    df.to_csv(os.path.join(a.out_dir,"target2_capacity50_per_run.csv"),index=False,lineterminator="\n")
    valid=int(df["transient_measurement_valid"].sum()); n=len(df)
    tvalid=df[df["transient_measurement_valid"]]
    summ={
        "topology":"abilene","scenario":"capacity_degradation_50","run_count":n,
        "transient_valid_run_count":valid,"transient_invalid_run_count":n-valid,
        "mean_transient_packet_loss_pct":round(float(pd.to_numeric(tvalid["transient_packet_loss_pct"]).mean()),4) if valid else None,
        "mean_post_recovery_packet_loss_pct":round(float(pd.to_numeric(df["post_recovery_packet_loss_pct"]).mean()),4),
        "mean_pre_throughput_mbps":round(float(df["pre_failure_throughput_mbps"].mean()),3),
        "mean_post_throughput_mbps":round(float(df["post_recovery_throughput_mbps"].mean()),3),
        "mean_pre_rtt_ms":round(float(df["pre_failure_rtt_ms"].mean()),3),
        "mean_post_rtt_ms":round(float(df["post_recovery_rtt_ms"].mean()),3),
        "mean_recovery_ms":round(float(pd.to_numeric(df["recovery_ms"]).mean()),3),
        "mean_controller_decision_ms":round(float(pd.to_numeric(df["controller_decision_ms"]).mean()),3),
        "mean_controller_pipeline_ms":round(float(pd.to_numeric(df["controller_pipeline_ms"]).mean()),3),
        "mean_legacy_controller_response_ms":round(float(pd.to_numeric(df["legacy_controller_response_ms"]).mean()),3),
        "max_disconnected_ods":int(pd.to_numeric(df["disconnected_ods"]).max()),
        "mean_affected_ods":round(float(pd.to_numeric(df["affected_ods"]).mean()),3),
        "capacity_event_link_remained_up_count":int(df["capacity_event_link_remained_up"].sum()),
        "capacity_application_method":";".join(sorted(set(df["capacity_application_method"].astype(str)))),
        "mode":";".join(sorted(set(df["mode"].astype(str)))),"rerun_on_fix1":"yes",
    }
    pd.DataFrame([summ]).to_csv(os.path.join(a.out_dir,"target2_capacity50_summary.csv"),index=False,lineterminator="\n")
    print("valid_transient_runs=%d/%d" % (valid,n))
    print(df[["run_id","transient_measurement_valid","transient_packet_loss_pct","post_recovery_packet_loss_pct","capacity_event_link_remained_up","capacity_application_method","recovery_ms"]].to_string(index=False))
    print("SUMMARY:",json.dumps(summ,indent=0))

if __name__=="__main__":
    main()
