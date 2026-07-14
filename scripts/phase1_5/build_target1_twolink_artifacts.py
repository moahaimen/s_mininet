#!/usr/bin/env python3
"""Aggregate abilene/two_link_failure live raw logs into per-run + summary CSVs with explicit
transient-measurement validity and failed-edge evidence. Primary source = raw JSON logs."""
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

COLS=["run_id","topology","scenario","mode","offered_udp_rate_mbps","pre_failure_throughput_mbps",
 "post_recovery_throughput_mbps","pre_failure_rtt_ms","post_recovery_rtt_ms","pre_failure_jitter_ms",
 "post_recovery_jitter_ms","transient_measurement_valid","transient_measurement_invalid_reason",
 "transient_packet_loss_pct","post_recovery_packet_loss_pct","affected_ods","disconnected_ods",
 "candidate_path_exhausted_ods","failed_link_keys","rule_count","changed_rules","unchanged_rules",
 "added_rules","removed_rules","modified_rules","initial_install_ms","incremental_install_ms",
 "recovery_ms","controller_decision_ms","controller_pipeline_ms","legacy_controller_response_ms",
 "raw_log_path"]

def row(path):
    d=json.load(open(path)); pre=d.get("pre_failure") or []; post=d.get("post_recovery") or []
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
        "failed_link_keys":";".join(d.get("failed_link_keys",[])),
        "rule_count":d.get("rule_count"),"changed_rules":d.get("changed_rules"),
        "unchanged_rules":d.get("unchanged_rules"),"added_rules":d.get("added_rules"),
        "removed_rules":d.get("removed_rules"),"modified_rules":d.get("modified_rules"),
        "initial_install_ms":d.get("initial_install_ms"),"incremental_install_ms":d.get("incremental_install_ms"),
        "recovery_ms":d.get("recovery_ms"),"controller_decision_ms":d.get("controller_decision_ms"),
        "controller_pipeline_ms":d.get("controller_pipeline_ms"),
        "legacy_controller_response_ms":d.get("legacy_controller_response_ms"),
        "raw_log_path":os.path.basename(path),
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--raw-dir",required=True); ap.add_argument("--out-dir",required=True)
    ap.add_argument("--scenario",default="two_link_failure"); ap.add_argument("--prefix",default="target1_twolink")
    a=ap.parse_args(); os.makedirs(a.out_dir,exist_ok=True)
    files=sorted(glob.glob(os.path.join(a.raw_dir,f"*__{a.scenario}__run*.json")))
    df=pd.DataFrame([row(p) for p in files])[COLS].sort_values("run_id")
    df.to_csv(os.path.join(a.out_dir,f"{a.prefix}_per_run.csv"),index=False,lineterminator="\n")
    valid=int(df["transient_measurement_valid"].sum()); n=len(df)
    tvalid=df[df["transient_measurement_valid"]]
    summ={"topology":str(df["topology"].iloc[0]),"scenario":a.scenario,"run_count":n,"transient_valid_run_count":valid,
     "transient_invalid_run_count":n-valid,
     "mean_transient_packet_loss_pct":round(float(pd.to_numeric(tvalid["transient_packet_loss_pct"]).mean()),4) if valid else None,
     "mean_post_recovery_packet_loss_pct":round(float(pd.to_numeric(df["post_recovery_packet_loss_pct"]).mean()),4),
     "mean_recovery_ms":round(float(pd.to_numeric(df["recovery_ms"]).mean()),3),
     "max_recovery_ms":round(float(pd.to_numeric(df["recovery_ms"]).max()),3),
     "recovery_timeouts":int((pd.to_numeric(df["recovery_ms"])>=7900).sum()),
     "max_disconnected_ods":int(pd.to_numeric(df["disconnected_ods"]).max()),
     "max_candidate_path_exhausted_ods":int(pd.to_numeric(df["candidate_path_exhausted_ods"]).max()),
     "mean_affected_ods":round(float(pd.to_numeric(df["affected_ods"]).mean()),3),
     "mean_controller_decision_ms":round(float(pd.to_numeric(df["controller_decision_ms"]).mean()),3),
     "mean_controller_pipeline_ms":round(float(pd.to_numeric(df["controller_pipeline_ms"]).mean()),3),
     "failed_link_keys":";".join(sorted(set(df["failed_link_keys"].astype(str)))),
     "mode":";".join(sorted(set(df["mode"].astype(str)))),"rerun_on_fix1":"yes"}
    pd.DataFrame([summ]).to_csv(os.path.join(a.out_dir,f"{a.prefix}_summary.csv"),index=False,lineterminator="\n")
    print("valid=%d/%d"%(valid,n))
    print(df[["run_id","transient_measurement_valid","transient_packet_loss_pct","post_recovery_packet_loss_pct","disconnected_ods","candidate_path_exhausted_ods","recovery_ms","failed_link_keys"]].to_string(index=False))
    print("SUMMARY:",json.dumps(summ,indent=0))

if __name__=="__main__": main()
