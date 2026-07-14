#!/usr/bin/env python3
"""Classify a Target-2 replay batch: are controller route/rule hashes identical across
replays, and are the live outcomes mixed (=> data-plane race) or uniform (=> deterministic)?"""
import argparse, glob, json, os, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    a = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(a.raw_dir, "abilene__capacity_degradation_50__run*.json"))):
        d = json.load(open(p))
        si = d.get("state_identity", {})
        post = d.get("post_recovery", [])
        iperf = [e.get("iperf_loss_pct") for e in post if e.get("iperf_loss_pct") is not None]
        ping = [e.get("ping_loss_pct") for e in post if e.get("ping_loss_pct") is not None]
        rec = d.get("recovery_ms")
        rows.append({
            "replay_id": d["run_id"], "tm_run_id": si.get("tm_run_id"), "tm_hash": si.get("tm_hash"),
            "post_action": si.get("post_action_name"), "post_route_hash": si.get("post_route_hash"),
            "post_selected_ods_hash": si.get("post_selected_ods_hash"),
            "post_selected_od_count": si.get("post_selected_od_count"),
            "event_caps_hash": si.get("event_caps_hash"), "pathlib_post_hash": si.get("pathlib_post_hash"),
            "changed_rules": d.get("changed_rules"), "unchanged_rules": d.get("unchanged_rules"),
            "added_rules": d.get("added_rules"), "removed_rules": d.get("removed_rules"),
            "modified_rules": d.get("modified_rules"),
            "incremental_install_ms": d.get("incremental_install_ms"),
            "recovery_ms": rec, "recovery_timed_out": bool(rec is not None and rec >= 7900),
            "transient_valid": d.get("transient_measurement_valid"),
            "transient_loss_pct": d.get("transient_packet_loss_pct"),
            "post_iperf_loss_mean": round(float(np.mean(iperf)), 3) if iperf else None,
            "post_ping_loss_mean": round(float(np.mean(ping)), 3) if ping else None,
            "post_combined_loss_pct": d.get("post_recovery_packet_loss_pct"),
        })
    import csv
    cols = list(rows[0].keys())
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
    # classification
    route_hashes = set(r["post_route_hash"] for r in rows)
    rule_sigs = set((r["changed_rules"], r["added_rules"], r["removed_rules"], r["modified_rules"]) for r in rows)
    tm_hashes = set(r["tm_hash"] for r in rows)
    fails = [r for r in rows if (r["post_iperf_loss_mean"] or 0) >= 1.0 or r["recovery_timed_out"] or (r["transient_loss_pct"] or 0) >= 10]
    print(f"n={len(rows)}  distinct tm_hash={len(tm_hashes)}  distinct post_route_hash={len(route_hashes)}  distinct rule_sig={len(rule_sigs)}")
    print(f"failing replays (iperf>=1% or timeout or transient>=10%): {len(fails)}/{len(rows)} -> ids {[r['replay_id'] for r in fails]}")
    print("post_iperf_loss per replay:", [r["post_iperf_loss_mean"] for r in rows])
    print("recovery_ms per replay   :", [r["recovery_ms"] for r in rows])
    print("transient_loss per replay:", [r["transient_loss_pct"] for r in rows])
    if len(route_hashes) == 1 and len(rule_sigs) == 1:
        cls = "IDENTICAL routing+rules across replays"
        if 0 < len(fails) < len(rows):
            cls += " with MIXED outcomes => DATA-PLANE race/timing/recovery-detection (B/C/D)"
        elif len(fails) == len(rows):
            cls += " and UNIFORM failure => deterministic TM/routing/infeasibility (A/E)"
        else:
            cls += " and UNIFORM pass"
    else:
        cls = "route/rule hashes DIFFER across replays => controller nondeterminism/state variation (C)"
    print("CLASSIFICATION:", cls)

if __name__ == "__main__":
    main()
