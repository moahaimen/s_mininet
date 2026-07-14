#!/usr/bin/env python3
"""Target 6 flow-install-path A/B: semantic final-flow-table equivalence + timing decomposition.

A = trusted incremental install (per-stale del-flow + per-modified mod-flows + batched add-flows).
B = optimized install (one atomic ovs-ofctl --bundle per changed switch: delete_strict + add).

Both batches run the SAME deterministic geant/spike_x3 TM sequence with the SAME optimized routing
(TARGET5_USE_OPT_ROUTING=1), so run N installs the SAME desired routing on both. This tool:
  1. Compares the normalized whole-network final flow tables per run (must be identical).
  2. Emits the flow_mod_send decomposition (ovs command count, per-switch subprocess time).
  3. Reports pipeline before/after.

Usage: run on the VM after both batches complete.
  python3 scripts/phase1_5/target6_flow_install_ab.py \
      --incr  <...>/sdn_t6_incr_prof_raw_logs \
      --bundle <...>/sdn_t6_bundle_prof_raw_logs \
      --profile-csv TARGET6_FLOW_MOD_SEND_PROFILE.csv
Exit 0 iff every run's final flow table is semantically identical between A and B.
"""
import argparse, glob, json, os, sys, csv


def load_runs(raw_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(raw_dir, "geant__spike_x3__run*.json"))):
        d = json.load(open(f))
        out[int(d["run_id"])] = d
    return out


def table_of(rec):
    ft = rec.get("forwarding_trace") or {}
    return ft.get("all_switch_tables") or {}


def compare_tables(ta, tb):
    """Return (equal, detail). Compares the set of normalized flow lines per switch."""
    sa, sb = set(ta), set(tb)
    if sa != sb:
        return False, f"switch set differs: only_A={sorted(sa - sb)} only_B={sorted(sb - sa)}"
    for sw in sorted(sa):
        rowsa, rowsb = sorted(ta[sw]), sorted(tb[sw])
        if rowsa != rowsb:
            onlya = [r for r in rowsa if r not in set(rowsb)]
            onlyb = [r for r in rowsb if r not in set(rowsa)]
            return False, f"switch {sw} differs: only_A={onlya} only_B={onlyb}"
    return True, "identical"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--incr", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--profile-csv", default="")
    a = ap.parse_args()

    incr = load_runs(a.incr)
    bundle = load_runs(a.bundle)
    runs = sorted(set(incr) & set(bundle))
    print(f"runs compared: {runs}")

    all_equal = True
    rows = []
    for r in runs:
        ra, rb = incr[r], bundle[r]
        ta, tb = table_of(ra), table_of(rb)
        eq, detail = compare_tables(ta, tb)
        na = sum(len(v) for v in ta.values()); nb = sum(len(v) for v in tb.values())
        all_equal = all_equal and eq
        tia = ra.get("timing_breakdown_ms", {}); tib = rb.get("timing_breakdown_ms", {})
        print(f"\nrun {r}: table_equal={eq} ({detail if not eq else 'identical'})")
        print(f"  A rules={na} ovs_cmds={ra.get('ovs_command_count')} flow_mod={tia.get('flow_mod_send_ms')} pipeline={ra.get('controller_pipeline_ms')}")
        print(f"  B rules={nb} ovs_cmds={rb.get('ovs_command_count')} flow_mod={tib.get('flow_mod_send_ms')} pipeline={rb.get('controller_pipeline_ms')}")
        # per-switch profile rows for the CSV (from both A and B)
        for tag, rec in (("A_incremental", ra), ("B_bundle", rb)):
            prof = rec.get("install_switch_profile") or {}
            for sw, pr in sorted(prof.items()):
                rows.append({
                    "batch": tag, "run_id": r, "switch": sw,
                    "required_rule_count": len((table_of(rec) or {}).get(sw, [])),
                    "changed_rule_count": int(pr.get("del_cmds", 0) + pr.get("mod_cmds", 0) + pr.get("add_cmds", 0)),
                    "del_cmds": int(pr.get("del_cmds", 0)),
                    "mod_cmds": int(pr.get("mod_cmds", 0)),
                    "add_cmds": int(pr.get("add_cmds", 0)),
                    "ovs_command_count": int(pr.get("ovs_cmds", 0)),
                    "subprocess_ms": round(float(pr.get("subprocess_ms", 0.0)), 3),
                })

    if a.profile_csv and rows:
        with open(a.profile_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.profile_csv} ({len(rows)} rows)")

    # timing summary
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")
    fa = [incr[r].get("timing_breakdown_ms", {}).get("flow_mod_send_ms") for r in runs]
    fb = [bundle[r].get("timing_breakdown_ms", {}).get("flow_mod_send_ms") for r in runs]
    pa = [incr[r].get("controller_pipeline_ms") for r in runs]
    pb = [bundle[r].get("controller_pipeline_ms") for r in runs]
    print(f"\nflow_mod_send mean: A={mean(fa):.1f}  B={mean(fb):.1f}  ({mean(fa)/max(1e-9,mean(fb)):.2f}x)")
    print(f"pipeline      mean: A={mean(pa):.1f}  B={mean(pb):.1f}")
    print(f"pipeline B all runs: {[round(x,1) for x in pb]}  (<650 target: {'PASS' if all(x<650 for x in pb if x) else 'FAIL'})")
    print(f"\nFINAL_FLOW_TABLE_SEMANTIC_EQUIVALENCE = {'PASS' if all_equal else 'FAIL'}")
    sys.exit(0 if all_equal else 1)


if __name__ == "__main__":
    main()
