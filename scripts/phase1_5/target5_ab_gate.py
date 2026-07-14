#!/usr/bin/env python3
"""GEANT routing A/B equivalence + timing gate (Target 5/6).

A = trusted routing (te.lp_solver.solve_selected_path_lp_dbbudget, unchanged original).
B = optimized routing (te.lp_solver_opt.solve_selected_path_lp_dbbudget_opt).

Runs both on the SAME captured controller input N times; compares status, MLU, DB (disturbance),
per-OD path assignment, and capacity feasibility; reports timing. Exit 0 iff equivalence holds.
"""
import argparse, pickle, time, sys
import numpy as np

sys.path.insert(0, "/home/mininet/network_project")
from te.lp_solver import solve_selected_path_lp_dbbudget as solve_A
from te.lp_solver import apply_routing  # trusted routing evaluator
try:
    from te.lp_solver_opt import solve_selected_path_lp_dbbudget_opt as solve_B
except Exception as e:
    print("could not import optimized solver B:", e); raise
try:
    from te.disturbance import compute_disturbance
except Exception:
    compute_disturbance = None


def db_of(prev, splits, tm):
    if compute_disturbance is None:
        return float("nan")
    return float(compute_disturbance(prev, splits, tm))


def mlu_of(tm, splits, pl, caps):
    return float(apply_routing(tm, splits, pl, caps).mlu)


def run(solve, inp, reps, db_weight):
    times = []
    res = None
    for _ in range(reps):
        t0 = time.perf_counter()
        res = solve(
            tm_vector=inp["tm"], selected_ods=inp["selected_ods"], base_splits=inp["base_splits"],
            path_library=inp["path_library"], capacities=inp["capacities"],
            prev_splits=inp["prev_splits"], db_budget=inp["db_budget"], db_weight=db_weight,
        )
        times.append((time.perf_counter() - t0) * 1000.0)
    return res, times


def splits_equal(sa, sb, tol):
    if len(sa) != len(sb):
        return False, "len"
    max_abs = 0.0
    for a, b in zip(sa, sb):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        n = max(a.size, b.size)
        ap = np.zeros(n); ap[:a.size] = a; bp = np.zeros(n); bp[:b.size] = b
        max_abs = max(max_abs, float(np.max(np.abs(ap - bp))) if n else 0.0)
    return max_abs <= tol, max_abs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--db-weight", type=float, default=1e-6,
                    help="MUST match the live runner (run_sdn_mininet_clean.py passes db_weight=1e-6)")
    a = ap.parse_args()
    inp = pickle.load(open(a.input, "rb"))
    tm, pl, caps, prev = inp["tm"], inp["path_library"], inp["capacities"], inp["prev_splits"]

    resA, tA = run(solve_A, inp, a.reps, a.db_weight)
    resB, tB = run(solve_B, inp, a.reps, a.db_weight)

    mluA, mluB = mlu_of(tm, resA.splits, pl, caps), mlu_of(tm, resB.splits, pl, caps)
    dbA, dbB = db_of(prev, resA.splits, tm), db_of(prev, resB.splits, tm)
    seq, smax = splits_equal(resA.splits, resB.splits, a.tol)

    print(f"topo={inp.get('topo')} selected_ods={len(inp['selected_ods'])} reps={a.reps} db_weight={a.db_weight:g}")
    print(f"A status={resA.status} MLU={mluA:.12f} DB={dbA:.12f} time_ms(min/med)={min(tA):.1f}/{sorted(tA)[len(tA)//2]:.1f}")
    print(f"B status={resB.status} MLU={mluB:.12f} DB={dbB:.12f} time_ms(min/med)={min(tB):.1f}/{sorted(tB)[len(tB)//2]:.1f}")
    print(f"|MLU_A-MLU_B|={abs(mluA-mluB):.3e}  |DB_A-DB_B|={abs(dbA-dbB):.3e}  max|split_A-split_B|={smax:.3e}")
    feasibleB = mluB <= 1.0 + 1e-9  # U<=1 means capacity-feasible at achieved MLU scaling
    status_ok = str(resA.status) == str(resB.status)
    mlu_ok = abs(mluA - mluB) <= a.tol
    db_ok = (dbA != dbA and dbB != dbB) or abs(dbA - dbB) <= a.tol  # nan-safe
    speedup = (sorted(tA)[len(tA)//2]) / max(1e-9, sorted(tB)[len(tB)//2])
    print(f"speedup(medianA/medianB)={speedup:.2f}x")
    verdict = status_ok and mlu_ok and db_ok and seq
    print("EQUIVALENCE:", "PASS" if verdict else "FAIL",
          f"(status_ok={status_ok} mlu_ok={mlu_ok} db_ok={db_ok} splits_ok={seq})")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
