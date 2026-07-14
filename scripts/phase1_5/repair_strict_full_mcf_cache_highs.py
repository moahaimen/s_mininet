#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path

import highspy
import numpy as np

import sys

sys.path.insert(0, "/Users/moahaimentalib/Desktop/f_flex_network_code_clean")

from phase1_reactive.eval.common import load_bundle
from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import CONFIG, OUT_ROOT, _build_spec_lookup, build_context


OUT = OUT_ROOT / "condition_compliant_k10_k50"
PREPASS = pickle.load(open(OUT / "_prepass.pkl", "rb"))
PARTIAL = OUT / "STRICT_FULL_MCF_PR" / "_partial"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Repair strict full-MCF cache rows with direct HiGHS LP.")
    ap.add_argument("--topo", required=True)
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--tm", type=int, nargs="*", default=None, help="Explicit TM indices to solve.")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--solver", default="ipm", choices=["ipm", "simplex"])
    ap.add_argument("--crossover", default="off", choices=["off", "on"])
    ap.add_argument("--progress-every", type=int, default=1)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def caps_key(topo: str, lo: int, hi: int) -> tuple[str, int, int]:
    key = (topo, lo, hi)
    if key in PREPASS:
        return key
    # VTL 200-TM extension reuses the frozen 40-TM synthetic capacities.
    if topo == "vtlwavenet2011" and (topo, 0, 40) in PREPASS:
        return (topo, 0, 40)
    raise KeyError(f"No matching prepass capacities for {(topo, lo, hi)}")


class StrictFullMCFHighs:
    def __init__(self, topo: str, lo: int, hi: int, time_limit: float, threads: int, solver: str, crossover: str, verbose: bool):
        self.topo = topo
        self.lo = lo
        self.hi = hi
        self.time_limit = float(time_limit)
        self.threads = int(threads)
        self.solver = solver
        self.crossover = crossover
        self.verbose = verbose

        bundle = load_bundle(CONFIG)
        lookup = _build_spec_lookup(bundle)
        ctx = build_context(bundle, lookup, topo, 8, "disjoint")
        self.ds = ctx["ds"]
        self.caps = np.asarray(PREPASS[caps_key(topo, lo, hi)]["caps"], dtype=np.float64)
        self.nodes = list(self.ds.nodes)
        self.edges = list(self.ds.edges)
        self.od_pairs = list(self.ds.od_pairs)

        self.num_nodes = len(self.nodes)
        self.num_edges = len(self.edges)
        self.num_od = len(self.od_pairs)

        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        self.edge_src_idx = np.asarray([self.node_index[s] for s, _ in self.edges], dtype=np.int32)
        self.edge_dst_idx = np.asarray([self.node_index[d] for _, d in self.edges], dtype=np.int32)
        self.src_idx_per_od = np.asarray([self.node_index[s] for s, _ in self.od_pairs], dtype=np.int32)
        self.dst_idx_per_od = np.asarray([self.node_index[d] for _, d in self.od_pairs], dtype=np.int32)
        self.flow_rhs_base = np.arange(self.num_od, dtype=np.int64) * self.num_nodes

        self.flow_row_ids = np.arange(self.num_edges, self.num_edges + self.num_od * self.num_nodes, dtype=np.int32)

        self._build_model()

    def _build_model(self) -> None:
        t0 = time.perf_counter()
        h = highspy.Highs()
        h.setOptionValue("output_flag", bool(self.verbose))
        h.setOptionValue("solver", self.solver)
        h.setOptionValue("run_crossover", self.crossover)
        h.setOptionValue("presolve", "on")
        h.setOptionValue("parallel", "on")
        h.setOptionValue("threads", self.threads)
        h.setOptionValue("time_limit", self.time_limit)
        h.setOptionValue("random_seed", 42)

        num_rows = self.num_edges + self.num_od * self.num_nodes
        cap_lower = np.full(self.num_edges, -highspy.kHighsInf, dtype=np.float64)
        cap_upper = np.zeros(self.num_edges, dtype=np.float64)
        flow_zero = np.zeros(self.num_od * self.num_nodes, dtype=np.float64)
        row_lower = np.concatenate([cap_lower, flow_zero])
        row_upper = np.concatenate([cap_upper, flow_zero])

        h.addRows(num_rows, row_lower, row_upper, 0, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64))

        num_flow_cols = self.num_od * self.num_edges
        num_cols = 1 + num_flow_cols
        nnz = self.num_edges + 3 * num_flow_cols

        starts = np.empty(num_cols + 1, dtype=np.int32)
        starts[0] = 0
        starts[1:] = self.num_edges + 3 * np.arange(num_flow_cols + 1, dtype=np.int64)

        indices = np.empty(nnz, dtype=np.int32)
        values = np.empty(nnz, dtype=np.float64)

        indices[: self.num_edges] = np.arange(self.num_edges, dtype=np.int32)
        values[: self.num_edges] = -self.caps

        edge_ids = np.tile(np.arange(self.num_edges, dtype=np.int32), self.num_od)
        od_ids = np.repeat(np.arange(self.num_od, dtype=np.int32), self.num_edges)
        src_rows = (self.num_edges + od_ids.astype(np.int64) * self.num_nodes + self.edge_src_idx[edge_ids]).astype(np.int32)
        dst_rows = (self.num_edges + od_ids.astype(np.int64) * self.num_nodes + self.edge_dst_idx[edge_ids]).astype(np.int32)

        base = self.num_edges
        indices[base + 0 :: 3] = edge_ids
        indices[base + 1 :: 3] = src_rows
        indices[base + 2 :: 3] = dst_rows
        values[base + 0 :: 3] = 1.0
        values[base + 1 :: 3] = 1.0
        values[base + 2 :: 3] = -1.0

        costs = np.zeros(num_cols, dtype=np.float64)
        costs[0] = 1.0
        lower = np.zeros(num_cols, dtype=np.float64)
        upper = np.full(num_cols, highspy.kHighsInf, dtype=np.float64)

        h.addCols(num_cols, costs, lower, upper, nnz, starts, indices, values)
        self.highs = h
        self.build_seconds = time.perf_counter() - t0

    def solve_tm(self, tm_index: int) -> tuple[int, float, str, float]:
        tm = np.asarray(self.ds.tm[int(tm_index)], dtype=np.float64)
        rhs = np.zeros(self.num_od * self.num_nodes, dtype=np.float64)
        rhs[self.flow_rhs_base + self.src_idx_per_od] = tm
        rhs[self.flow_rhs_base + self.dst_idx_per_od] = -tm

        self.highs.changeRowsBounds(len(self.flow_row_ids), self.flow_row_ids, rhs, rhs)
        t0 = time.perf_counter()
        self.highs.run()
        solve_s = time.perf_counter() - t0
        model_status = self.highs.getModelStatus()
        status = self.highs.modelStatusToString(model_status)
        obj = float(self.highs.getObjectiveValue()) if model_status == highspy.HighsModelStatus.kOptimal else float("inf")
        return int(tm_index), obj, status, solve_s


def read_existing(path: Path) -> dict[int, tuple[float, str]]:
    if not path.exists():
        return {}
    out: dict[int, tuple[float, str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[int(row["tm_index"])] = (float(row["strict_full_mcf_MLU"]), str(row["mcf_status"]))
    return out


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tm_index", "strict_full_mcf_MLU", "mcf_status", "solve_time_s"])
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    args = parse_args()
    out_path = args.output or (PARTIAL / f"{args.topo}.repair_highs.csv")
    todo = list(args.tm) if args.tm else list(range(args.lo, args.hi))
    existing = read_existing(out_path) if args.resume else {}
    todo = [t for t in todo if t not in existing]

    solver = StrictFullMCFHighs(
        topo=args.topo,
        lo=args.lo,
        hi=args.hi,
        time_limit=args.time_limit,
        threads=args.threads,
        solver=args.solver,
        crossover=args.crossover,
        verbose=args.verbose,
    )
    print(
        f"[build] topo={args.topo} rows={solver.num_edges + solver.num_od * solver.num_nodes} "
        f"cols={1 + solver.num_od * solver.num_edges} build_s={solver.build_seconds:.3f}",
        flush=True,
    )

    solved_rows = [
        {
            "tm_index": t,
            "strict_full_mcf_MLU": mlu,
            "mcf_status": status,
            "solve_time_s": "",
        }
        for t, (mlu, status) in sorted(existing.items())
    ]

    for i, tm_index in enumerate(todo, start=1):
        t, obj, status, solve_s = solver.solve_tm(tm_index)
        solved_rows.append(
            {
                "tm_index": t,
                "strict_full_mcf_MLU": obj,
                "mcf_status": status,
                "solve_time_s": round(solve_s, 3),
            }
        )
        solved_rows.sort(key=lambda x: int(x["tm_index"]))
        write_rows(out_path, solved_rows)
        if i % args.progress_every == 0 or i == len(todo):
            print(
                f"[solve] topo={args.topo} done={i}/{len(todo)} tm={t} status={status} "
                f"mlu={obj} solve_s={solve_s:.3f}",
                flush=True,
            )

    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
