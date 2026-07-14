#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/Users/moahaimentalib/Desktop/f_flex_network_code_clean")
sys.path.insert(0, str(ROOT))

from scripts.phase1_5.gnn_lpd_dqn_selective_db_lp import (  # noqa: E402
    GNNLPDScorer,
    GNN_CHECKPOINT_DEFAULT,
    _make_envs,
    apply_routing,
    clone_splits,
)
from scripts.phase1_5.run_final_iter2 import bottleneck_rank, build_mixed, kp_for, pad_to_lib  # noqa: E402
from scripts.phase1_5.run_fulldata_gated_fix1 import ACTIONS, ANAME  # noqa: E402
from scripts.phase1_5 import agnostic_lib as A  # noqa: E402
from te.baselines import ecmp_splits  # noqa: E402
from te.disturbance import compute_disturbance  # noqa: E402
from te.lp_solver import solve_all_od_path_lp, solve_selected_path_lp_dbbudget  # noqa: E402
from te.paths import PathLibrary  # noqa: E402


OUT = ROOT / "results/gnn_lpd_dqn_selective_db_lp/condition_compliant_k10_k50"
FIX1 = OUT / "FULLDATA_GATED_PRESERVED_FIX1"
COMPLETED = OUT / "FINAL_REPORT_FIX1/completed_metrics"
LOGS = COMPLETED / "logs"
COMPLETED.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    "single_link_failure",
    "two_link_failure",
    "three_link_failure",
    "capacity_degradation_50",
    "spike",
    "mixed_spike_failure",
]
TOPOS = {
    "abilene": (2016, 2036),
    "geant": (672, 692),
    "cernet": (200, 220),
    "sprintlink": (200, 220),
    "tiscali": (200, 220),
    "ebone": (200, 220),
    "germany50": (0, 20),
    "vtlwavenet2011": (0, 20),
}
GNN_MS = {
    "abilene": 3,
    "geant": 7,
    "cernet": 22,
    "sprintlink": 27,
    "tiscali": 33,
    "ebone": 12,
    "germany50": 26,
    "vtlwavenet2011": 140,
}
CYCLES = 20
KEEP_IDX = [i for i, v in ACTIONS.items() if v[0] == "keep"][0]

gnn = GNNLPDScorer(str(GNN_CHECKPOINT_DEFAULT), device="cpu")
AGN = OUT / "TOPOLOGY_AGNOSTIC_BOTTLENECK_DDQN" / "_cache"
SCALER = json.load(open(AGN / "scaler.json"))
MEAN = np.array(SCALER["mean"], np.float32)
STD = np.array(SCALER["std"], np.float32)
DIM = len(A.AGN_FEAT_NAMES)

teacher = A.QNet(DIM, 7)
teacher.load_state_dict(torch.load(OUT / "FROZEN_FINAL_LEARNED_RUNTIME_SAFE_ITER2/final_learned_4of5_iter2_model.pt", map_location="cpu")["state_dict"])
teacher.eval()
final_net = A.QNet(DIM, 7)
final_net.load_state_dict(torch.load(FIX1 / "fulldata_gated_model.pt", map_location="cpu")["state_dict"])
final_net.eval()


def pr_of(opt_mlu: float, achieved_mlu: float) -> float:
    return float(min(1.0, opt_mlu / achieved_mlu)) if achieved_mlu > 0 else 0.0


def pick_failed_links(caps: np.ndarray, n: int) -> np.ndarray:
    return np.argsort(caps)[::-1][: min(n, int(np.sum(caps > 0)))]


def modify_caps(caps: np.ndarray, scenario: str) -> np.ndarray:
    out = caps.copy()
    if scenario == "single_link_failure":
        out[pick_failed_links(caps, 1)] = 0.0
    elif scenario == "two_link_failure":
        out[pick_failed_links(caps, 2)] = 0.0
    elif scenario == "three_link_failure":
        out[pick_failed_links(caps, 3)] = 0.0
    elif scenario == "capacity_degradation_50":
        out *= 0.5
    elif scenario == "mixed_spike_failure":
        out[pick_failed_links(caps, 1)] = 0.0
    return out


def tm_scale(scenario: str) -> float:
    return 3.0 if scenario in ("spike", "mixed_spike_failure") else 1.0


def n_failed_links(scenario: str) -> int:
    return {
        "single_link_failure": 1,
        "two_link_failure": 2,
        "three_link_failure": 3,
        "mixed_spike_failure": 1,
    }.get(scenario, 0)


def prune_pathlib(pl: PathLibrary, caps: np.ndarray) -> PathLibrary:
    node_paths, edge_paths, edge_idx_paths, costs = [], [], [], []
    for i in range(len(pl.edge_idx_paths_by_od)):
        keep = [j for j, ep in enumerate(pl.edge_idx_paths_by_od[i]) if all(float(caps[e]) > 0 for e in ep)]
        node_paths.append([pl.node_paths_by_od[i][j] for j in keep])
        edge_paths.append([pl.edge_paths_by_od[i][j] for j in keep])
        edge_idx_paths.append([pl.edge_idx_paths_by_od[i][j] for j in keep])
        costs.append([pl.costs_by_od[i][j] for j in keep])
    return PathLibrary(
        od_pairs=pl.od_pairs,
        node_paths_by_od=node_paths,
        edge_paths_by_od=edge_paths,
        edge_idx_paths_by_od=edge_idx_paths,
        costs_by_od=costs,
    )


def standardize(raw, keep_mlu, emlu):
    return A.standardize(A.raw_to_vec(raw, keep_mlu, emlu), MEAN, STD)


def run():
    rows = []
    for topo, (lo, hi) in TOPOS.items():
        env = _make_envs([topo], {topo: (lo, hi)}, gnn, hi - lo, 30)[0]
        ds, pl0, caps0 = env.ctx["ds"], env.ctx["pl"], np.asarray(env.ctx["caps"], float)
        struct = A.struct_feats(ds)
        for scenario in SCENARIOS:
            caps = modify_caps(caps0, scenario)
            pl = prune_pathlib(pl0, caps)
            ecmp = ecmp_splits(pl)
            accepted = clone_splits(ecmp)
            prev_tm = None
            for offset, t in enumerate(range(lo, lo + CYCLES)):
                tm = np.asarray(ds.tm[t], float) * tm_scale(scenario)
                act = [od for od in range(len(tm)) if tm[od] > 0]
                disconnected = [od for od in act if len(pl.edge_idx_paths_by_od[od]) == 0]
                opt = float(solve_all_od_path_lp(tm, pl, caps, time_limit_sec=60).mlu)
                util = apply_routing(tm, ecmp, pl, caps).utilization
                scr, _, _ = gnn.score(dataset=ds, tm_vector=tm, path_library=pl, capacities=caps, ecmp_base=ecmp)
                scr = np.asarray(scr, float).ravel()
                av = scr[act] if act else np.zeros(1)
                ranked = bottleneck_rank(tm, ecmp, pl, caps, scr)
                keep_mlu = float(apply_routing(tm, accepted, pl, caps).mlu)
                chg = 0.0 if prev_tm is None else float(np.abs(tm - prev_tm).sum() / (np.abs(prev_tm).sum() + 1e-9))
                emlu = float(apply_routing(tm, ecmp, pl, caps).mlu)
                dd = dict(ranked={t: np.asarray(ranked, np.int32)}, tm_cache=ds.tm, num_nodes=len(ds.nodes))
                dpre = dict(
                    tmstat={t: (float(np.log1p(tm.sum())), float(tm.max() / (tm.sum() + 1e-9)), min(chg, 3.0), len(act))},
                    sstat={t: (float(av.mean()), float(np.quantile(av, 0.95)), float(av.max()))},
                    emlu={t: emlu},
                )
                raw = A.raw_static(topo, t, dd, dpre, pl, ecmp, caps, scr, util, struct)
                s = standardize(raw, keep_mlu, emlu)
                with torch.no_grad():
                    q = teacher(torch.tensor(s).unsqueeze(0)) if offset == 0 else final_net(torch.tensor(s).unsqueeze(0))
                    a = int(q.argmax())
                if offset == 0 and a == KEEP_IDX:
                    qn = q.clone()
                    qn[..., KEEP_IDX] = -1e9
                    a = int(qn.argmax())
                kind, K, _ = ACTIONS[a]
                if kind == "keep":
                    splits = accepted
                    mlu = keep_mlu
                    decision_ms = GNN_MS[topo] + 0.5
                    selected_k = 0
                else:
                    selected = list(ranked[:K])
                    db_budget = 1.0 if scenario != "capacity_degradation_50" else 0.051
                    mixed = build_mixed(pl, set(int(o) for o in selected), kp_for(K))
                    t0 = time.perf_counter()
                    lp = solve_selected_path_lp_dbbudget(
                        tm_vector=tm,
                        selected_ods=selected,
                        base_splits=ecmp,
                        path_library=mixed,
                        capacities=caps,
                        prev_splits=accepted,
                        db_budget=db_budget,
                        db_weight=1e-6,
                        time_limit_sec=60,
                    )
                    splits = pad_to_lib(lp.splits, pl)
                    mlu = float(apply_routing(tm, splits, pl, caps).mlu)
                    decision_ms = (time.perf_counter() - t0) * 1000.0 + GNN_MS[topo]
                    selected_k = int(len([o for o in selected if tm[o] > 0]))
                pr = pr_of(opt, mlu)
                rows.append(
                    {
                        "Scenario": scenario,
                        "Topology": topo,
                        "tm_index": int(t),
                        "Mean PR": pr,
                        "PR>=0.90 flag": int(pr >= 0.90),
                        "PR(fail)": pr if scenario != "spike" else np.nan,
                        "Mean DB": float(compute_disturbance(accepted, splits, tm)),
                        "decision_ms": round(float(decision_ms), 3),
                        "selected_K": int(selected_k),
                        "action": ANAME[a],
                        "disconnected_ODs": int(len(disconnected)),
                        "failed_links": int(n_failed_links(scenario)),
                    }
                )
                accepted = splits
                prev_tm = tm
            print(f"[done] {topo} {scenario}", flush=True)
    pc = pd.DataFrame(rows)
    pc.to_csv(COMPLETED / "failure_scenarios_fix1_per_cycle.csv", index=False)
    summary_rows = []
    for (scenario, topo), g in pc.groupby(["Scenario", "Topology"], sort=False):
        summary_rows.append(
            {
                "Scenario": scenario,
                "Topology": topo,
                "Mean PR": round(float(g["Mean PR"].mean() * 100.0), 3),
                "PR>=0.90": round(float(g["PR>=0.90 flag"].mean() * 100.0), 3),
                "PR(fail)": round(float(g["PR(fail)"].dropna().mean() * 100.0), 3) if g["PR(fail)"].notna().any() else "",
                "Mean DB": round(float(g["Mean DB"].mean() * 100.0), 3),
                "P95 DB": round(float(np.percentile(g["Mean DB"], 95) * 100.0), 3),
                "Max disconnected ODs": int(g["disconnected_ODs"].max()),
                "Mean ms": round(float(g["decision_ms"].mean()), 3),
                "P95 ms": round(float(np.percentile(g["decision_ms"], 95)), 3),
            }
        )
    pd.DataFrame(summary_rows).to_csv(COMPLETED / "failure_scenarios_fix1_COMPLETED.csv", index=False)


if __name__ == "__main__":
    run()
