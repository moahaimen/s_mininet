#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

from te.parser_sndlib import (
    DATASET_SPECS,
    build_tm_matrix,
    discover_tm_files,
    parse_native_topology,
    safe_extract_tgz,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_TOPO_DIR = DATA_DIR / "raw" / "topology"
RAW_DYNAMIC_DIR = DATA_DIR / "raw" / "dynamic"
PROCESSED_DIR = DATA_DIR / "processed"
SNDLIB_BASE_URL = "https://sndlib.put.poznan.pl/"
DEFAULT_MAX_STEPS = {
    "abilene": 2050,
    "geant": 700,
    "germany50": 300,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap a real SNDlib dataset into data/processed/*.npz")
    parser.add_argument("--topology", choices=sorted(DATASET_SPECS.keys()), required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def _download(url: str, dest: Path, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[download] keeping existing {dest}", flush=True)
        return
    print(f"[download] {url} -> {dest}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    dest.write_bytes(payload)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def build_dataset(topology: str, *, max_steps: int | None, force_download: bool, force_rebuild: bool) -> Path:
    spec = DATASET_SPECS[topology]
    topology_url = f"{SNDLIB_BASE_URL}download/sndlib-networks-native/{spec['topology_file']}"
    dynamic_url = f"{SNDLIB_BASE_URL}download/{spec['dynamic_archive']}"

    topology_path = RAW_TOPO_DIR / spec["topology_file"]
    archive_path = RAW_DYNAMIC_DIR / spec["dynamic_archive"]
    archive_stem = spec["dynamic_archive"][:-4] if spec["dynamic_archive"].endswith(".tgz") else spec["dynamic_archive"]
    extract_dir = RAW_DYNAMIC_DIR / "extracted" / archive_stem
    processed_path = PROCESSED_DIR / f"{topology}.npz"

    _download(topology_url, topology_path, force=force_download)
    _download(dynamic_url, archive_path, force=force_download)
    safe_extract_tgz(archive_path, extract_dir, force=force_rebuild)

    topo_data = parse_native_topology(topology_path)
    od_pairs = [(src, dst) for src in topo_data.nodes for dst in topo_data.nodes if src != dst]
    tm_files = discover_tm_files(extract_dir)
    tm_matrix, used_tm_files = build_tm_matrix(tm_files, od_pairs, max_steps=max_steps)

    metadata = {
        "dataset_key": topology,
        "dynamic_archive": spec["dynamic_archive"],
        "topology_file": _rel(topology_path),
        "normalization_rule": topo_data.normalization_rule,
        "num_tm_files_discovered": int(len(tm_files)),
        "num_tm_files_used": int(len(used_tm_files)),
        "max_steps": None if max_steps is None else int(max_steps),
        "source_topology_url": topology_url,
        "source_dynamic_url": dynamic_url,
    }

    payload = {
        "nodes": np.asarray(topo_data.nodes, dtype=object),
        "edge_src": np.asarray([edge.src for edge in topo_data.links], dtype=object),
        "edge_dst": np.asarray([edge.dst for edge in topo_data.links], dtype=object),
        "capacities": np.asarray([edge.capacity for edge in topo_data.links], dtype=np.float32),
        "weights": np.asarray([edge.weight for edge in topo_data.links], dtype=np.float32),
        "od_src": np.asarray([src for src, _ in od_pairs], dtype=object),
        "od_dst": np.asarray([dst for _, dst in od_pairs], dtype=object),
        "tm": np.asarray(tm_matrix, dtype=np.float32),
        "tm_files": np.asarray([_rel(path) for path in used_tm_files], dtype=object),
        "metadata_json": np.asarray(json.dumps(metadata), dtype=object),
    }

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(processed_path, **payload)
    print(
        f"[build] wrote {processed_path} "
        f"(nodes={len(topo_data.nodes)} directed_edges={len(topo_data.links)} "
        f"od_pairs={len(od_pairs)} tm_steps={tm_matrix.shape[0]})",
        flush=True,
    )
    return processed_path


def main() -> None:
    args = parse_args()
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = DEFAULT_MAX_STEPS.get(args.topology)
    build_dataset(
        args.topology,
        max_steps=max_steps,
        force_download=bool(args.force_download),
        force_rebuild=bool(args.force_rebuild),
    )


if __name__ == "__main__":
    main()
