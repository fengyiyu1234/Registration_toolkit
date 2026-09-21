#!/usr/bin/env python3
"""Scriptable coordinate tracing for the full-chain QC plan.

``--xyz`` always remains the requested viewing centre. The nearest registered
cell is reported as extra context and is never substituted for that point.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qc import region_cells as rc
from qc.coordinate_contract import global_pixel_to_physical, physical_to_label_voxel
from qc.view_detection_qc import DetectionSession
from shared import local_config


@dataclass
class CoordinateTarget:
    global_xyz: list
    physical_um: list
    label_voxel_xyz: list
    nearest_cell_id: str | None = None
    nearest_distance_um: float | None = None
    lookup_region_id: int | None = None
    boundary_depth_um: float | None = None


def resolve_target(session, global_xyz):
    """Resolve arbitrary global pixels without snapping the requested point."""
    xyz = np.asarray(global_xyz, dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError("global coordinate must be three finite values (x, y, z)")
    physical = global_pixel_to_physical(xyz, session.cell_voxel_um)
    label_xyz = physical_to_label_voxel(
        physical, session.labels.origin, session.labels.spacing)
    target = CoordinateTarget(xyz.tolist(), physical.tolist(), label_xyz.tolist())
    if len(session.cells):
        distances = np.linalg.norm(session.phys - physical, axis=1)
        i = int(np.argmin(distances))
        target.nearest_cell_id = str(session.cells.iloc[i]["cell_id"])
        target.nearest_distance_um = float(distances[i])
    target.lookup_region_id = int(session.labels.lookup(physical)[0])
    target.boundary_depth_um = float(
        rc.signed_depth_um(session.labels, session.region_ids, physical[None])[0])
    return target


def trace_cell(session, cell_id):
    rows = session.cells.index[session.cells["cell_id"] == cell_id]
    if not len(rows):
        raise ValueError(f"unknown cell_id: {cell_id}")
    row = session.cells.loc[rows[0]]
    target = resolve_target(session, row[["x", "y", "z"]].to_numpy(float))
    target.nearest_cell_id, target.nearest_distance_um = str(cell_id), 0.0
    return target


def atomic_json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_verdict(path, target, *, run_dir, cell_id=None, source_tile="",
                   source_tiff="", detection_verdict="uncertain",
                   colocalization_verdict="uncertain",
                   registration_verdict="boundary_uncertain", note=""):
    """Atomically write schema-versioned, independently judged verdicts."""
    path = Path(path)
    rows = []
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        rows = old.get("records", []) if isinstance(old, dict) else old
    rows.append({
        "schema_version": 1, "run_dir": str(run_dir),
        "target_kind": "cell" if cell_id else "coordinate", "cell_id": cell_id,
        "global_x": target.global_xyz[0], "global_y": target.global_xyz[1],
        "global_z": target.global_xyz[2], "source_tile": source_tile,
        "source_tiff": source_tiff, "table_region_id": None,
        "lookup_region_id": target.lookup_region_id,
        "boundary_depth_um": target.boundary_depth_um,
        "detection_verdict": detection_verdict,
        "colocalization_verdict": colocalization_verdict,
        "registration_verdict": registration_verdict, "note": note,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    atomic_json_write(path, {"schema_version": 1, "records": rows})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    local_config.add_config_arg(parser, "detection_qc")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--xyz", nargs=3, type=float, metavar=("X", "Y", "Z"))
    group.add_argument("--cell-id")
    args = parser.parse_args(argv)
    cfg = local_config.load_config(
        "detection_qc", args.config,
        required=("run_dir", "ontology_json", "regions", "out_dir", "detection_dir"))
    session = DetectionSession(cfg, load_boxes=False)
    target = trace_cell(session, args.cell_id) if args.cell_id else resolve_target(session, args.xyz)
    result = asdict(target)
    grids = getattr(session.source, "grids", {})
    if grids:
        # This is deliberately independent of cell lookup: --xyz can point at
        # an empty region and still returns every overlapping source tile.
        result["source_locations"] = {
            channel: grid.locate(target.global_xyz)
            for channel, grid in grids.items()
        }
    half = np.asarray(session.half_um, dtype=float)
    result["view_window_physical_um"] = {
        "lo": (np.asarray(target.physical_um) - half).tolist(),
        "hi": (np.asarray(target.physical_um) + half).tolist(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
