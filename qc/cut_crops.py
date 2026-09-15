#!/usr/bin/env python
"""Cut blind annotation crops out of the full-resolution tiles, for measuring
how many cells the detector actually misses.

WHY CROPS AND NOT A LIST OF CELLS
---------------------------------
Sampling from the detector's own output and checking each hit measures
precision and nothing else: a cell the pipeline never reported has zero chance
of entering the sample, so the quantity actually in doubt -- the Sox9 false
negatives -- is invisible to that design by construction.  Sampling a VOLUME
and annotating every cell in it from scratch changes the denominator from
"what the pipeline said" to "what is there", which is what recall needs.

The sampling unit is therefore the crop, not the cell.  Cells inside one crop
share depth, tile, staining and local density, so they are correlated; the
confidence interval on recall has to be taken across crops, which is why this
cuts a dozen small ones rather than two big ones.

WHAT MAKES IT BLIND
-------------------
`out_dir` receives only images.  Nothing under it records which sample or
which group a crop came from, and no predictions are written there.  All of
that goes to `manifest_path`, which this refuses to place inside `out_dir`.
The annotation tool (qc/annotate_crop.py) takes a crop directory and has no
code path that reads a manifest, so blinding is a property of what the
annotator is handed rather than of their self-discipline.

Ship `out_dir` to whoever annotates.  Keep `manifest_path` behind.

USAGE (antsreg env, on the machine that holds the full-resolution tiles)

    cp configs/qc_crops.example.yaml configs/qc_crops.yaml   # once, then edit
    python qc/cut_crops.py                        # uses configs/qc_crops.yaml
    python qc/cut_crops.py configs/other.yaml     # or an explicit one
    python qc/cut_crops.py --dry-run              # pick sites, cut nothing

READ THE SIGNAL CHECK IT PRINTS.  Every crop reports the mean intensity at the
pipeline's predicted cell centres over the mean at random positions in the
same crop.  A correct cut scores well above 1.  Around 1 means the box being
read is not the box the predictions describe -- a wrong z base, a channel
whose tiles are offset, the wrong merging xml -- and that failure is otherwise
completely silent, because the crop still contains real tissue and still looks
like a brain.  Fix it before anyone annotates anything.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qc import crop_geometry as geom  # noqa: E402
from shared import local_config  # noqa: E402

TOOL = "qc_crops"

# Columns 0-2 are the raw global pixel centroid; 9 is the raw CCF id.  Written
# header-less by registration_ants.cell_points -- see stats/cell_tables.py in
# the pipeline repo for the same reader and why column 9 is not a graph_order.
PRED_COLUMNS = ["x", "y", "z", "xr", "yr", "zr", "xt", "yt", "zt",
                "region_id", "region_name", "slice_name", "tile_name", "score"]


def find_labels_volume(run_dir):
    hits = sorted(Path(run_dir).glob("*_labels_in_sample.nii.gz"))
    if not hits:
        raise FileNotFoundError(
            f"{run_dir} 里没有 *_labels_in_sample.nii.gz。"
            "这个文件就是反向映射本身，没有它选不了 crop。")
    return hits[0]


def load_predictions(run_dir):
    """-> DataFrame of every registered cell in this run, with its class.

    The class comes from the directory name, which is how brain_detector
    records a physical cell's one composite label, e.g. glia_GFP_Sox9.
    """
    frames = []
    for csv_path in sorted(Path(run_dir).glob("cell_registration/*/cell_registration.csv")):
        try:
            df = pd.read_csv(csv_path, header=None, names=range(20), engine="python")
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        for i, name in enumerate(PRED_COLUMNS):
            if i in df.columns:
                df = df.rename(columns={i: name})
        df = df[["x", "y", "z", "region_id", "score"]].copy()
        df["class_name"] = csv_path.parent.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"{run_dir}/cell_registration/ 下没有读到任何细胞。")
    out = pd.concat(frames, ignore_index=True)
    for col in ("x", "y", "z"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["x", "y", "z"]).reset_index(drop=True)


def predictions_in_box(preds, origin_px, size_px):
    """Predictions inside the crop, with coordinates rebased to crop-local."""
    o, s = np.asarray(origin_px, int), np.asarray(size_px, int)
    m = ((preds["x"] >= o[0]) & (preds["x"] < o[0] + s[0]) &
         (preds["y"] >= o[1]) & (preds["y"] < o[1] + s[1]) &
         (preds["z"] >= o[2]) & (preds["z"] < o[2] + s[2]))
    sub = preds.loc[m].copy()
    sub["lx"] = sub["x"] - o[0]
    sub["ly"] = sub["y"] - o[1]
    sub["lz"] = sub["z"] - o[2]
    return sub


def resolve_paths(cfg):
    out_dir = Path(cfg["out_dir"]).resolve()
    manifest_path = Path(cfg["manifest_path"]).resolve()
    if out_dir == manifest_path.parent or out_dir in manifest_path.parents:
        raise ValueError(
            f"manifest_path 不能放在 out_dir 里面。\n"
            f"  out_dir       {out_dir}\n"
            f"  manifest_path {manifest_path}\n"
            "out_dir 是要整个交给标注者的，清单一旦在里面，盲法就只剩自觉了。")
    return out_dir, manifest_path


def cut_sample(sample_name, sample_cfg, cfg, out_dir, rng, dry_run=False):
    from registration_ants.atlas_utils import load_ccf_ontology_json

    run_dir = Path(sample_cfg["run_dir"])
    labels_path = find_labels_volume(run_dir)
    labels_img = nib.load(str(labels_path))
    labels = np.asarray(labels_img.dataobj).astype(np.int64)
    label_spacing = np.array(labels_img.header.get_zooms()[:3], float)

    cell_voxel = np.array(cfg["cell_voxel_um"], float)
    crop_px = np.array(cfg["crop_px"], int)

    structures = load_ccf_ontology_json(cfg["ontology_json"])
    ids = set()
    for name in cfg["regions"]:
        ids |= geom.descendant_ids(structures, name)

    sites = geom.pick_crop_sites(
        labels, ids, crop_px, cell_voxel, label_spacing,
        n_crops=int(sample_cfg.get("n_crops", cfg.get("n_crops_per_sample", 3))),
        n_z_bands=int(cfg.get("z_bands", 3)), rng=rng)

    preds = load_predictions(run_dir)
    grids = {}
    if not dry_run:
        for ch_name, ch in sample_cfg["channels"].items():
            ch = {"dir": ch} if isinstance(ch, str) else dict(ch)
            grids[ch_name] = geom.TileGrid(ch["dir"], ch.get("xml"),
                                           ch.get("offset_px", (0, 0, 0)))
    channel_names = list(grids) or list(sample_cfg["channels"])
    anchor = cfg.get("signal_check_channel") or channel_names[0]
    records = []
    for site in sites:
        crop_id = f"{rng.integers(0, 16 ** 8):08x}"
        crop_dir = out_dir / crop_id
        if not dry_run:
            crop_dir.mkdir(parents=True, exist_ok=True)

        in_box = predictions_in_box(preds, site["origin_px"], crop_px)
        ratio = float("nan")
        for ch_name, grid in grids.items():
            vol = grid.cut(site["origin_px"], crop_px)
            tifffile.imwrite(crop_dir / f"{ch_name}.tif", vol,
                             photometric="minisblack")
            if ch_name == anchor:
                soma = in_box[~in_box["class_name"].str.contains("Sox9", case=False)]
                ratio, _ = geom.signal_check(
                    vol, soma[["lx", "ly", "lz"]].to_numpy(float), rng)

        # Written into the crop: only what the annotator needs to see the
        # tissue.  No sample, no group, no region, no predictions.
        if not dry_run:
            (crop_dir / "crop.json").write_text(json.dumps({
                "crop_id": crop_id,
                "channels": channel_names,
                "size_px_xyz": crop_px.tolist(),
                "voxel_um_xyz": cell_voxel.tolist(),
            }, indent=2), encoding="utf-8")

        records.append({
            "crop_id": crop_id,
            "sample": sample_name,
            "group": sample_cfg.get("group", ""),
            "run_dir": str(run_dir),
            "origin_px_xyz": np.asarray(site["origin_px"], int).tolist(),
            "size_px_xyz": crop_px.tolist(),
            "voxel_um_xyz": cell_voxel.tolist(),
            "region_id": site["region_id"],
            "region_name": structures.get(site["region_id"], {}).get("name", ""),
            "z_band": site["z_band"],
            "signal_check_ratio": None if not np.isfinite(ratio) else round(ratio, 2),
            "predictions": [
                {"lx": float(r.lx), "ly": float(r.ly), "lz": float(r.lz),
                 "class_name": r.class_name, "score": float(r.score)
                 if pd.notna(r.score) else None}
                for r in in_box.itertuples()],
        })
        if dry_run:
            origin = ",".join(str(int(v)) for v in site["origin_px"])
            print(f"  z_band {site['z_band']}  origin [{origin}]  "
                  f"cells {len(in_box):4d}  "
                  f"{structures.get(site['region_id'], {}).get('name', '')}")
        else:
            flag = "" if np.isfinite(ratio) and ratio >= 1.5 else "   <-- CHECK"
            print(f"  {crop_id}  z_band {site['z_band']}  "
                  f"cells {len(in_box):4d}  signal {ratio:5.2f}{flag}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    local_config.add_config_arg(parser, TOOL)
    parser.add_argument("--dry-run", action="store_true",
                        help="选点、数预测细胞并打印，但不碰 tile、不写任何文件。"
                             "只需要 run_dir，所以在没挂载全分辨率数据的机器上也能跑")
    args = parser.parse_args()

    cfg = local_config.load_config(
        TOOL, args.config,
        required=("out_dir", "manifest_path", "ontology_json", "regions",
                  "cell_voxel_um", "crop_px", "samples"))
    out_dir, manifest_path = resolve_paths(cfg)
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    if args.dry_run:
        print("--dry-run: 选点并统计，不读 tile 也不写文件\n")

    records = []
    for sample_name, sample_cfg in cfg["samples"].items():
        print(f"\n[{sample_name}]  group={sample_cfg.get('group', '')}")
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
        records += cut_sample(sample_name, sample_cfg, cfg, out_dir, rng,
                              dry_run=args.dry_run)

    if args.dry_run:
        counts = [len(r["predictions"]) for r in records]
        if counts:
            print(f"\n{len(counts)} 个 crop，预测细胞数 中位 {int(np.median(counts))}，"
                  f"范围 {min(counts)}-{max(counts)}，合计 {sum(counts)}。")
            print("这就是人工要判的量级。太少就把 crop_px 调大，"
                  "别靠增加 crop 数来凑 —— 但也别调到一个 crop 标不完。")
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "out_dir": str(out_dir),
        "config": {k: v for k, v in cfg.items() if k != "samples"},
        "crops": records,
    }, indent=2), encoding="utf-8")

    low = [r for r in records if (r["signal_check_ratio"] or 0) < 1.5]
    print(f"\n{len(records)} 个 crop -> {out_dir}")
    print(f"封存清单 -> {manifest_path}")
    print("\n把 out_dir 整个交给标注者，清单留在自己这里。")
    if low:
        print(f"\n⚠️  {len(low)} 个 crop 的 signal check 低于 1.5："
              f"{', '.join(r['crop_id'] for r in low)}\n"
              "    切出来的框和预测描述的不是同一块组织。先查 z 起始约定、"
              "通道 offset_px、以及用的是不是这个通道自己的 merging xml，"
              "再让任何人开始标注。")


if __name__ == "__main__":
    main()
