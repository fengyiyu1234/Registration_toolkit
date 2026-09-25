#!/usr/bin/env python
"""Blind exhaustive annotation of one crop, in napari.

    conda activate antsreg
    python qc/annotate_crop.py /path/to/crops/a7f3c91e

EXHAUSTIVE, not a spot check.  Mark EVERY cell you can see in the volume,
including the obvious ones and including the ones you assume the pipeline
already found.  The point of the exercise is the denominator: recall is
"cells found / cells there", and "cells there" only exists if the crop is
annotated completely.  A crop annotated most of the way through is not a
smaller sample, it is a biased one, so finish a crop or discard it.

WHY THIS TOOL IS SEPARATE FROM THE PIPELINE'S OWN VIEWER
--------------------------------------------------------
brain_detector's visualize.py is built to display detections, and it is
organised per tile.  Neither fits here.  A crop crosses tile boundaries, and
an annotation session that has the predictions loaded is not blind no matter
which layers are switched off.  So this loads a self-contained crop directory
and there is deliberately no code path in this file that can reach a sample
id, a group label, or a prediction -- all of that lives in the sealed manifest
qc/cut_crops.py wrote somewhere else entirely.

Use visualize.py afterwards, on the disagreements qc/score_crops.py reports,
to work out WHY a cell was missed.  Different question, different tool.

THE THREE CALLS

    1  soma, Sox9 negative     no nucleus signal inside this cell
    2  soma, Sox9 positive     a nucleus you would call positive
    3  uncertain               you genuinely cannot tell

`uncertain` is not a way of avoiding a decision, it is one of the results.
Forcing a binary call on a cell you cannot read pushes the error toward
whichever way you lean, and the rate of uncertain calls is itself the answer
to whether Sox9 status can be scored by eye in this data at all.  If that rate
comes out high, the recall number was never going to mean anything and the
honest fallback is the intensity-distribution route instead.

KEYS

    1 / 2 / 3   make that layer active, then click to place points
    s           save to annotation.csv in the crop directory
    d           toggle the currently active layer's visibility

Saving is manual and repeatable.  Re-running on a crop that already has an
annotation.csv loads it back, so a crop can be finished across sessions.
"""
import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

# Mirrors brain_detector's CHANNEL_VIS so a crop looks like the tiles do in
# the pipeline's own viewer.  Copied rather than imported: this repo depends on
# registration_ants, not on brain_detector, and five lines of colour are not
# worth a second cross-repo dependency.
CHANNEL_COLORMAP = {"RFP": "red", "GFP": "green", "Sox9": "cyan", "Olig2": "magenta"}

CALLS = [
    ("sox9_neg", "1  soma, Sox9 NEGATIVE", "dodgerblue"),
    ("sox9_pos", "2  soma, Sox9 POSITIVE", "orange"),
    ("uncertain", "3  UNCERTAIN", "lightgray"),
]
ANNOTATION_CSV = "annotation.csv"


def load_crop(crop_dir):
    crop_dir = Path(crop_dir)
    meta_path = crop_dir / "crop.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"{crop_dir} 里没有 crop.json —— 这不是 qc/cut_crops.py 切出来的 crop 目录。")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    volumes = {}
    for ch in meta["channels"]:
        path = crop_dir / f"{ch}.tif"
        if path.is_file():
            volumes[ch] = tifffile.imread(str(path))
    if not volumes:
        raise FileNotFoundError(f"{crop_dir} 里一个通道的 tif 都没有。")
    return meta, volumes


def read_existing(crop_dir):
    """-> {call: (N, 3) array of (z, y, x)}, so a crop can be resumed."""
    path = Path(crop_dir) / ANNOTATION_CSV
    out = {call: [] for call, _, _ in CALLS}
    if not path.is_file():
        return {k: np.empty((0, 3)) for k in out}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["call"] in out:
                out[row["call"]].append(
                    [float(row["lz"]), float(row["ly"]), float(row["lx"])])
    return {k: (np.array(v) if v else np.empty((0, 3))) for k, v in out.items()}


def save_annotation(crop_dir, meta, layers, annotator):
    path = Path(crop_dir) / ANNOTATION_CSV
    rows = []
    for call, _, _ in CALLS:
        for z, y, x in np.asarray(layers[call].data).reshape(-1, 3):
            rows.append({"crop_id": meta["crop_id"], "call": call,
                         "lx": round(float(x), 2), "ly": round(float(y), 2),
                         "lz": round(float(z), 2)})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["crop_id", "call", "lx", "ly", "lz"])
        writer.writeheader()
        writer.writerows(rows)

    (Path(crop_dir) / "annotation.meta.json").write_text(json.dumps({
        "crop_id": meta["crop_id"],
        "annotator": annotator,
        "saved": datetime.now().isoformat(timespec="seconds"),
        "counts": {call: int(len(layers[call].data)) for call, _, _ in CALLS},
    }, indent=2), encoding="utf-8")

    counts = ", ".join(f"{call} {len(layers[call].data)}" for call, _, _ in CALLS)
    print(f"saved {path}  ({counts})")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("crop_dir", help="qc/cut_crops.py 切出来的一个 crop 目录")
    parser.add_argument("--annotator", default=os.environ.get("USER", ""),
                        help="记进 annotation.meta.json，用于算标注者之间的一致性")
    args = parser.parse_args()

    import napari

    meta, volumes = load_crop(args.crop_dir)
    vx, vy, vz = meta["voxel_um_xyz"]
    scale = (vz, vy, vx)   # napari axis order is (z, y, x)

    viewer = napari.Viewer(title=f"crop {meta['crop_id']}")
    for ch, vol in volumes.items():
        viewer.add_image(
            vol, name=ch, scale=scale, blending="additive",
            colormap=CHANNEL_COLORMAP.get(ch, "gray"),
            contrast_limits=(float(np.percentile(vol, 1)),
                             float(np.percentile(vol, 99.9))))

    existing = read_existing(args.crop_dir)
    layers = {}
    for call, label, color in CALLS:
        layers[call] = viewer.add_points(
            existing[call], name=label, ndim=3, scale=scale,
            size=max(vx, vy) * 12, face_color=color, border_color="black",
            out_of_slice_display=True)

    def _activate(call):
        layer = layers[call]
        viewer.layers.selection = {layer}
        layer.visible = True
        layer.mode = "add"

    for key, (call, _, _) in zip("123", CALLS):
        viewer.bind_key(key, lambda v, c=call: _activate(c), overwrite=True)

    @viewer.bind_key("s", overwrite=True)
    def _save(v):
        save_annotation(args.crop_dir, meta, layers, args.annotator)

    @viewer.bind_key("d", overwrite=True)
    def _toggle(v):
        for layer in list(viewer.layers.selection):
            layer.visible = not layer.visible

    _activate(CALLS[0][0])
    total = sum(len(existing[c]) for c, _, _ in CALLS)
    print(f"crop {meta['crop_id']}  size {meta['size_px_xyz']} px  "
          f"channels {', '.join(volumes)}")
    if total:
        print(f"载入已有标注 {total} 个点，可以接着标。")
    print("1/2/3 选层，点击加点，s 保存。标完整个体积，不要只标一部分。")
    napari.run()


if __name__ == "__main__":
    main()
