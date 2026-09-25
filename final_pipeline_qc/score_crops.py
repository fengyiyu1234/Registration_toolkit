#!/usr/bin/env python
"""Score the blind annotations against the pipeline's predictions: the
confusion matrix, and what it does to the group comparison.

    conda activate antsreg
    python qc/score_crops.py                      # uses configs/qc_crops.yaml
    python qc/score_crops.py --csv out.csv        # also dump per-cell matches

Run this only after every crop is finished.  It opens the sealed manifest, so
running it early and reading the output un-blinds whoever sees it.

WHAT COMES OUT, AND WHAT EACH NUMBER IS FOR
-------------------------------------------
soma recall / precision
    Whether cells are being found at all.  Affects Count and Density; largely
    cancels in Percentage and RegionProportion, because a uniformly missing
    fraction leaves ratios alone.

Sox9 sensitivity / specificity
    The one under suspicion.  Sox9 is not detected as a cell: brain_detector
    attaches it to an already-detected soma by strict bbox containment, so a
    Sox9 miss does not remove a cell, it MOVES it from X_Sox9 to X.  That has
    three consequences worth keeping straight:
      * all_cells, GFP_any and RFP_any are immune -- the cell is still there,
        only relabelled.
      * A Sox9 class's Percentage survives a spatially uniform miss rate,
        since numerator and denominator scale together.
      * Density and RegionProportion of a Sox9 class scale directly with the
        local hit rate.

sensitivity difference between groups
    The only quantity here that can manufacture a result.  If both groups miss
    Sox9 at the same rate, the effect is attenuated toward null: that costs
    statistical power and produces false negatives, never a false positive.
    A rate that DIFFERS between groups creates a group difference out of
    nothing, and with three samples a side nothing downstream can tell that
    apart from biology.  This is the number to read first.

uncertain rate
    Reported separately and excluded from every rate above.  If it is high,
    Sox9 status cannot be scored by eye in this data and no recall number from
    this exercise means anything -- fall back to comparing the 730 nm
    intensity distribution inside Sox9-negative somas against Sox9-positive
    ones, which needs no human call.

Confidence intervals are a cluster bootstrap over CROPS, not over cells.
Cells inside one crop share depth, tile, staining and local density, so they
are not independent draws; a binomial interval over pooled cells would be far
too narrow and would make a group difference look decisive when the evidence
is a handful of crops.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared import local_config  # noqa: E402

TOOL = "qc_crops"
UNCERTAIN = "uncertain"


def read_annotation(crop_dir):
    path = Path(crop_dir) / "annotation.csv"
    if not path.is_file():
        return None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({"call": row["call"], "lx": float(row["lx"]),
                         "ly": float(row["ly"]), "lz": float(row["lz"])})
    return rows


def match_points(ann, preds, voxel_um, radius_um):
    """Optimal one-to-one assignment under `radius_um`, in microns.

    Optimal rather than greedy: in dense cortex a greedy pass can chain two
    nearby cells onto each other's partner and report two errors where there
    are none.  Cost is Euclidean distance in physical space, so the 8 um z
    step is weighted as it really is instead of counting one slice as one
    pixel.
    """
    if not ann or not preds:
        return [], list(range(len(ann))), list(range(len(preds)))
    v = np.asarray(voxel_um, float)
    a = np.array([[p["lx"], p["ly"], p["lz"]] for p in ann], float) * v
    b = np.array([[p["lx"], p["ly"], p["lz"]] for p in preds], float) * v
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(np.where(cost <= radius_um, cost, 1e6))
    pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] <= radius_um]
    matched_a = {i for i, _ in pairs}
    matched_b = {j for _, j in pairs}
    return (pairs,
            [i for i in range(len(ann)) if i not in matched_a],
            [j for j in range(len(preds)) if j not in matched_b])


def score_crop(record, crop_dir, radius_um):
    ann = read_annotation(crop_dir)
    if ann is None:
        return None
    preds = record["predictions"]
    pairs, lone_ann, lone_pred = match_points(
        ann, preds, record["voxel_um_xyz"], radius_um)

    counts = dict(tp=0, fp=0, fn=0, uncertain=0,
                  s_tp=0, s_fn=0, s_fp=0, s_tn=0)
    counts["uncertain"] = sum(1 for p in ann if p["call"] == UNCERTAIN)
    counts["fp"] = len(lone_pred)
    counts["fn"] = sum(1 for i in lone_ann if ann[i]["call"] != UNCERTAIN)
    for i, j in pairs:
        if ann[i]["call"] == UNCERTAIN:
            continue
        counts["tp"] += 1
        truth = ann[i]["call"] == "sox9_pos"
        called = "sox9" in preds[j]["class_name"].lower()
        if truth and called:
            counts["s_tp"] += 1
        elif truth and not called:
            counts["s_fn"] += 1
        elif not truth and called:
            counts["s_fp"] += 1
        else:
            counts["s_tn"] += 1
    return counts


def _rate(num, den):
    return float("nan") if den == 0 else num / den


def pooled_rates(crops):
    s = {k: sum(c[k] for c in crops) for k in crops[0]} if crops else {}
    if not s:
        return {}
    return {
        "n_crops": len(crops),
        "n_annotated": s["tp"] + s["fn"] + s["uncertain"],
        "soma_recall": _rate(s["tp"], s["tp"] + s["fn"]),
        "soma_precision": _rate(s["tp"], s["tp"] + s["fp"]),
        "sox9_sensitivity": _rate(s["s_tp"], s["s_tp"] + s["s_fn"]),
        "sox9_specificity": _rate(s["s_tn"], s["s_tn"] + s["s_fp"]),
        "uncertain_rate": _rate(s["uncertain"],
                                s["tp"] + s["fn"] + s["uncertain"]),
    }


def bootstrap_ci(crops, key, rng, n=4000, alpha=0.05):
    """Percentile CI, resampling whole crops with replacement."""
    if len(crops) < 2:
        return float("nan"), float("nan")
    draws = []
    for _ in range(n):
        pick = [crops[i] for i in rng.integers(0, len(crops), len(crops))]
        value = pooled_rates(pick).get(key, float("nan"))
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return float("nan"), float("nan")
    return (float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def diff_ci(a, b, key, rng, n=4000, alpha=0.05):
    """CI on group A minus group B, resampling crops within each group."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    draws = []
    for _ in range(n):
        va = pooled_rates([a[i] for i in rng.integers(0, len(a), len(a))]).get(key)
        vb = pooled_rates([b[i] for i in rng.integers(0, len(b), len(b))]).get(key)
        if va is not None and vb is not None and np.isfinite(va) and np.isfinite(vb):
            draws.append(va - vb)
    if not draws:
        return float("nan"), float("nan")
    return (float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def _fmt(value, lo=None, hi=None):
    if not np.isfinite(value):
        return "     n/a"
    text = f"{value * 100:6.1f}%"
    if lo is not None and np.isfinite(lo):
        text += f" [{lo * 100:.1f}, {hi * 100:.1f}]"
    return text


def report(by_group, rng):
    keys = ["soma_recall", "soma_precision", "sox9_sensitivity",
            "sox9_specificity", "uncertain_rate"]
    print(f"\n{'':22s}" + "".join(f"{g:>26s}" for g in by_group))
    for key in keys:
        line = f"{key:22s}"
        for crops in by_group.values():
            r = pooled_rates(crops)
            lo, hi = bootstrap_ci(crops, key, rng)
            line += f"{_fmt(r.get(key, float('nan')), lo, hi):>26s}"
        print(line)
    line = f"{'crops / cells':22s}"
    for crops in by_group.values():
        r = pooled_rates(crops)
        line += f"{r['n_crops']:>13d} /{r['n_annotated']:>11d}"
    print(line)

    if len(by_group) == 2:
        (ga, ca), (gb, cb) = list(by_group.items())
        print(f"\n组间差（{ga} 减 {gb}），crop 层面 bootstrap 95% CI：")
        for key in ("sox9_sensitivity", "soma_recall"):
            d = pooled_rates(ca).get(key, float("nan")) - \
                pooled_rates(cb).get(key, float("nan"))
            lo, hi = diff_ci(ca, cb, key, rng)
            spans_zero = np.isfinite(lo) and lo <= 0 <= hi
            verdict = "包含 0" if spans_zero else "不含 0  <-- 两组的检出率本身就不同"
            print(f"  {key:20s} {d * 100:+6.1f}%  "
                  f"[{lo * 100:+.1f}, {hi * 100:+.1f}]  {verdict}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    local_config.add_config_arg(parser, TOOL)
    parser.add_argument("--csv", help="把逐 crop 的计数也写成一份 CSV")
    parser.add_argument("--match-radius-um", type=float, default=12.0,
                        help="标注点和预测点配对的最大距离，微米（默认 12）")
    args = parser.parse_args()

    cfg = local_config.load_config(TOOL, args.config,
                                   required=("out_dir", "manifest_path"))
    manifest = json.loads(Path(cfg["manifest_path"]).read_text(encoding="utf-8"))
    out_dir = Path(cfg["out_dir"])

    scored, missing = [], []
    for record in manifest["crops"]:
        counts = score_crop(record, out_dir / record["crop_id"],
                            args.match_radius_um)
        if counts is None:
            missing.append(record["crop_id"])
            continue
        counts.update(crop_id=record["crop_id"], sample=record["sample"],
                      group=record["group"], z_band=record["z_band"])
        scored.append(counts)

    if missing:
        print(f"⚠️  {len(missing)}/{len(manifest['crops'])} 个 crop 还没有 annotation.csv，"
              f"本次不计入：{', '.join(missing)}")
        print("    半标的 crop 不是更小的样本，是有偏的样本 —— 标完再跑。\n")
    if not scored:
        print("没有任何已标注的 crop。")
        return

    rng = np.random.default_rng(0)
    by_group = {}
    for row in scored:
        by_group.setdefault(row["group"] or "(no group)", []).append(row)
    report(by_group, rng)

    bands = {}
    for row in scored:
        bands.setdefault(row["z_band"], []).append(row)
    print("\n按成像深度（z_band 0 最浅）：")
    for band in sorted(bands):
        r = pooled_rates(bands[band])
        print(f"  band {band}  crops {r['n_crops']:2d}  "
              f"sox9_sensitivity {_fmt(r['sox9_sensitivity'])}  "
              f"soma_recall {_fmt(r['soma_recall'])}")

    if args.csv:
        fields = ["crop_id", "sample", "group", "z_band", "tp", "fp", "fn",
                  "uncertain", "s_tp", "s_fn", "s_fp", "s_tn"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: row[k] for k in fields} for row in scored)
        print(f"\n逐 crop 计数 -> {args.csv}")


if __name__ == "__main__":
    main()
