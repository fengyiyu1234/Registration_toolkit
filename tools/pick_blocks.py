#!/usr/bin/env python
"""Pick the fixed atlas-space cubes a block-based group comparison samples, and
show where they landed.

WHY BLOCKS INSTEAD OF REGIONS
-----------------------------
A per-region Density is cells divided by that region's volume IN SAMPLE SPACE,
so it carries every per-sample scale difference clearing left behind.  On this
dataset that is not hypothetical: whole-brain volume runs from 49.1 to 82.4 mm3
across six same-age brains, 26 of 28 level-2/3 structures come out smaller in
one group by 13-23%, while RelativeVolume is flat -- a global scale difference,
not regional atrophy.  Roughly half of each significant Density effect is that
denominator.

A cube fixed in ATLAS space has the same volume in every sample, so dividing
by it changes no statistic: the metric becomes "cells in one fixed anatomical
territory", immune to shrinkage by construction.  What it is NOT immune to is
tissue missing inside the block, which is why every block carries a per-sample
coverage figure and why blocks are placed away from the fragile dorsal surface.

WHY CUBES AND WHY THIS SIZE (measured on DeMBA P5, not assumed)
--------------------------------------------------------------
Isocortex's largest inscribed sphere has radius 649 um and the median inside
cortex is 200 um, so an 800 um cube fits nowhere inside it and neither does any
large flat slab -- 1600x1600x300 has 1163 legal positions, 2000x2000x300 has
none.  The limit is the sheet's curvature, not its thickness.

Cuboids buy volume per block and pay for it in block count: 1200x1200x300 is
0.43 mm3 but only 7 non-overlapping positions, against 0.216 mm3 and 27 for a
600 um cube.  That trade is not worth taking here, because cells per block is
never the binding constraint -- a 500 um cube already holds ~900 reporter+
cells, a 3% Poisson CV -- while a block thin in one axis loses a third to four
fifths of its intended tissue to a 100-250 um registration error, against a
sixth to two fifths for a cube.  A cube has no short axis.

WHAT "SAME BLOCK IN EVERY SAMPLE" ACTUALLY REQUIRES
---------------------------------------------------
The samples are not all on one atlas grid.  s10 is a left hemisphere prepared
with orientation [-1, 3, -2] and slicing [285,514]/[44,365]; the other five use
[1, 3, 2] and [285,510]/[35,356], a different shape.  Indexing all six with the
same voxel numbers would silently displace s10's blocks.

So blocks are defined once on the NATIVE atlas grid and then pushed through
each sample's own orientation, slicing and padding, using
registration_ants.atlas_utils' own functions rather than an inverse derived
here.  A block that any sample's preparation would clip is dropped, which also
takes care of the hemisphere: a block in the wrong half does not survive the
slicing.

USAGE

    cp configs/pick_blocks.example.yaml configs/pick_blocks.yaml   # once
    conda activate antsreg
    python tools/pick_blocks.py                    # pick, write, draw PNGs
    python tools/pick_blocks.py --napari           # + browse them in 3D
    python tools/pick_blocks.py --dry-run          # count legal positions only

Writes into out_dir:
    blocks.json    the definitive record: native + per-sample voxel bounds,
                   region composition.  Archive this with the results -- it IS
                   the pre-registration of where the sampling looked.
    blocks.csv     the same thing flat, one row per block
    blocks_*.png   atlas sections with the block footprints drawn on them
"""
import argparse
import csv
import json
import sys
import re
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.local_config import add_config_arg, load_config  # noqa: E402

TOOL = "pick_blocks"


# ── Atlas preparation, reusing the pipeline's own steps ───────────────────────

def variant_params(pipeline_config_path):
    """Read one sample's atlas prep parameters out of its Registration_ants
    config, rather than copying them into this tool's config where they would
    drift away from what the registration actually used."""
    cfg = yaml.safe_load(Path(pipeline_config_path).read_text(encoding="utf-8"))
    source = cfg["atlas"]["source"]
    variant = cfg["atlas_variants"][source]
    return {
        "source": source,
        "orientation": variant.get("orientation"),
        "slicing": variant.get("slicing"),
        "margin": variant.get("background_margin_voxels"),
    }


def ml_axis_of(annotation):
    """The mediolateral axis: the one whose two halves mirror each other.

    Derived, not assumed. On the DeMBA P5 native grid this comes out as axis 0
    with ~93% of voxels matching across the midline, which is the same fact
    stats/region_maps.py records for its own use.
    """
    best, best_score = None, -1.0
    for axis in range(3):
        n = annotation.shape[axis]
        half = n // 2
        lo = np.take(annotation, range(half), axis=axis) > 0
        hi = np.flip(np.take(annotation, range(n - half, n), axis=axis), axis=axis) > 0
        score = float((lo == hi).mean())
        if score > best_score:
            best, best_score = axis, score
    return int(best), best_score


def mirrors_ml(params, ml_axis):
    """Does this sample's orientation flip the mediolateral axis?

    True for a sample imaged as the OTHER hemisphere: the pipeline mirrors the
    atlas onto it, which is the whole point -- with systemic delivery the two
    hemispheres are equivalent, so a left-hemisphere sample is compared against
    right-hemisphere ones by flipping the atlas rather than the data.

    It matters here because a block defined on one native hemisphere does not
    survive the other hemisphere's slicing at all. Blocks therefore have to be
    pre-mirrored for such a sample, so that every sample ends up sampling the
    same position in its OWN hemisphere.
    """
    orientation = params.get("orientation") or []
    return any(o < 0 and abs(int(o)) - 1 == ml_axis for o in orientation)


def prepare_like_pipeline(arr_xyz, annotation_xyz, params):
    """Apply one variant's reorient + slice + pad to `arr_xyz`.

    `annotation_xyz` is the native annotation, needed because the pad width is
    measured off the ANNOTATION's tissue extent after slicing and then applied
    identically to everything else -- exactly as prepare_custom_atlas does it.
    """
    from registration_ants.atlas_utils import (
        _parse_slicing, background_pad_width, reorient_volume)

    out = reorient_volume(arr_xyz, params["orientation"])
    ann = reorient_volume(annotation_xyz, params["orientation"])
    slicing = _parse_slicing(params["slicing"])
    if slicing is not None:
        out, ann = out[slicing], ann[slicing]
    if params["margin"]:
        pad = background_pad_width(ann, params["margin"])
        out = np.pad(out, pad, mode="constant")
    return out


def dorsal_axis(annotation, region_mask):
    """Which native axis is dorsoventral, and whether dorsal is the low end.

    The axis is the SHORTEST one: a mouse brain is longer front-to-back and
    wider left-to-right than it is tall, and the DeMBA P5 native grid comes out
    (ML, DV, AP) = (570, 400, 563) after _read_atlas_array_xyz.  Picking it by
    "where is cortex furthest from the brain centroid" does not work, because
    cortex is displaced further along AP than along DV on this atlas.

    Which end is dorsal is then read off the target structure: on this grid
    Isocortex's centroid sits at DV 151 against the whole brain's 202, so
    dorsal is the LOW index.  A displacement too small to call raises rather
    than guessing, since guessing wrong excludes the wrong side in silence.
    """
    tissue = np.argwhere(annotation > 0)
    region = np.argwhere(region_mask)
    if len(region) == 0:
        raise ValueError("目标结构是空的，无法判断背腹轴。")
    axis = int(np.argmin(annotation.shape))
    offset = float(region[:, axis].mean() - tissue[:, axis].mean())
    if abs(offset) < 0.05 * annotation.shape[axis]:
        raise ValueError(
            f"目标结构在轴 {axis} 上相对全脑质心只偏移了 {offset:.1f} 个体素，"
            "分不出哪一端是背侧。换一个明确偏背侧的 regions，或者别用 "
            "avoid_dorsal_um。")
    return axis, bool(offset < 0)


# ── Block placement ───────────────────────────────────────────────────────────

def region_ids(structures, names):
    out = set()
    by_name = {info.get("name"): int(sid) for sid, info in structures.items()}
    for name in names:
        if name not in by_name:
            raise ValueError(f"没有叫 {name!r} 的图谱结构，检查拼写。")
        root = by_name[name]
        out |= {int(sid) for sid, info in structures.items()
                if root in [int(v) for v in info.get("structure_id_path", [])]}
    return out


def legal_centres(mask, block_vox, min_purity):
    """Voxels where a block_vox cube is at least `min_purity` inside `mask`.

    min_purity 1.0 uses a minimum filter, which is exact: the cube is legal
    only where every one of its voxels is in the mask.  Anything less uses a
    box mean, and the block then carries a mixture -- fair, because every
    sample sees the SAME mixture, but it has to be reported.
    """
    from scipy import ndimage
    if min_purity >= 1.0:
        return ndimage.minimum_filter(mask.astype(np.uint8), size=block_vox,
                                      mode="constant") > 0
    frac = ndimage.uniform_filter(mask.astype(np.float32), size=block_vox,
                                  mode="constant")
    return frac >= float(min_purity)


def pack_blocks(legal, block_vox, n_blocks, rng, spacing_scale=1.0):
    """Greedy non-overlapping placement over a shuffled candidate list.

    Non-overlap is enforced on the Chebyshev distance between centres, which is
    exactly the condition for two axis-aligned cubes of the same size to be
    disjoint.  `spacing_scale` above 1 leaves a gap between blocks; the point
    of a gap is that neighbouring blocks should not share the same registration
    error, and touching cubes largely do.
    """
    cand = np.argwhere(legal)
    if len(cand) == 0:
        return []
    rng.shuffle(cand)
    need = int(round(block_vox * spacing_scale))
    chosen = []
    for c in cand:
        if n_blocks and len(chosen) >= n_blocks:
            break
        if all(np.max(np.abs(c - p)) >= need for p in chosen):
            chosen.append(c)
    return [np.asarray(c, int) for c in chosen]


def composition(annotation, structures, centre, block_vox, top=4):
    half = block_vox // 2
    lo = np.maximum(centre - half, 0)
    hi = np.minimum(centre - half + block_vox, annotation.shape)
    box = annotation[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    total = box.size
    counts = Counter(int(v) for v in box.ravel())
    rows = []
    for sid, n in counts.most_common(top + 1):
        if sid == 0:
            continue
        rows.append({"id": sid,
                     "name": structures.get(sid, {}).get("name", f"id {sid}"),
                     "pct": round(100.0 * n / total, 1)})
        if len(rows) >= top:
            break
    return rows


# ── Columns: boxes whose long axis follows the cortical normal ────────────────
#
# An axis-aligned cube required to sit entirely inside Isocortex cannot reach
# the upper layers, and shrinking it does not fix that. Measured on DeMBA P5,
# as a percentage of block volume:
#
#     block                 L1   L2/3    L4     L5    L6a
#     600 um cube          3.0    7.4   2.5   39.3   47.7
#     200 um cube          2.2   16.6   7.8   40.3   33.0
#     300 um column        7.7   17.9   4.5   30.6   38.8
#     whole Isocortex     13.0   20.8   5.1   29.2   29.4
#
# A smaller cube does pick up the middle layers, but L1 stays at ~2% at every
# size, because L1 is the outermost shell and "entirely inside the mask" always
# shaves it off. A box whose top face is parallel to the pia does not have that
# problem.
#
# The payoff is not only composition. Sliced into axial slabs, a column gives a
# LAYER PROFILE per block, which is the quantity a radial migration phenotype
# actually lives in. Dominant-layer purity of such a slab, measured:
#
#     300 um column, true 3D normal      87.4%   depth spread p10-p90  0.038
#     300 um column, locked to coronal   84.3%                         0.092
#     600 um cube, slabs along DV        67.6%                         0.275
#
# Locking the tilt to the coronal plane barely changes how many columns fit
# (49.0% of candidates against 44.6%), so the fit rate hides its real cost: the
# true normal is a median 23 degrees out of the coronal plane, and over an
# 800 um column that tilt smears each slab across layers. L4 spans about 0.19
# in normalised depth, so a spread of 0.038 resolves it, 0.092 half-resolves it
# and 0.275 cannot see it at all. Hence the full 3D normal.
#
# Do not push the height to the full thickness. Only thin, flat cortex can hold
# a straight box spanning 100% of it, which in practice means agranular motor
# and cingulate areas: coverage 100% returns blocks with 0.0% L4, not because
# the geometry improved but because it silently restricted sampling to the
# areas that have no L4. 90% is the measured knee.

def depth_field(annotation, region_mask, res_um):
    """-> (depth, thickness_um, normal) over the region.

    depth is 0 at the pia and 1 at the white matter, from two distance
    transforms: one to the nearest background voxel (the pia side, since cortex
    is the outermost sheet) and one to the nearest brain voxel that is NOT in
    the region (the white matter side). Their sum is the local thickness.

    The normal is the gradient of that field, smoothed first: on a 20 um grid
    the raw gradient of a distance transform is dominated by voxel staircasing,
    and the direction is the whole point here.
    """
    from scipy import ndimage
    brain = annotation > 0
    d_pia = ndimage.distance_transform_edt(brain).astype(np.float32) * res_um
    d_wm = ndimage.distance_transform_edt(~(brain & ~region_mask)).astype(np.float32) * res_um
    thickness = d_pia + d_wm
    depth = np.where(region_mask, d_pia / np.maximum(thickness, 1e-6), np.nan)
    smooth = ndimage.gaussian_filter(np.nan_to_num(depth, nan=0.5), sigma=3.0)
    grad = np.stack(np.gradient(smooth), axis=-1).astype(np.float32)
    norm = np.linalg.norm(grad, axis=-1, keepdims=True)
    return depth.astype(np.float32), thickness, grad / np.maximum(norm, 1e-8)


def orthonormal_frames(axis):
    """-> (n, t1, t2), one right-handed frame per row of `axis`.

    Which way t1 points around the axis is arbitrary and left arbitrary: the
    footprint is square and the analysis never uses the tangential directions
    individually, only the axis they are perpendicular to.
    """
    n = axis / np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-8)
    ref = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (len(n), 1))
    alt = np.tile(np.array([1.0, 0.0, 0.0], np.float32), (len(n), 1))
    ref = np.where(np.abs((n * ref).sum(1, keepdims=True)) > 0.9, alt, ref)
    t1 = np.cross(n, ref)
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-8)
    return n, t1, np.cross(n, t1)


def column_candidates(region_mask, depth, normal, band, stride, res_um,
                      axial_slack=1.25, max_um=1600.0):
    """-> (centres, frames, spans_um), one column per candidate.

    The span is found by MARCHING along the axis until the mask ends, not from
    the distance transform: the transform measures the nearest boundary in any
    direction, which in curved cortex is shorter than the distance along the
    normal. The centre is then slid to the midpoint of what the march found, so
    the axial window sits on the sheet rather than wherever the candidate voxel
    happened to be.

    The window itself is deliberately LONGER than the span (axial_slack), since
    the region mask cuts the ends. Nothing here has to get the ends right; it
    only has to make sure the window covers them.
    """
    ok = np.isfinite(depth) & (depth > band[0]) & (depth < band[1]) & region_mask
    cc = np.argwhere(ok)
    cc = cc[(cc[:, 0] % stride == 0) & (cc[:, 1] % stride == 0) & (cc[:, 2] % stride == 0)]
    if len(cc) == 0:
        empty = np.zeros((0, 3), np.float32)
        return empty, (empty, empty, empty), np.zeros(0, np.float32)
    idx = (cc[:, 0], cc[:, 1], cc[:, 2])
    n, t1, t2 = orthonormal_frames(normal[idx])
    c = cc.astype(np.float32)
    shape = np.array(region_mask.shape)

    def march(direction):
        """Voxels from each centre along the axis to where the mask ends."""
        steps = int(max_um / res_um / 0.5)
        out = np.zeros(len(c), np.float32)
        alive = np.ones(len(c), bool)
        for k in range(1, steps + 1):
            p = np.round(c + direction * n * (0.5 * k)).astype(np.int32)
            inb = np.all((p >= 0) & (p < shape), axis=1)
            good = np.zeros(len(c), bool)
            good[inb] = region_mask[p[inb, 0], p[inb, 1], p[inb, 2]]
            alive &= good
            out[alive] = 0.5 * k
            if not alive.any():
                break
        return out

    up = march(-1.0)      # towards the pia
    down = march(+1.0)    # towards the white matter
    centres = c + n * ((down - up) / 2.0)[:, None]
    spans_um = (up + down) * res_um * axial_slack
    return centres, (n, t1, t2), spans_um


def screen_columns(region_mask, centres, frames, spans_um, w_vox,
                   span_range_um=(300.0, 1600.0)):
    """-> bool mask over candidates. Cheap, and only a screen.

    There is no purity test any more: a column is intersected with the region
    mask, so every voxel it keeps is in the region by construction. Two things
    can still go wrong, and both are cheap to see. A span outside a sane range
    means the ray never crossed a proper sheet, which is what a sliver or a
    clipped corner looks like. A footprint whose corners are already outside
    the mask at mid-depth means the column is hanging off an edge rather than
    sitting on the sheet.

    Everything that survives packing is rasterised exactly afterwards and then
    held to a voxel-count floor, so a borderline candidate is settled there.
    """
    n, t1, t2 = frames
    shape = np.array(region_mask.shape)
    ok = (spans_um >= span_range_um[0]) & (spans_um <= span_range_um[1])
    half = w_vox / 2.0
    for da, db in ((half, half), (half, -half), (-half, half), (-half, -half)):
        p = np.round(centres + t1 * da + t2 * db).astype(np.int32)
        inb = np.all((p >= 0) & (p < shape), axis=1)
        good = np.zeros(len(centres), bool)
        good[inb] = region_mask[p[inb, 0], p[inb, 1], p[inb, 2]]
        ok &= good
    return ok


def column_voxels(shape, centre, frame, w_vox, h_vox, region_mask=None):
    """-> (idx_x, idx_y, idx_z) of the voxels inside one oriented column.

    Exact, by testing every voxel of the box's bounding box against the three
    local half-extents. Rasterising by oversampling the box instead would leave
    holes or duplicate work depending on the step, and the bounding box of a
    300x300x900 um column is only about 90k voxels.

    With `region_mask`, the axial extent is a generous WINDOW and the mask
    itself cuts the two ends, so the column ends on the pial surface instead of
    on a flat face. That distinction is the whole point of the shape. A flat
    square top face on an outward-curving surface can never sit on the pia: its
    corners leave the tissue first, so the face has to be pushed down by the
    corner drop, and measured on DeMBA P5 that leaves the top of the column at
    normalised depth 0.11 and the block with 1.4% L1 against 13.0% in the
    sheet. Letting the mask cut the ends puts the top at 0.03 and L1 at 9.7%.
    """
    n, t1, t2 = frame
    R = np.stack([t1, t2, n])                      # rows: the local axes
    half = np.array([w_vox / 2.0, w_vox / 2.0, h_vox / 2.0], np.float32)
    signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)],
                     np.float32)
    corners = centre + (signs * half) @ R
    lo = np.maximum(np.floor(corners.min(0)).astype(int), 0)
    hi = np.minimum(np.ceil(corners.max(0)).astype(int) + 1, shape)
    if np.any(hi <= lo):
        return None
    grid = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i]) for i in range(3)],
                                indexing="ij"), axis=-1).astype(np.float32)
    local = (grid - centre) @ R.T
    inside = np.all(np.abs(local) <= half, axis=-1)
    pts = np.argwhere(inside) + lo
    vx = (pts[:, 0], pts[:, 1], pts[:, 2])
    if region_mask is None:
        return vx
    keep = region_mask[vx]
    return vx[0][keep], vx[1][keep], vx[2][keep]


def pack_columns(centres, order, min_sep_vox, n_blocks=None):
    """Greedy non-overlapping placement on EUCLIDEAN centre distance.

    Cubes use Chebyshev because that is exactly the non-overlap condition for
    axis-aligned cubes. Two arbitrarily oriented boxes have no such closed form,
    so this uses the footprint width as a centre-to-centre minimum, which is
    conservative for the tangential directions and ignores the axial one --
    columns all sit at mid-depth, so two of them are never stacked.
    """
    cell = float(min_sep_vox)
    grid, chosen = {}, []
    for i in order:
        c = centres[i]
        key = tuple((c / cell).astype(int))
        clash = False
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for d in (-1, 0, 1):
                    prev = grid.get((key[0] + a, key[1] + b, key[2] + d))
                    if prev is not None and np.linalg.norm(c - centres[prev]) < min_sep_vox:
                        clash = True
        if clash:
            continue
        grid[key] = i
        chosen.append(i)
        if n_blocks and len(chosen) >= n_blocks:
            break
    return chosen


# ── Per-sample bounds ─────────────────────────────────────────────────────────

def sample_bounds(block_volume_native, annotation_native, params, n_blocks,
                  block_vox, ml_axis):
    """-> {block_index: bounds or None}. None means this sample's atlas
    preparation clips the block, so it cannot be compared across samples and is
    dropped upstream.

    For a mirrored sample the block volume is flipped on the native ML axis
    FIRST, and the annotation is not: the reorient then flips the blocks back,
    landing each one at the same index a non-mirrored sample puts it at. The
    annotation has to go through unflipped because the padding width is measured
    off it and must match what the registration actually used.
    """
    blocks_in = (np.flip(block_volume_native, axis=ml_axis)
                 if mirrors_ml(params, ml_axis) else block_volume_native)
    pushed = prepare_like_pipeline(blocks_in, annotation_native, params)
    expect = block_vox ** 3
    out = {}
    for i in range(1, n_blocks + 1):
        hit = np.argwhere(pushed == i)
        if len(hit) != expect:
            out[i - 1] = None
            continue
        lo, hi = hit.min(axis=0), hit.max(axis=0) + 1
        out[i - 1] = {"lo_xyz": lo.tolist(), "hi_xyz": hi.tolist()}
    return out


def sample_volume(block_volume_native, annotation_native, params, ml_axis):
    """The native block label volume pushed into one sample's prepared grid.

    Same mirroring rule as sample_bounds: a left-hemisphere sample has the
    blocks pre-flipped on the native ML axis so its own reorient flips them
    back onto the same anatomy, while the annotation goes through unflipped
    because the pad width is measured off it.
    """
    blocks_in = (np.flip(block_volume_native, axis=ml_axis)
                 if mirrors_ml(params, ml_axis) else block_volume_native)
    return prepare_like_pipeline(blocks_in, annotation_native, params)


def run_columns(cfg, annotation, structures, mask, res_um, out_dir, dry_run,
                no_figures=False):
    """Pick surface-normal columns instead of axis-aligned cubes.

    Everything downstream of the geometry is deliberately the same as the cube
    path: the columns are rasterised into one native label volume and pushed
    through each sample's own atlas preparation, so the hard part -- s10 being
    a mirrored left hemisphere on a different grid -- is solved once, in one
    place, for both block shapes.

    What cannot be the same is the per-sample record. A cube survives the trip
    as a cube, so six lo/hi bounds describe it exactly; a column does not, so
    each sample gets its LABEL VOLUME written out and stats reads that instead
    of reconstructing the shape from numbers.
    """
    from scipy import ndimage
    w_um = float(cfg.get("column_um", 300))
    band = tuple(cfg.get("depth_band", (0.45, 0.55)))
    stride = int(cfg.get("candidate_stride", 3))
    slack = float(cfg.get("axial_slack", 1.25))
    w_vox = w_um / res_um
    # A column that keeps less than this is not a column. Default: a quarter of
    # what a straight one of this footprint through median cortex would hold.
    min_voxels = int(cfg.get("min_voxels", 0.25 * (w_um / res_um) ** 2 * (900 / res_um)))

    depth, thickness, normal = depth_field(annotation, mask, res_um)
    th_ctx = thickness[mask]
    print(f"皮层厚度 中位 {np.median(th_ctx):.0f} µm "
          f"(p25 {np.percentile(th_ctx, 25):.0f}, p75 {np.percentile(th_ctx, 75):.0f})")

    centres, frames, spans_um = column_candidates(
        mask, depth, normal, band, stride, res_um, slack)
    print(f"深度 {band[0]}–{band[1]} 之间、步长 {stride} 体素的候选中心: {len(centres)}")
    if len(centres) == 0:
        print("没有候选中心。")
        return
    print(f"  沿轴射线量出的跨度中位 {np.median(spans_um) / slack:.0f} µm，"
          f"轴向窗口放宽到 {slack:.2f} 倍，两端由 mask 自己截断")

    ok = screen_columns(mask, centres, frames, spans_um, w_vox)
    print(f"{w_um:.0f} µm 截面的柱子，通过初筛: {int(ok.sum())} ({ok.mean() * 100:.1f}%)")
    if not ok.any():
        print("一个位置都放不下。把 column_um 调小。")
        return

    if cfg.get("avoid_dorsal_um"):
        # On the cube path this trims the fragile dorsal cap. A column reaches
        # the pia by design, so here it can only act on the CENTRE, and setting
        # it removes dorsal cortex outright rather than moving blocks inwards.
        axis, dorsal_low = dorsal_axis(annotation, mask)
        present = np.where((annotation > 0).any(
            axis=tuple(a for a in range(3) if a != axis)))[0]
        cut = float(cfg["avoid_dorsal_um"]) / res_um
        pos = centres[:, axis]
        ok &= (pos >= present[0] + cut) if dorsal_low else (pos <= present[-1] - cut)
        print(f"  背侧 {cfg['avoid_dorsal_um']} µm 内的柱心已排除: 剩 {int(ok.sum())}")

    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    idx = np.where(ok)[0]
    order = idx[rng.permutation(len(idx))]
    sep = w_vox * float(cfg.get("spacing_scale", 1.0))
    chosen = pack_columns(centres, order, sep, cfg.get("n_blocks"))
    print(f"互不重叠地放下了 {len(chosen)} 根柱子")
    if dry_run or not chosen:
        return

    # Exact rasterisation, with the region mask cutting both ends.
    n, t1, t2 = frames
    shape = np.array(annotation.shape)
    block_volume = np.zeros(annotation.shape, dtype=np.uint16)
    kept, voxels = [], []
    for i in chosen:
        # column_voxels takes the FULL axial extent, and spans_um is already
        # the full window, so it converts to voxels and stops there.
        vx = column_voxels(shape, centres[i], (n[i], t1[i], t2[i]),
                           w_vox, spans_um[i] / res_um, mask)
        if vx is None or len(vx[0]) < min_voxels:
            continue
        kept.append(i)
        voxels.append(vx)
    if not kept:
        print(f"没有柱子达到 min_voxels={min_voxels}。")
        return
    for label, vx in enumerate(voxels, start=1):
        block_volume[vx] = label
    med = int(np.median([len(v[0]) for v in voxels]))
    print(f"精确光栅化后保留 {len(kept)} 根，每根中位 {med} 个体素 "
          f"= {med * (res_um / 1000) ** 3:.4f} mm3")

    ml_axis, sym = ml_axis_of(annotation)
    print(f"\n左右轴 = 轴 {ml_axis}（两半对称度 {sym:.0%}）")
    native_counts = np.bincount(block_volume.ravel(), minlength=len(kept) + 1)
    pushed, survives = {}, np.ones(len(kept), bool)
    for name, pipeline_cfg in cfg["samples"].items():
        params = variant_params(pipeline_cfg)
        vol = sample_volume(block_volume, annotation, params, ml_axis)
        counts = np.bincount(vol.ravel(), minlength=len(kept) + 1)
        intact = counts[1:len(kept) + 1] == native_counts[1:len(kept) + 1]
        survives &= intact
        pushed[name] = vol
        mirror = "  [镜像半脑，柱子已预先翻转]" if mirrors_ml(params, ml_axis) else ""
        print(f"  {name:6s} orientation {params['orientation']} "
              f"slicing {params['slicing']}   被裁掉 {int((~intact).sum())} 根{mirror}")

    if not survives.all():
        print(f"\n{int((~survives).sum())} 根柱子在至少一个样本的图谱准备里被裁掉，"
              "已剔除 —— 那些柱子在各样本之间不是同一片组织。")
    final = [k for k, alive in zip(range(len(kept)), survives) if alive]
    if not final:
        print("没有柱子在所有样本里都完整。")
        return

    # Renumber to 1..N so the written volumes have no gaps, then rewrite both
    # the native volume and every pushed volume with the new labels.
    remap = np.zeros(len(kept) + 1, dtype=np.uint16)
    for new_label, old in enumerate(final, start=1):
        remap[old + 1] = new_label
    block_volume = remap[block_volume]
    out_dir.mkdir(parents=True, exist_ok=True)
    # Depth bins travel with the labels. Slicing a column by its AXIAL
    # coordinate is not the same thing: the ends are cut by the mask, so a
    # fixed axial slice near the pia spans a range of depths. Binning on the
    # depth field instead makes every slab a true depth shell, and it is the
    # depth field, not the column, that the layer question is about.
    #
    # Bin 0 means "not in any column", so bins run 1..n_bins.
    n_bins = int(cfg.get("depth_bins", 10))
    depth_bin = np.zeros(annotation.shape, dtype=np.uint8)
    inside = block_volume > 0
    dv = np.clip(depth[inside], 0.0, 1.0 - 1e-6)
    depth_bin[inside] = (dv * n_bins).astype(np.uint8) + 1

    import nibabel as nib
    label_files, depth_files = {}, {}
    for name, pipeline_cfg in cfg["samples"].items():
        params = variant_params(pipeline_cfg)
        vol = remap[pushed[name]]
        fname = f"block_labels_{name}.nii.gz"
        # Identity affine: these are index-space volumes on the sample's own
        # prepared atlas grid, and stats matches them to that grid by shape.
        nib.save(nib.Nifti1Image(vol.astype(np.uint16), np.eye(4)),
                 str(out_dir / fname))
        label_files[name] = fname
        dbin = sample_volume(depth_bin, annotation, params, ml_axis)
        dname = f"block_depth_{name}.nii.gz"
        nib.save(nib.Nifti1Image(dbin.astype(np.uint8), np.eye(4)),
                 str(out_dir / dname))
        depth_files[name] = dname

    records = []
    for new_label, old in enumerate(final, start=1):
        i = kept[old]
        vx = voxels[old]
        c = centres[i]
        counts = Counter(int(v) for v in annotation[vx])
        comp = []
        for sid, cnt in counts.most_common(5):
            if sid == 0:
                continue
            comp.append({"id": sid,
                         "name": structures.get(sid, {}).get("name", f"id {sid}"),
                         "pct": round(100.0 * cnt / len(vx[0]), 1)})
            if len(comp) >= 4:
                break
        records.append({
            "block": new_label,
            "centre_native_xyz": [round(float(v), 2) for v in c],
            "centre_mm_xyz": [round(float(v) * res_um / 1000, 3) for v in c],
            "axis_xyz": [round(float(v), 4) for v in n[i]],
            "t1_xyz": [round(float(v), 4) for v in t1[i]],
            "t2_xyz": [round(float(v), 4) for v in t2[i]],
            "footprint_um": w_um,
            "axial_window_um": round(float(spans_um[i]), 1),
            "n_voxels": int(len(vx[0])),
            "volume_mm3": round(len(vx[0]) * (res_um / 1000) ** 3, 4),
            "depth_p2_p98": [round(float(np.percentile(depth[vx], 2)), 3),
                             round(float(np.percentile(depth[vx], 98)), 3)],
            "composition": comp,
        })

    (out_dir / "blocks.json").write_text(json.dumps({
        "atlas": {"native_annotation": str(cfg["native_annotation"]),
                  "res_um": res_um, "shape_xyz": list(annotation.shape)},
        "regions": cfg["regions"],
        "block_shape": "column",
        "column_um": w_um,
        "axial_slack": slack,
        "depth_band": list(band),
        "min_voxels": min_voxels,
        "depth_bins": n_bins,
        "spacing_scale": cfg.get("spacing_scale", 1.0),
        "avoid_dorsal_um": cfg.get("avoid_dorsal_um"),
        "seed": cfg.get("seed", 0),
        "per_sample_labels": label_files,
        "per_sample_depth_bins": depth_files,
        "blocks": records,
    }, indent=2), encoding="utf-8")

    with open(out_dir / "blocks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["block", "x_mm", "y_mm", "z_mm", "n_voxels", "volume_mm3",
                    "depth_lo", "depth_hi", "top_region", "top_pct"])
        for r in records:
            comp = r["composition"]
            w.writerow([r["block"], *r["centre_mm_xyz"], r["n_voxels"],
                        r["volume_mm3"], *r["depth_p2_p98"],
                        comp[0]["name"] if comp else "",
                        comp[0]["pct"] if comp else ""])

    print(f"\n{'#':>3s}  {'centre (mm)':>22s}  {'深度':>11s}  主要结构")
    for r in records:
        comp = r["composition"]
        top = "  ".join(f"{c['name']} {c['pct']}%" for c in comp[:2])
        mm = ", ".join(f"{v:5.2f}" for v in r["centre_mm_xyz"])
        lo, hi = r["depth_p2_p98"]
        print(f"{r['block']:3d}  [{mm}]  {lo:.2f}–{hi:.2f}  {top}")
    nib.save(nib.Nifti1Image(block_volume.astype(np.uint16), np.eye(4)),
             str(out_dir / "block_labels_native.nii.gz"))
    print(f"\n写出 -> {out_dir / 'blocks.json'}  加逐样本的 {len(label_files)} 个标签体"
          f"和 {len(depth_files)} 个深度分箱体（{n_bins} 层）")
    if not no_figures:
        for fig_path in draw_column_sections(annotation, mask, block_volume,
                                             structures, out_dir, res_um):
            print(f"画图 -> {fig_path}")
    return records


# ── Figures ───────────────────────────────────────────────────────────────────

def draw_sections(annotation, region_mask, blocks, block_vox, out_dir, res_um,
                  axis=1, max_panels=12):
    """Atlas sections with the block footprints outlined on them.

    Sections are chosen at the block centres rather than spread evenly: the
    question a reader has is "where did block 7 go", and a panel that contains
    no block answers nothing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not blocks:
        return []
    planes = sorted({int(b[axis]) for b in blocks})[:max_panels]
    others = [a for a in range(3) if a != axis]
    ncol = min(4, len(planes))
    nrow = int(np.ceil(len(planes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.2 * nrow),
                             squeeze=False)
    half = block_vox // 2
    for ax in axes.ravel():
        ax.set_axis_off()
    for panel, plane in zip(axes.ravel(), planes):
        idx = [slice(None)] * 3
        idx[axis] = plane
        ann_slice = annotation[tuple(idx)]
        reg_slice = region_mask[tuple(idx)]
        panel.set_axis_on()
        panel.imshow((ann_slice > 0).T, cmap="gray", vmin=0, vmax=2,
                     interpolation="nearest")
        panel.imshow(np.ma.masked_where(~reg_slice, reg_slice).T, cmap="autumn",
                     alpha=0.35, interpolation="nearest")
        for i, b in enumerate(blocks):
            if abs(int(b[axis]) - plane) > half:
                continue
            x0 = b[others[0]] - half
            y0 = b[others[1]] - half
            panel.add_patch(Rectangle((x0, y0), block_vox, block_vox,
                                      fill=False, ec="cyan", lw=1.4))
            panel.text(x0, y0 - 3, str(i + 1), color="cyan", fontsize=8)
        panel.set_title(f"axis {axis} = {plane}  "
                        f"({plane * res_um / 1000:.1f} mm)", fontsize=9)
        panel.set_xticks([])
        panel.set_yticks([])
    fig.suptitle(f"{len(blocks)} blocks, {block_vox * res_um:.0f} um cubes "
                 f"(cyan), target region in orange", fontsize=11)
    fig.tight_layout()
    path = Path(out_dir) / f"blocks_axis{axis}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [path]


def run_roi(cfg, annotation, structures, mask, res_um, out_dir, dry_run,
            no_figures=False):
    """One region of interest: the middle of cortex, ends trimmed off.

    WHY THIS INSTEAD OF BLOCKS
    --------------------------
    Blocks buy spatial resolution and pay for it in arbitrariness -- where the
    packing seed put them -- and in a multiple-comparison family. At 300 um
    spacing 233 columns cover the sheet so densely that the exercise is close to
    using every cell anyway, so if the question is purely "are there more cells
    in one group", a single ROI is the cleaner instrument: no packing, no seed,
    complete layer coverage by construction, and counts large enough that
    Poisson noise is nothing next to animal-to-animal variance.

    What it trades away is every spatial question. One ROI cannot say where.

    The trim is measured on the REGION's extent in ATLAS space, not in each
    sample's own grid. "The middle of this brain's cortex" would be a different
    anatomical territory in every animal, which is exactly the confound that
    made per-region density unusable here; the middle of the ATLAS's cortex is
    the same territory in all of them, because that is what registration means.
    """
    trim = dict(cfg.get("roi_trim") or {})
    ml_axis, sym = ml_axis_of(annotation)
    dv_axis, dorsal_low = dorsal_axis(annotation, mask)
    ap_axis = [a for a in range(3) if a not in (ml_axis, dv_axis)][0]
    axis_of = {"ML": ml_axis, "DV": dv_axis, "AP": ap_axis}
    print(f"左右轴 = {ml_axis}（对称度 {sym:.0%}），背腹轴 = {dv_axis}"
          f"（{'低端' if dorsal_low else '高端'}为背侧），前后轴 = {ap_axis}")

    keep = mask.copy()
    report = []
    for name, frac in trim.items():
        frac = float(frac)
        if frac <= 0:
            continue
        if not 0 < frac < 0.5:
            raise ValueError(f"roi_trim.{name} 要在 0 和 0.5 之间，两端各删这么多。")
        axis = axis_of[name.upper()]
        present = np.where(mask.any(axis=tuple(a for a in range(3) if a != axis)))[0]
        lo_i, hi_i = int(present[0]), int(present[-1])
        span = hi_i - lo_i + 1
        cut = int(round(span * frac))
        idx = np.arange(annotation.shape[axis]).reshape(
            [-1 if a == axis else 1 for a in range(3)])
        keep &= (idx >= lo_i + cut) & (idx <= hi_i - cut)
        report.append(f"{name}: 皮层跨度 {span * res_um / 1000:.1f} mm，两端各删 "
                      f"{cut * res_um / 1000:.1f} mm，留 "
                      f"{(span - 2 * cut) * res_um / 1000:.1f} mm")
    for line in report:
        print("  " + line)
    vol = keep.sum() * (res_um / 1000) ** 3
    print(f"ROI = {vol:.1f} mm3，占目标结构的 "
          f"{keep.sum() / max(mask.sum(), 1) * 100:.0f}%")
    if not keep.any():
        print("裁完之后什么都不剩。")
        return
    if dry_run:
        return

    block_volume = keep.astype(np.uint16)
    out_dir.mkdir(parents=True, exist_ok=True)
    import nibabel as nib
    label_files, n_native = {}, int(block_volume.sum())
    for name, pipeline_cfg in cfg["samples"].items():
        params = variant_params(pipeline_cfg)
        pushed = sample_volume(block_volume, annotation, params, ml_axis)
        n = int((pushed > 0).sum())
        # The ROI is one piece and it is large, so "does this sample's atlas
        # preparation clip it" is a real question here in a way it was not for
        # a 300 um block: the slicing takes one hemisphere, so roughly half is
        # expected to go, and anything else is worth seeing.
        print(f"  {name:6s} orientation {params['orientation']}   "
              f"保留 {n} 体素 = 原生的 {n / n_native * 100:.0f}%"
              + ("  [镜像半脑]" if mirrors_ml(params, ml_axis) else ""))
        fname = f"block_labels_{name}.nii.gz"
        nib.save(nib.Nifti1Image(pushed.astype(np.uint16), np.eye(4)),
                 str(out_dir / fname))
        label_files[name] = fname

    counts = Counter(int(v) for v in annotation[keep])
    comp = []
    for sid, n in counts.most_common(20):
        if sid == 0:
            continue
        comp.append({"id": sid,
                     "name": structures.get(sid, {}).get("name", f"id {sid}"),
                     "pct": round(100.0 * n / int(keep.sum()), 1)})
        if len(comp) >= 8:
            break
    record = {"block": 1, "n_voxels": int(keep.sum()),
              "volume_mm3": round(vol, 4), "composition": comp}
    (out_dir / "blocks.json").write_text(json.dumps({
        "atlas": {"native_annotation": str(cfg["native_annotation"]),
                  "res_um": res_um, "shape_xyz": list(annotation.shape)},
        "regions": cfg["regions"],
        "block_shape": "roi",
        "roi_trim": trim,
        "axes": {"ML": ml_axis, "DV": dv_axis, "AP": ap_axis},
        "per_sample_labels": label_files,
        "blocks": [record],
    }, indent=2), encoding="utf-8")
    nib.save(nib.Nifti1Image(block_volume, np.eye(4)),
             str(out_dir / "block_labels_native.nii.gz"))
    print(f"\n主要结构: " + "  ".join(f"{c['name']} {c['pct']}%" for c in comp[:4]))
    print(f"写出 -> {out_dir / 'blocks.json'}")
    if not no_figures:
        for fig_path in draw_column_sections(annotation, mask, block_volume,
                                             structures, out_dir, res_um):
            print(f"画图 -> {fig_path}")
    return [record]


def draw_column_sections(annotation, region_mask, block_volume, structures,
                         out_dir, res_um, axis=2, n_panels=8):
    """Coronal sections with the columns drawn as they actually are.

    A cube can be outlined with a rectangle; a column cannot, because it is
    tilted and its ends are cut by the region mask. So the panels show the
    label volume itself, sliced. Cortex is tinted BY LAYER underneath, which is
    the point of the picture: what a reader wants to check is that the columns
    cross the layers rather than run along them.

    Coronal by default (the AP axis), because that is the plane the curvature
    and the layering are easiest to read in.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    # Figure text is English on purpose: the default matplotlib font carries no
    # CJK glyphs, so Chinese labels render as tofu boxes on any machine that
    # happens to lack a CJK face, and a figure that breaks depending on where it
    # is drawn is worse than one that is simply in English.
    order = ["layer 1", "layer 2/3", "layer 4", "layer 5", "layer 6a", "layer 6b"]
    layer = np.zeros(annotation.shape, dtype=np.uint8)
    for sid, info in structures.items():
        name = str(info.get("name", "")).lower()
        for k, key in enumerate(order, start=1):
            if name.endswith(" " + key):
                layer[annotation == int(sid)] = k
                break
    layer[~region_mask] = 0
    cmap = ListedColormap(["#e8e0d8", "#f6d743", "#7ec850", "#3aa7a0",
                           "#e2725b", "#1f6f5c"])
    norm = BoundaryNorm(np.arange(0.5, 7.5), cmap.N)

    present = np.unique(np.argwhere(block_volume > 0)[:, axis])
    if len(present) == 0:
        return []
    planes = [int(v) for v in np.linspace(present.min(), present.max(), n_panels)]
    others = [a for a in range(3) if a != axis]
    ncol = min(4, len(planes))
    nrow = int(np.ceil(len(planes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.4 * nrow),
                             squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for panel, plane in zip(axes.ravel(), planes):
        idx = [slice(None)] * 3
        idx[axis] = plane
        ann_s = annotation[tuple(idx)].T
        lay_s = layer[tuple(idx)].T
        blk_s = block_volume[tuple(idx)].T
        panel.set_axis_on()
        panel.imshow((ann_s > 0), cmap="gray_r", vmin=0, vmax=6,
                     interpolation="nearest")
        panel.imshow(np.ma.masked_where(lay_s == 0, lay_s), cmap=cmap, norm=norm,
                     alpha=0.75, interpolation="nearest")
        panel.imshow(np.ma.masked_where(blk_s == 0, np.ones_like(blk_s)),
                     cmap=ListedColormap(["#101820"]), alpha=0.85,
                     interpolation="nearest")
        panel.set_title(f"AP {plane * res_um / 1000:.1f} mm  "
                        f"({len(np.unique(blk_s)) - 1} columns)", fontsize=9)
        panel.set_xticks([])
        panel.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, fc=cmap(i)) for i in range(6)]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="#101820"))
    fig.legend(handles, [o.replace("layer ", "L") for o in order] + ["column"],
               loc="lower center", ncol=7, frameon=False, fontsize=10)
    n_blocks = int(block_volume.max())
    fig.suptitle(f"{n_blocks} columns, {res_um:.0f} um grid, along the cortical "
                 f"normal with both ends cut by the Isocortex boundary", fontsize=13)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    path = Path(out_dir) / "columns_coronal.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return [path]


def open_napari(annotation, region_mask, blocks, block_vox, res_um):
    import napari
    viewer = napari.Viewer(title="pick_blocks")
    viewer.add_image((annotation > 0).astype(np.uint8), name="atlas",
                     colormap="gray", scale=(res_um,) * 3, opacity=0.6)
    viewer.add_labels(region_mask.astype(np.uint8), name="target region",
                      scale=(res_um,) * 3, opacity=0.3)
    if blocks:
        cube = np.zeros(annotation.shape, dtype=np.uint16)
        half = block_vox // 2
        for i, b in enumerate(blocks, start=1):
            lo = np.maximum(b - half, 0)
            hi = np.minimum(lo + block_vox, annotation.shape)
            cube[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = i
        viewer.add_labels(cube, name=f"{len(blocks)} blocks",
                          scale=(res_um,) * 3, opacity=0.55)
    napari.run()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_config_arg(parser, TOOL)
    parser.add_argument("--napari", action="store_true", help="布点之后打开 napari")
    parser.add_argument("--dry-run", action="store_true",
                        help="只数合法位置和能放几个块，不写文件")
    parser.add_argument("--no-figures", action="store_true", help="不画图")
    args = parser.parse_args()

    cfg = load_config(TOOL, args.config,
                      required=("out_dir", "native_annotation", "ontology_json",
                                "regions", "samples"))
    from registration_ants.atlas_utils import (
        _read_atlas_array_xyz, load_ccf_ontology_json)

    shape_mode = str(cfg.get("block_shape", "cube"))
    if shape_mode not in ("cube", "column", "roi"):
        raise ValueError("block_shape 只能是 cube、column 或 roi。")
    res_um = float(cfg.get("atlas_res_um", 20))
    block_vox = 0
    if shape_mode == "cube":
        if not cfg.get("block_um"):
            raise ValueError("block_shape: cube 需要 block_um。")
        block_vox = int(round(float(cfg["block_um"]) / res_um))
        if block_vox < 1:
            raise ValueError("block_um 小于一个图谱体素。")

    annotation = _read_atlas_array_xyz(cfg["native_annotation"], preserve_labels=True)
    annotation = annotation.astype(np.int64)
    structures = load_ccf_ontology_json(cfg["ontology_json"])
    mask = np.isin(annotation, list(region_ids(structures, cfg["regions"])))
    print(f"native atlas {annotation.shape} @ {res_um:.0f} um   "
          f"target {', '.join(cfg['regions'])}  {mask.sum() * (res_um / 1000) ** 3:.1f} mm3")

    if shape_mode == "column":
        run_columns(cfg, annotation, structures, mask, res_um,
                    Path(cfg["out_dir"]), args.dry_run, args.no_figures)
        return
    if shape_mode == "roi":
        run_roi(cfg, annotation, structures, mask, res_um,
                Path(cfg["out_dir"]), args.dry_run, args.no_figures)
        return

    legal = legal_centres(mask, block_vox, float(cfg.get("min_purity", 1.0)))

    if cfg.get("min_surface_dist_um"):
        # Distance from ANY tissue surface. This is about registration
        # robustness, not damage: a block hugging the pial surface is where the
        # warp is least constrained and where a 100-250 um error most easily
        # puts the block half outside the brain.
        from scipy import ndimage
        depth = ndimage.distance_transform_edt(annotation > 0) * res_um
        legal &= depth >= float(cfg["min_surface_dist_um"])
        print(f"离组织表面 < {cfg['min_surface_dist_um']} um 的位置已排除: "
              f"剩 {int(legal.sum())}")

    if cfg.get("avoid_dorsal_um"):
        # The dorsal cap specifically, which is where these brains crack.
        # Measured from the dorsal-most tissue voxel along the derived DV axis,
        # so it means the same thing regardless of where the array starts.
        axis, dorsal_low = dorsal_axis(annotation, mask)
        present = np.where((annotation > 0).any(
            axis=tuple(a for a in range(3) if a != axis)))[0]
        cut = float(cfg["avoid_dorsal_um"]) / res_um
        idx = np.arange(annotation.shape[axis]).reshape(
            [-1 if a == axis else 1 for a in range(3)])
        if dorsal_low:
            legal &= idx >= present[0] + cut
        else:
            legal &= idx <= present[-1] - cut
        print(f"背侧 {cfg['avoid_dorsal_um']} um（轴 {axis}，"
              f"{'低端' if dorsal_low else '高端'}为背侧）已排除: "
              f"剩 {int(legal.sum())}")
    print(f"{cfg['block_um']} um 立方块的合法中心: {int(legal.sum())} 体素")

    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    blocks = pack_blocks(legal, block_vox, cfg.get("n_blocks"), rng,
                         float(cfg.get("spacing_scale", 1.0)))
    print(f"互不重叠地放下了 {len(blocks)} 个块")
    if args.dry_run or not blocks:
        return

    # Push the blocks through every sample's own atlas preparation and keep
    # only those that survive all of them intact.
    block_volume = np.zeros(annotation.shape, dtype=np.uint16)
    half = block_vox // 2
    for i, b in enumerate(blocks, start=1):
        lo = np.maximum(b - half, 0)
        hi = np.minimum(lo + block_vox, annotation.shape)
        block_volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = i

    ml_axis, sym = ml_axis_of(annotation)
    print(f"\n左右轴 = 轴 {ml_axis}（两半对称度 {sym:.0%}）")
    per_sample = {}
    for name, pipeline_cfg in cfg["samples"].items():
        params = variant_params(pipeline_cfg)
        per_sample[name] = sample_bounds(block_volume, annotation, params,
                                         len(blocks), block_vox, ml_axis)
        dropped = sum(1 for v in per_sample[name].values() if v is None)
        mirror = "  [镜像半脑，块已预先翻转]" if mirrors_ml(params, ml_axis) else ""
        print(f"  {name:6s} orientation {params['orientation']} "
              f"slicing {params['slicing']}   被裁掉 {dropped} 个块{mirror}")

    keep = [i for i in range(len(blocks))
            if all(per_sample[s][i] is not None for s in per_sample)]
    if len(keep) < len(blocks):
        print(f"\n{len(blocks) - len(keep)} 个块在至少一个样本的图谱准备里被裁掉，"
              "已剔除 —— 那些块在各样本之间不是同一片组织。")
    blocks = [blocks[i] for i in keep]

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for new_i, old_i in enumerate(keep):
        centre = blocks[new_i]
        comp = composition(annotation, structures, centre, block_vox)
        records.append({
            "block": new_i + 1,
            "centre_native_xyz": centre.tolist(),
            "centre_mm_xyz": [round(float(v) * res_um / 1000, 3) for v in centre],
            "size_um": float(cfg["block_um"]),
            "volume_mm3": round((float(cfg["block_um"]) / 1000) ** 3, 4),
            "composition": comp,
            "per_sample_bounds": {s: per_sample[s][old_i] for s in per_sample},
        })

    (out_dir / "blocks.json").write_text(json.dumps({
        "atlas": {"native_annotation": str(cfg["native_annotation"]),
                  "res_um": res_um, "shape_xyz": list(annotation.shape)},
        "regions": cfg["regions"],
        "block_um": cfg["block_um"],
        "min_purity": cfg.get("min_purity", 1.0),
        "spacing_scale": cfg.get("spacing_scale", 1.0),
        "avoid_dorsal_um": cfg.get("avoid_dorsal_um"),
        "min_surface_dist_um": cfg.get("min_surface_dist_um"),
        "seed": cfg.get("seed", 0),
        "blocks": records,
    }, indent=2), encoding="utf-8")

    with open(out_dir / "blocks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["block", "x", "y", "z", "x_mm", "y_mm", "z_mm",
                    "top_region", "top_pct", "second_region", "second_pct"])
        for r in records:
            c, m, comp = r["centre_native_xyz"], r["centre_mm_xyz"], r["composition"]
            w.writerow([r["block"], *c, *m,
                        comp[0]["name"] if comp else "", comp[0]["pct"] if comp else "",
                        comp[1]["name"] if len(comp) > 1 else "",
                        comp[1]["pct"] if len(comp) > 1 else ""])

    print(f"\n{'#':>3s}  {'centre (mm)':>22s}  主要结构")
    for r in records:
        comp = r["composition"]
        top = "  ".join(f"{c['name']} {c['pct']}%" for c in comp[:2])
        mm = ", ".join(f"{v:5.2f}" for v in r["centre_mm_xyz"])
        print(f"{r['block']:3d}  [{mm}]  {top}")

    # The layer breakdown is the interpretive headline, not a detail: a cube
    # required to sit entirely inside Isocortex cannot reach the upper layers at
    # all, because they are a thin outer shell.  Measured on DeMBA P5, a 600 um
    # cube's legal centres are 35% layer 5 and 58% layer 6a with ~3% each in
    # layers 1 and 2/3; a 400 um cube only shifts that to 55/38.  So this design
    # samples DEEP cortex, and that has to be stated rather than discovered by a
    # reader.  Widening to the upper layers would need a layer-following slab,
    # which a 100-250 um registration error cannot keep on the right layer.
    layers = Counter()
    for r in records:
        for c in r["composition"]:
            m = re.search(r"[Ll]ayer\s*(1|2/3|4|5|6a|6b)", c["name"])
            layers[m.group(1) if m else "other"] += 1
    if layers:
        total = sum(layers.values())
        breakdown = "  ".join(
            f"layer {k} {100 * v / total:.0f}%" for k, v in sorted(layers.items()))
        print(f"\n块落在哪些层（按各块前几位结构计）: {breakdown}")
        print("纯立方块放不进上层皮层，上层是一层薄壳 —— 这是几何限制，"
              "不是参数没调好，要写进 methods。")

    pngs = []
    for axis in (1, 2):
        pngs += draw_sections(annotation, mask, blocks, block_vox, out_dir,
                              res_um, axis=axis)
    print(f"\nblocks.json / blocks.csv / {len(pngs)} 张 PNG -> {out_dir}")
    print("blocks.json 就是「块放在哪」的权威记录，和结果一起归档。")

    if args.napari:
        open_napari(annotation, mask, blocks, block_vox, res_um)


if __name__ == "__main__":
    main()
