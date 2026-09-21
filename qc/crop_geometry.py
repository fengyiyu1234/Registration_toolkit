"""Shared geometry for the crop-based detection QC: atlas region -> global
stitched pixel box -> assembled multi-channel volume.

Imported by qc/cut_crops.py and qc/score_crops.py, never run on its own.
qc/annotate_crop.py deliberately does NOT import this -- see qc/README.md for
why the annotation step is kept unable to reach anything that identifies a crop.

Two coordinate frames, and the whole file exists to keep them straight:

GLOBAL PIXEL
    The frame TeraStitcher's merging xml defines and the one brain_detector
    writes cell centroids in: x,y in stitched pixels at `cells.voxel_size_um`
    (0.65 um here), z as a 1-indexed slice number at 8 um.  This is also
    columns 0-2 of a run's cell_registration.csv -- registration_ants'
    cell_points.py takes those numbers, multiplies by the voxel size and
    treats the product as microns, so global pixel * voxel size IS the shared
    physical frame every volume in the pipeline lives in.

LABEL VOXEL
    The grid of <run>/<name>_labels_in_sample.nii.gz: 20 um isotropic, origin
    0, on the UNCROPPED fine grid (pipeline.py keeps it there on purpose).
    Because origin is 0 and spacing is in microns, going between the two
    frames is a scalar multiply and nothing else -- no ANTs call, no inverse
    warp.  Warping the annotation onto the 0.65 um grid instead would be
    ~9.6e10 voxels, so this is not merely the simpler route, it is the only
    feasible one.

        label_voxel = global_px * cell_voxel_um / 20.0

The z convention is the one place this can silently go wrong, so nothing here
guesses: `TileGrid` follows visualize.py's `_get_tile_offset` exactly
(tile_z0 = max(ABS_D) - ABS_D, local_z_0idx = global_z - 1 + tile_z0) and
cut_crops.py runs `signal_check()` on every crop it writes, which catches an
off-by-one or a flipped sign in x, y or z as a collapsed intensity ratio.
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


# ── Frame conversion ──────────────────────────────────────────────────────────

def global_px_to_label_voxel(px_xyz, cell_voxel_um, label_spacing_um):
    """Global stitched pixel (x, y, z) -> float index into the labels volume."""
    px = np.asarray(px_xyz, dtype=float)
    return px * np.asarray(cell_voxel_um, float) / np.asarray(label_spacing_um, float)


def label_voxel_to_global_px(vox_xyz, cell_voxel_um, label_spacing_um):
    """Labels-volume index (x, y, z) -> float global stitched pixel."""
    vox = np.asarray(vox_xyz, dtype=float)
    return vox * np.asarray(label_spacing_um, float) / np.asarray(cell_voxel_um, float)


# ── Region selection ──────────────────────────────────────────────────────────

def descendant_ids(structures, root_name):
    """Every atlas id at or under the structure called `root_name`.

    `structures` is registration_ants' flat {id: {...}} dict.  Descent is read
    off `structure_id_path` when it is there, which is what
    atlas_utils.load_ccf_ontology_json actually writes: the whole ancestor
    chain per node, so "is a descendant" is one membership test and no tree
    has to be rebuilt.  `parent_structure_id` is the fallback for a dict that
    carries parents instead.

    Matching is on the name because that is what a person writes in a config;
    an unknown name raises rather than silently selecting nothing, which would
    hand back an empty crop list looking like "no cortex in this sample".
    """
    by_name = {info.get("name"): sid for sid, info in structures.items()}
    if root_name not in by_name:
        raise ValueError(
            f"没有叫 {root_name!r} 的图谱结构。检查拼写，或换成 ontology JSON 里的准确名字。")
    root_id = int(by_name[root_name])

    if any("structure_id_path" in info for info in structures.values()):
        return {int(sid) for sid, info in structures.items()
                if root_id in [int(v) for v in info.get("structure_id_path", [])]}

    children = {}
    for sid, info in structures.items():
        parent = info.get("parent_structure_id")
        if parent is not None:
            children.setdefault(int(parent), []).append(int(sid))
    out, stack = set(), [root_id]
    while stack:
        node = stack.pop()
        if node in out:
            continue
        out.add(node)
        stack.extend(children.get(node, []))
    return out


def region_mask(labels, ids):
    return np.isin(labels, np.fromiter(ids, dtype=labels.dtype, count=len(ids)))


def erode_mask(mask, iters_xyz):
    """Binary erosion by a per-axis amount, given as (nx, ny, nz) voxels.

    Per-axis rather than isotropic because the crop is strongly anisotropic:
    200 x 200 x 50 pixels at 0.65/0.65/8 um is 130 um wide and 400 um deep, so
    the margin it needs in z is three times what it needs in x.  Eroding every
    axis by the largest of the three would throw away most of a thin structure
    for no reason.

    Crop sites are drawn from the eroded mask so that the WHOLE crop sits
    inside the target region.  Without this a site one label voxel from the
    boundary yields a crop that is 20 um inside cortex and 110 um outside it,
    and the annotator scores tissue the result was never about.
    """
    out = mask
    for axis, iters in enumerate(np.asarray(iters_xyz, int)):
        for _ in range(int(iters)):
            shrunk = out & np.roll(out, 1, axis=axis) & np.roll(out, -1, axis=axis)
            # np.roll wraps, so the outermost plane on this axis is not trustworthy
            idx = [slice(None)] * 3
            idx[axis] = 0
            shrunk[tuple(idx)] = False
            idx[axis] = -1
            shrunk[tuple(idx)] = False
            out = shrunk
    return out


def erosion_iters_for(crop_px, cell_voxel_um, label_spacing_um):
    """Per-axis label voxels to erode so a crop of `crop_px` fits inside.

    Half the crop's physical extent on each axis, in label voxels, rounded up,
    plus one for the 20 um quantisation of the label itself.
    """
    half_um = np.asarray(crop_px, float) * np.asarray(cell_voxel_um, float) / 2.0
    return np.ceil(half_um / np.asarray(label_spacing_um, float)).astype(int) + 1


def z_band_of(z_px, n_z, n_bands):
    """Which depth band a global z falls in. Sox9 false negatives are expected
    to grow with imaging depth, so crops are stratified on this and the band is
    carried into the manifest to be reported against."""
    if n_bands <= 1:
        return 0
    return int(min(n_bands - 1, max(0, z_px * n_bands // max(1, n_z))))


def pick_crop_sites(labels, ids, crop_px, cell_voxel_um, label_spacing_um,
                    n_crops, n_z_bands, rng, min_separation_px=None):
    """-> list of dicts {origin_px, center_px, region_id, z_band}.

    Sites are spread over depth bands first and drawn uniformly from the
    eroded region mask within each band, then thinned so no two crops are
    closer than `min_separation_px`.  Spreading matters more than randomness
    here: with a dozen crops, three that happen to land in the same cortical
    column carry roughly one crop's worth of information about recall, because
    everything that drives a miss -- depth, staining, local density -- is
    shared inside a column.
    """
    crop_px = np.asarray(crop_px, int)
    mask = region_mask(labels, ids)
    if not mask.any():
        raise ValueError("目标脑区在这个样本的 labels_in_sample 里一个体素都没有。")
    mask = erode_mask(mask, erosion_iters_for(crop_px, cell_voxel_um, label_spacing_um))
    if not mask.any():
        raise ValueError(
            "腐蚀之后目标脑区没有体素剩下 —— crop 比这个区还大。"
            "把 crop_px 改小，或者换一个更大的区。")

    if min_separation_px is None:
        min_separation_px = crop_px.astype(float) * 2.0
    min_separation_px = np.asarray(min_separation_px, float)

    cand = np.argwhere(mask)  # (N, 3) in labels-volume x, y, z
    n_z_slices = int(round(labels.shape[2] * label_spacing_um[2] / cell_voxel_um[2]))
    cand_center_px = label_voxel_to_global_px(
        cand + 0.5, cell_voxel_um, label_spacing_um)
    bands = np.array([z_band_of(c[2], n_z_slices, n_z_bands) for c in cand_center_px])

    sites, per_band = [], max(1, int(np.ceil(n_crops / max(1, n_z_bands))))
    for band in range(n_z_bands):
        pool = np.where(bands == band)[0]
        if len(pool) == 0:
            continue
        rng.shuffle(pool)
        taken = 0
        for i in pool:
            if taken >= per_band or len(sites) >= n_crops:
                break
            center = cand_center_px[i]
            if any(np.all(np.abs(center - s["center_px"]) < min_separation_px)
                   for s in sites):
                continue
            origin = np.round(center - crop_px / 2.0).astype(int)
            sites.append({
                "origin_px": origin,
                "center_px": center,
                "region_id": int(labels[tuple(cand[i])]),
                "z_band": int(band),
            })
            taken += 1
    return sites


# ── Tile grid ─────────────────────────────────────────────────────────────────

class TileGrid:
    """One channel's stitched tile mosaic, read from TeraStitcher's merging xml.

    Reproduces visualize.py `_get_tile_offset` exactly, because the global
    coordinates in cell_registration.csv were produced under that convention
    and a crop cut under any other one would be compared against predictions
    that describe different tissue:

        tile_x0 = ABS_H - min(ABS_H)
        tile_y0 = ABS_V - min(ABS_V)
        tile_z0 = max(ABS_D) - ABS_D
        local_z_0idx = global_z - 1 + tile_z0
    """

    def __init__(self, channel_dir, xml_name=None, offset_px=(0, 0, 0)):
        self.channel_dir = Path(channel_dir)
        self.offset_px = np.asarray(offset_px, int)
        self.xml_path = self._find_xml(xml_name)
        self.tiles = self._parse()
        self.tile_shape = None  # (H, W), filled lazily from the first tif read

    def _find_xml(self, xml_name):
        names = [xml_name] if xml_name else ["xml_merging.xml", "xml_import.xml"]
        for name in names:
            path = self.channel_dir / name
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"{self.channel_dir} 下没有 {' 或 '.join(str(n) for n in names)}。"
            "全分辨率 tile 必须和它的拼接 xml 放在一起 —— 没有 xml 就没有全局坐标，"
            "这一步不能靠猜。")

    def _parse(self):
        root = ET.parse(self.xml_path).getroot()
        stacks = list(root.find("STACKS"))
        if not stacks:
            raise ValueError(f"{self.xml_path} 里没有 STACKS 条目。")
        abs_h = [int(s.get("ABS_H", 0)) for s in stacks]
        abs_v = [int(s.get("ABS_V", 0)) for s in stacks]
        abs_d = [int(s.get("ABS_D", 0)) for s in stacks]
        x_min, y_min, z_start = min(abs_h), min(abs_v), max(abs_d)

        tiles = []
        for stack in stacks:
            dir_name = (stack.get("DIR_NAME") or "").replace("\\", "/")
            leaf = os.path.basename(dir_name.rstrip("/"))
            path = self.channel_dir / dir_name
            if not path.is_dir():
                alt = self.channel_dir / leaf
                path = alt if alt.is_dir() else path
            tiles.append({
                "name": leaf,
                "path": path,
                "x0": int(stack.get("ABS_H", 0)) - x_min,
                "y0": int(stack.get("ABS_V", 0)) - y_min,
                "z0": z_start - int(stack.get("ABS_D", 0)),
            })
        return tiles

    def _slice_files(self, tile):
        files = sorted(f for f in os.listdir(tile["path"])
                       if f.lower().endswith((".tif", ".tiff")))
        if not files:
            raise FileNotFoundError(f"tile 目录里没有 tif: {tile['path']}")
        return files

    def _read_slice(self, tile, files, local_z):
        # tifffile rather than brain_detector's cv2: this repo's env does not
        # carry opencv, and a light-sheet slice is a plain 16-bit tif that
        # tifffile reads without the IMREAD_ANYDEPTH dance.
        import tifffile
        if local_z < 0 or local_z >= len(files):
            return None
        try:
            img = tifffile.imread(str(Path(tile["path"]) / files[local_z]))
        except (OSError, ValueError):
            return None
        return img[:, :, 0] if img.ndim == 3 else img

    def probe_tile_shape(self):
        """(H, W) of a tile, read off a real image rather than assumed to be 2048."""
        if self.tile_shape is None:
            for tile in self.tiles:
                try:
                    files = self._slice_files(tile)
                    img = self._read_slice(tile, files, 0)
                except (FileNotFoundError, NotADirectoryError, OSError):
                    continue
                if img is not None:
                    self.tile_shape = img.shape[:2]
                    break
        if self.tile_shape is None:
            raise RuntimeError(f"{self.channel_dir} 下没有任何 tile 能读出图像。")
        return self.tile_shape

    def locate(self, global_px):
        """All source slices covering a global point, including overlapping tiles.

        Local x/y remain fractional; the nearest acquired z plane is reported
        with a zero-based index, using exactly the convention used by cut().
        """
        point = np.asarray(global_px, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("global_px 必须是三个有限数值 [x, y, z]")
        x, y, z = point + self.offset_px
        height, width = self.probe_tile_shape()
        hits = []
        for tile in self.tiles:
            lx, ly = x - tile["x0"], y - tile["y0"]
            if not (0 <= lx < width and 0 <= ly < height):
                continue
            files = self._slice_files(tile)
            iz = int(np.floor(z - 1 + tile["z0"] + 0.5))
            if 0 <= iz < len(files):
                hits.append({"tile": tile["name"],
                             "path": str(tile["path"] / files[iz]),
                             "local_xyz": [float(lx), float(ly), iz]})
        return hits

    def cut(self, origin_px, size_px, dtype=np.uint16):
        """Assemble the global box [origin, origin+size) into one (Z, Y, X) array.

        `offset_px` is added to the requested box before reading, so a channel
        with a known residual alignment against the soma channels can be pulled
        into line without re-cutting the others.

        Tiles overlap.  The tile whose centre is nearest the crop centre is
        pasted LAST and therefore wins, so a crop that straddles a seam is
        shown mostly through one tile rather than as a patchwork -- an
        annotator judging a nucleus at a seam should be looking at whichever
        tile actually imaged it best, not at whichever the loop reached last.
        """
        origin = np.asarray(origin_px, int) + self.offset_px
        size = np.asarray(size_px, int)
        H, W = self.probe_tile_shape()
        out = np.zeros((size[2], size[1], size[0]), dtype=dtype)

        crop_center = origin + size / 2.0
        order = sorted(
            self.tiles,
            key=lambda t: -float(np.hypot(t["x0"] + W / 2 - crop_center[0],
                                          t["y0"] + H / 2 - crop_center[1])))

        for tile in order:
            gx0, gx1 = max(origin[0], tile["x0"]), min(origin[0] + size[0], tile["x0"] + W)
            gy0, gy1 = max(origin[1], tile["y0"]), min(origin[1] + size[1], tile["y0"] + H)
            if gx0 >= gx1 or gy0 >= gy1:
                continue
            try:
                files = self._slice_files(tile)
            except (FileNotFoundError, NotADirectoryError, OSError):
                continue
            for gz in range(origin[2], origin[2] + size[2]):
                img = self._read_slice(tile, files, gz - 1 + tile["z0"])
                if img is None:
                    continue
                out[gz - origin[2], gy0 - origin[1]:gy1 - origin[1],
                    gx0 - origin[0]:gx1 - origin[0]] = \
                    img[gy0 - tile["y0"]:gy1 - tile["y0"],
                        gx0 - tile["x0"]:gx1 - tile["x0"]]
        return out


# ── Self-check ────────────────────────────────────────────────────────────────

def signal_check(volume, points_local_xyz, rng, radius_px=(4, 4, 1), n_random=400):
    """-> (ratio, n_used). Mean intensity at predicted cell centres divided by
    mean intensity at random positions in the same crop.

    This is the guard against the whole chain being silently misaligned.  If
    the global frame, the z base, or a channel's tile offset is wrong, the
    crop holds real tissue and looks perfectly plausible, the predictions land
    on nothing in particular, and every downstream recall number is garbage
    without anything having raised an error.  A correct cut puts predicted
    soma centres on bright pixels: expect a ratio well above 1.  Around 1
    means the box being read is not the box the predictions describe.
    """
    if len(points_local_xyz) == 0:
        return float("nan"), 0
    Z, Y, X = volume.shape
    rx, ry, rz = int(radius_px[0]), int(radius_px[1]), int(radius_px[2])

    def _mean_at(pts):
        vals = []
        for x, y, z in pts:
            x, y, z = int(round(x)), int(round(y)), int(round(z))
            if not (0 <= x < X and 0 <= y < Y and 0 <= z < Z):
                continue
            box = volume[max(0, z - rz):z + rz + 1,
                         max(0, y - ry):y + ry + 1,
                         max(0, x - rx):x + rx + 1]
            if box.size:
                vals.append(float(box.mean()))
        return (float(np.mean(vals)) if vals else float("nan")), len(vals)

    hit, n_used = _mean_at(points_local_xyz)
    background, _ = _mean_at(np.column_stack([
        rng.integers(0, X, n_random),
        rng.integers(0, Y, n_random),
        rng.integers(0, Z, n_random)]))
    if not np.isfinite(hit) or not np.isfinite(background) or background <= 0:
        return float("nan"), n_used
    return hit / background, n_used
