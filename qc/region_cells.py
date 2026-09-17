"""Post-registration QC, GUI-free half: pick registered cells by atlas region,
cut the raw data around them, and describe where each one sits relative to
the region it was assigned to.

Imported by qc/view_region_cells.py, never run on its own.

WHAT THIS CHECKS THAT THE OTHER TOOLS DO NOT
--------------------------------------------
single_sample.py puts the cells on the 20 um registration grid: good for
"is the atlas roughly on the brain", useless for "is THIS cell in the layer
the table says it is in", because a 20 um voxel holds a dozen nuclei.
qc/cut_crops.py goes back to the full-resolution tiles but is blind on
purpose -- no regions, no predictions.  This is the non-blind counterpart:
select cells by region (name, acronym or id, descendants included), go back
to the raw pixels they were detected on, and lay the region boundary over
them.

FRAMES
------
Everything is placed in the pipeline's one physical frame (microns, origin
0), the same one registration_ants.cell_points uses:

    cell   phys = (x, y, z) from cell_registration.csv  *  cells.voxel_size_um
    tiles  phys = global stitched px (1-indexed z)       *  cells.voxel_size_um
    tiff   phys = index into sample.raw_tiff             *  sample.voxel_size_um
    labels phys = index into *_labels_in_sample.nii.gz   *  20 um

so a box is requested in microns and every source converts it to its own
grid.  The tile convention is qc/crop_geometry.TileGrid's, which copies
brain_detector's visualize.py -- the frame the global cell coordinates were
written in.

TWO REGION ANSWERS PER CELL, ON PURPOSE
---------------------------------------
Column 9 of the cell table is the atlas id found by pushing the cell INTO
atlas space and reading the annotation there.  The boundary drawn on screen
is the annotation pulled BACK into sample space (labels_in_sample).  They are
two resamplings of one transform and mostly agree; where they do not, the
cell is within a voxel or so of a boundary.  Both are reported, and cells are
selected on column 9 because that is what the statistics used.

Repositioned runs (sample.reposition_plan): columns 0-2 are the cell's
original position but labels_in_sample describes the closed brain, so on a
fragment the drawn boundary is in the wrong place.  Selection by column 9 is
still right.  The session warns; it does not try to undo the reposition.
"""
import difflib
import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from qc import crop_geometry as geom

# registration_ants.cell_points column layout, header-less.
CELL_COLUMNS = ["x", "y", "z", "xr", "yr", "zr", "xt", "yt", "zt",
                "region_id", "region_name", "slice_name", "tile_name", "score"]

MARKER_COLORS = {            # RGBA, keyed by the fluorescent-protein part only
    "GFP": (0.25, 1.0, 0.25, 1.0),
    "RFP": (1.0, 0.3, 0.3, 1.0),
    "GFP_RFP": (1.0, 0.9, 0.1, 1.0),
}
OTHER_COLOR = (0.6, 0.6, 0.6, 0.6)
CHANNEL_COLORMAPS = {"RFP": "red", "GFP": "green", "Sox9": "cyan"}


# ── Run directory ─────────────────────────────────────────────────────────────

def load_run_config(run_dir):
    """The pipeline config snapshot run_pipeline.sh copies into the output dir.

    -> (dict, path) or (None, None).  Same lookup single_sample.py does.
    """
    import yaml
    for path in sorted(list(Path(run_dir).glob("*.yaml")) + list(Path(run_dir).glob("*.yml"))):
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(cfg, dict) and isinstance(cfg.get("sample"), dict):
            return cfg, path
    return None, None


def load_cells(run_dir):
    """Every registered cell in the run, one row each.

    `class_name` is the folder name (brain_detector's composite label) and
    `row` is the line in that folder's csv, so (class_name, row) names a cell
    stably for as long as the run is not re-run -- verdicts are keyed on it.
    """
    frames = []
    for csv_path in sorted(Path(run_dir).glob("cell_registration/*/cell_registration.csv")):
        try:
            df = pd.read_csv(csv_path, header=None)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        if df.shape[1] < 11:
            raise ValueError(
                f"{csv_path} 只有 {df.shape[1]} 列。这不是 registration_ants 写的细胞表"
                "（ClearMap 的表第 9 列是 graph_order，不能直接按 atlas id 选区）。")
        df = df.iloc[:, :len(CELL_COLUMNS)]
        df.columns = CELL_COLUMNS[:df.shape[1]]
        df["class_name"] = csv_path.parent.name
        df["row"] = np.arange(len(df))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"{run_dir}/cell_registration/ 下没有读到任何细胞。")
    out = pd.concat(frames, ignore_index=True)
    for col in ("x", "y", "z"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["region_id"] = pd.to_numeric(out["region_id"], errors="coerce").fillna(0).astype(np.int64)
    out = out.dropna(subset=["x", "y", "z"]).reset_index(drop=True)
    out["cell_id"] = out["class_name"] + ":" + out["row"].astype(str)
    return out


def marker_key(class_name):
    """glia_GFP_RFP_Sox9 -> ('GFP_RFP', True).  Cells are compared by marker
    combination only; the neuron/glia prefix is not used for anything."""
    parts = str(class_name).split("_")
    fps = [p for p in ("GFP", "RFP") if p in parts]
    return "_".join(fps), "Sox9" in parts


# ── Regions ───────────────────────────────────────────────────────────────────

def find_structure(structures, spec):
    """One config entry -> atlas id.  Accepts an id, an exact name, or an
    acronym (exact first, then case-insensitive).  Unknown raises with the
    closest names, because silently selecting nothing looks exactly like
    "this sample has no cells there"."""
    if isinstance(spec, (int, np.integer)) or (isinstance(spec, str) and spec.strip().isdigit()):
        sid = int(spec)
        if sid not in structures:
            raise ValueError(f"本体里没有 id {sid}。")
        return sid
    spec = str(spec).strip()
    for key in ("name", "acronym"):
        for sid, info in structures.items():
            if info.get(key) == spec:
                return int(sid)
    low = spec.lower()
    hits = [sid for sid, info in structures.items() if str(info.get("acronym", "")).lower() == low]
    if len(hits) == 1:
        return int(hits[0])
    pool = [str(i.get("name")) for i in structures.values()] + \
           [str(i.get("acronym")) for i in structures.values()]
    close = difflib.get_close_matches(spec, pool, n=5, cutoff=0.6)
    raise ValueError(f"没有叫 {spec!r} 的图谱结构（名字或缩写）。"
                     + (f" 相近的：{', '.join(close)}" if close else ""))


def subtree_ids(structures, root_id):
    return {int(sid) for sid, info in structures.items()
            if root_id in [int(v) for v in info.get("structure_id_path", [])]}


def resolve_region_ids(structures, specs):
    """-> (ids, [(root_id, name, acronym), ...]).  Descendants included."""
    if isinstance(specs, (str, int)):
        specs = [specs]
    ids, roots = set(), []
    for spec in specs:
        sid = find_structure(structures, spec)
        ids |= subtree_ids(structures, sid)
        roots.append((sid, structures[sid].get("name"), structures[sid].get("acronym")))
    return ids, roots


def select_cells(cells, ids, class_patterns=None):
    """Cells whose column-9 region is in `ids` and whose class folder matches
    any of the fnmatch patterns (all classes if none)."""
    mask = cells["region_id"].isin(list(ids))
    if class_patterns:
        pats = [class_patterns] if isinstance(class_patterns, str) else list(class_patterns)
        names = cells["class_name"].unique()
        keep = [n for n in names if any(fnmatch.fnmatch(n, p) for p in pats)]
        if not keep:
            raise ValueError(f"classes {pats} 一个类别都没匹配上。现有类别：{sorted(names)}")
        mask &= cells["class_name"].isin(keep)
    return cells.loc[mask]


# ── Label volume ──────────────────────────────────────────────────────────────

class LabelVolume:
    """<run>/*_labels_in_sample.nii.gz in the pipeline's physical frame.

    nibabel hands back an RAS affine for a file ANTs wrote in LPS, so the
    origin is taken off the affine with the first two signs flipped back.  For
    every run so far it is 0 anyway -- pipeline.py keeps the label volume on
    the uncropped fine grid on purpose.
    """

    def __init__(self, path):
        import nibabel as nib
        img = nib.load(str(path))
        self.path = Path(path)
        self.data = np.asarray(img.dataobj)
        self.spacing = np.array(img.header.get_zooms()[:3], float)
        t = img.affine[:3, 3]
        self.origin = np.array([-t[0], -t[1], t[2]], float)
        self.shape = np.array(self.data.shape[:3])

    def index_of(self, phys):
        return np.rint((np.asarray(phys, float) - self.origin) / self.spacing).astype(np.int64)

    def lookup(self, phys):
        """Label under each physical point (N, 3); 0 outside the volume."""
        idx = np.atleast_2d(self.index_of(phys))
        ok = np.all((idx >= 0) & (idx < self.shape), axis=1)
        out = np.zeros(len(idx), dtype=np.int64)
        out[ok] = self.data[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        return out

    def box(self, lo_phys, hi_phys):
        """Sub-volume covering [lo, hi] -> (array xyz, origin_phys xyz).

        Zero-padded where the box leaves the volume, so the returned grid is
        always exactly the requested one.
        """
        i0 = np.floor((np.asarray(lo_phys) - self.origin) / self.spacing).astype(int)
        i1 = np.ceil((np.asarray(hi_phys) - self.origin) / self.spacing).astype(int) + 1
        out = np.zeros(tuple(i1 - i0), dtype=self.data.dtype)
        s0, s1 = np.maximum(i0, 0), np.minimum(i1, self.shape)
        if np.all(s1 > s0):
            out[tuple(slice(a - b, c - b) for a, b, c in zip(s0, i0, s1))] = \
                self.data[tuple(slice(a, c) for a, c in zip(s0, s1))]
        return out, self.origin + i0 * self.spacing

    def resample_to(self, origin_phys, voxel_um, shape_xyz):
        """Nearest-neighbour labels on another regular grid (for the PNG
        renderer, which draws on the image grid)."""
        axes = []
        for a in range(3):
            centres = origin_phys[a] + np.arange(shape_xyz[a]) * voxel_um[a]
            axes.append(np.rint((centres - self.origin[a]) / self.spacing[a]).astype(np.int64))
        valid = [(ix >= 0) & (ix < self.shape[a]) for a, ix in enumerate(axes)]
        clipped = [np.clip(ix, 0, self.shape[a] - 1) for a, ix in enumerate(axes)]
        out = self.data[np.ix_(*clipped)].astype(np.int64)
        out[~valid[0], :, :] = 0
        out[:, ~valid[1], :] = 0
        out[:, :, ~valid[2]] = 0
        return out


def _find_labels(run_dir):
    hits = sorted(Path(run_dir).glob("*_labels_in_sample.nii.gz"))
    if not hits:
        raise FileNotFoundError(f"{run_dir} 里没有 *_labels_in_sample.nii.gz。")
    return hits[0]


def signed_depth_um(labels, ids, phys, pad_vox=16):
    """Distance from each point to the boundary of the region set, in microns:
    positive inside (per labels_in_sample), negative outside.

    Computed on the 20 um label grid inside the region's bounding box plus
    `pad_vox`; a point farther out than that gets -inf ("far outside").  The
    distance is voxel centre to nearest non-region voxel centre, so values
    carry roughly half a voxel (10 um) of quantisation.
    """
    from scipy import ndimage as ndi
    phys = np.atleast_2d(np.asarray(phys, float))
    mask = np.isin(labels.data, np.fromiter(ids, dtype=np.int64, count=len(ids)))
    out = np.full(len(phys), -np.inf)
    if not mask.any():
        return out
    nz = np.argwhere(mask)
    lo = np.maximum(nz.min(0) - pad_vox, 0)
    hi = np.minimum(nz.max(0) + pad_vox + 1, labels.shape)
    sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    # pad with False so a region touching the volume edge still has a boundary
    sub = np.pad(sub, 1)
    inside = ndi.distance_transform_edt(sub, sampling=labels.spacing)
    outside = ndi.distance_transform_edt(~sub, sampling=labels.spacing)
    idx = labels.index_of(phys) - lo + 1
    ok = np.all((idx >= 0) & (idx < np.array(sub.shape)), axis=1)
    i = idx[ok]
    is_in = sub[i[:, 0], i[:, 1], i[:, 2]]
    out[ok] = np.where(is_in, inside[i[:, 0], i[:, 1], i[:, 2]],
                       -outside[i[:, 0], i[:, 1], i[:, 2]])
    return out


# ── Image sources ─────────────────────────────────────────────────────────────

@dataclass
class Box:
    """A cut: channel -> (Z, Y, X) array on a regular grid in the physical frame."""
    arrays: dict
    origin_um: np.ndarray          # xyz, physical position of voxel (0, 0, 0)
    voxel_um: np.ndarray           # xyz
    source: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def shape_xyz(self):
        z, y, x = next(iter(self.arrays.values())).shape
        return np.array([x, y, z])

    def local(self, phys):
        """Physical points (N, 3) -> fractional voxel index (x, y, z) in this box."""
        return (np.atleast_2d(phys) - self.origin_um) / self.voxel_um


def _index_range(lo_phys, hi_phys, voxel_um):
    i0 = np.floor(np.asarray(lo_phys) / voxel_um).astype(int)
    i1 = np.ceil(np.asarray(hi_phys) / voxel_um).astype(int) + 1
    return i0, i1 - i0


class TileSource:
    """Full-resolution tiles, one TileGrid per channel (the machine that holds
    them).  Global z is 1-indexed, exactly as the cell table stores it, so a
    box whose z origin is g reads global slices g, g+1, ... and voxel k sits
    at physical (g + k) * voxel_z."""
    kind = "tiles"

    def __init__(self, channels, cell_voxel_um):
        self.voxel_um = np.asarray(cell_voxel_um, float)
        self.grids = {}
        for name, ch in channels.items():
            ch = {"dir": ch} if isinstance(ch, str) else dict(ch)
            self.grids[name] = geom.TileGrid(ch["dir"], ch.get("xml"),
                                             ch.get("offset_px", (0, 0, 0)))

    def cut(self, lo_phys, hi_phys):
        origin, size = _index_range(lo_phys, hi_phys, self.voxel_um)
        arrays = {name: g.cut(origin, size) for name, g in self.grids.items()}
        return Box(arrays, origin * self.voxel_um, self.voxel_um, self.kind)


class VolumeSource:
    """One or more whole-brain tiffs on the registration grid (sample.raw_tiff
    and any other channel written on the same grid).  Works on a machine
    without the tiles; 2.6 x 2.6 x 32 um is enough to see tissue edges and
    layers, not single nuclei in z."""
    kind = "volume"

    def __init__(self, images, voxel_um):
        import tifffile
        self.voxel_um = np.asarray(voxel_um, float)
        self.vols = {}
        shapes = set()
        for name, path in images.items():
            try:
                vol = tifffile.memmap(str(path), mode="r")
            except (ValueError, OSError):
                # compressed or otherwise not memory-mappable: fall back to a
                # lazy zarr view, still without reading the whole stack
                import zarr
                vol = zarr.open(tifffile.imread(str(path), aszarr=True), mode="r")
            if vol.ndim != 3:
                raise ValueError(f"{path} 读出来是 {vol.ndim} 维，这里要 (Z, Y, X) 单通道。")
            self.vols[name] = vol
            shapes.add(tuple(vol.shape))
        if len(shapes) > 1:
            raise ValueError(f"images 里的几张图尺寸不一样 {shapes}，不在同一个网格上，不能共用体素大小。")
        self.shape_zyx = np.array(next(iter(shapes)))

    def cut(self, lo_phys, hi_phys):
        origin, size = _index_range(lo_phys, hi_phys, self.voxel_um)
        shape_xyz = self.shape_zyx[::-1]
        s0, s1 = np.maximum(origin, 0), np.minimum(origin + size, shape_xyz)
        arrays = {}
        for name, vol in self.vols.items():
            out = np.zeros(tuple(size[::-1]), dtype=vol.dtype)
            if np.all(s1 > s0):
                out[s0[2] - origin[2]:s1[2] - origin[2],
                    s0[1] - origin[1]:s1[1] - origin[1],
                    s0[0] - origin[0]:s1[0] - origin[0]] = \
                    np.asarray(vol[s0[2]:s1[2], s0[1]:s1[1], s0[0]:s1[0]])
            arrays[name] = out
        return Box(arrays, origin * self.voxel_um, self.voxel_um, self.kind)


# ── Session ───────────────────────────────────────────────────────────────────

def _as_xyz(value, name):
    arr = np.asarray(value, float).ravel()
    if arr.size == 1:
        arr = np.repeat(arr, 3)
    if arr.size != 3:
        raise ValueError(f"{name} 要写成一个数或 [x, y, z]，实际是 {value!r}")
    return arr


class Session:
    """Everything one QC round needs, loaded once.

    cfg keys: see configs/region_qc.example.yaml.
    """

    def __init__(self, cfg, log=print):
        from registration_ants.atlas_utils import load_ccf_ontology_json

        self.cfg = cfg
        self.log = log
        self.run_dir = Path(cfg["run_dir"])
        self.out_dir = Path(cfg["out_dir"])
        self.rng_seed = int(cfg.get("seed", 0))

        self.run_cfg, self.run_cfg_path = load_run_config(self.run_dir)
        run_sample = (self.run_cfg or {}).get("sample", {}) or {}
        run_cells = (self.run_cfg or {}).get("cells", {}) or {}

        cv = cfg.get("cell_voxel_um") or run_cells.get("voxel_size_um")
        if not cv:
            raise ValueError("不知道细胞坐标的体素大小：run_dir 里没有 pipeline 配置快照，"
                             "请在配置里写 cell_voxel_um: [x, y, z]。")
        self.cell_voxel_um = _as_xyz(cv, "cell_voxel_um")

        self.warnings = []
        if run_sample.get("reposition_plan"):
            self.warnings.append(
                "这个 run 做过 reposition：碎片上的细胞，画出来的脑区边界是合拢后的位置，"
                "和原图对不上。选区（按第 9 列）仍然正确；判断碎片上的细胞时不要看边界。")

        self.structures = load_ccf_ontology_json(cfg["ontology_json"])
        self.region_ids, self.region_roots = resolve_region_ids(self.structures, cfg["regions"])

        self.labels = LabelVolume(_find_labels(self.run_dir))
        self.cells = load_cells(self.run_dir)
        self.phys = self.cells[["x", "y", "z"]].to_numpy(float) * self.cell_voxel_um
        self._check_resample_columns()

        sel = select_cells(self.cells, self.region_ids, cfg.get("classes"))
        self.selected_index = sel.index.to_numpy()
        self.selected = sel.copy()
        sel_phys = self.phys[self.selected_index]
        self.selected["lookup_id"] = self.labels.lookup(sel_phys)
        self.selected["depth_um"] = signed_depth_um(self.labels, self.region_ids, sel_phys)
        self.selected["lookup_agrees"] = self.selected["lookup_id"].isin(list(self.region_ids))

        self.half_um = None
        self.source = self._build_source()
        self.sites = self._pick_sites()

    # -- setup ---------------------------------------------------------------

    def _check_resample_columns(self):
        """cell_points wrote column 3-5 as phys / fine spacing.  If the voxel
        size here does not reproduce them, every box below would be cut in the
        wrong place, silently -- so refuse, like single_sample.py does."""
        if "xr" not in self.cells.columns:
            return
        xr = self.cells[["xr", "yr", "zr"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        pred = (self.phys - self.labels.origin) / self.labels.spacing
        ok = np.isfinite(xr).all(axis=1)
        if not ok.any():
            return
        drift = np.abs(pred[ok] - xr[ok]).max(axis=1)
        moved = drift > 1e-3
        if moved.mean() > 0.5:
            raise ValueError(
                f"cell_voxel_um = {self.cell_voxel_um.tolist()} 复现不出细胞表自己的第 3-5 列"
                f"（{100 * moved.mean():.0f}% 的细胞对不上，最大差 {drift.max():.3g} 个标签体素）。"
                "体素大小写错了，照这个切出来的框和细胞不在同一块组织上。")
        if moved.any():
            self.warnings.append(
                f"{int(moved.sum())} 个细胞的第 3-5 列和原始坐标换算不一致 —— 通常是 reposition "
                "搬过的碎片细胞。")

    def _build_source(self):
        cfg = self.cfg
        kind = cfg.get("source", "volume")
        run_sample = (self.run_cfg or {}).get("sample", {}) or {}
        if kind == "tiles":
            if not cfg.get("channels"):
                raise ValueError("source: tiles 需要 channels（每个通道的 tile 根目录，xml 所在层）。")
            self.half_um = _as_xyz(cfg.get("half_extent_um", [200, 200, 40]), "half_extent_um")
            return TileSource(cfg["channels"], self.cell_voxel_um)
        if kind == "volume":
            images = cfg.get("images")
            if not images:
                if not run_sample.get("raw_tiff"):
                    raise ValueError("source: volume 没写 images，run_dir 的配置快照里也没有 sample.raw_tiff。")
                images = {Path(run_sample["raw_tiff"]).stem: run_sample["raw_tiff"]}
            vox = cfg.get("image_voxel_um") or run_sample.get("voxel_size_um")
            if not vox:
                raise ValueError("source: volume 需要 image_voxel_um（或配置快照里的 sample.voxel_size_um）。")
            self.half_um = _as_xyz(cfg.get("half_extent_um", [600, 600, 160]), "half_extent_um")
            return VolumeSource(images, _as_xyz(vox, "image_voxel_um"))
        raise ValueError(f"source 只能是 tiles 或 volume，实际是 {kind!r}")

    def _pick_sites(self):
        """Cells to visit, in visiting order.

        `sites` in the config (a list of cell_id strings, "class:row") pins
        the list exactly.  Otherwise draw `n_sites` at random, round-robin
        over classes so rare classes are seen at all, optionally only near the
        boundary (`prefer: boundary`, |depth| <= boundary_um), and thinned so
        no two sites are closer than `min_separation_um` -- two sites in the
        same box are one site.
        """
        cfg = self.cfg
        sel = self.selected
        if cfg.get("sites"):
            want = [str(s) for s in cfg["sites"]]
            missing = [s for s in want if s not in set(sel["cell_id"])]
            if missing:
                raise ValueError(f"sites 里这些细胞不在选中的集合里：{missing[:5]}")
            return sel.set_index("cell_id").loc[want].reset_index()

        rng = np.random.default_rng(self.rng_seed)
        pool = sel
        if cfg.get("prefer", "random") == "boundary":
            lim = float(cfg.get("boundary_um", 100))
            pool = sel[np.abs(sel["depth_um"]) <= lim]
            if pool.empty:
                self.warnings.append(f"没有细胞在边界 {lim:g} µm 以内，改为随机选。")
                pool = sel
        elif cfg.get("prefer") not in (None, "random"):
            raise ValueError("prefer 只能是 random 或 boundary")

        queues = {c: list(rng.permutation(g.index.to_numpy()))
                  for c, g in pool.groupby("class_name")}
        order = []
        while any(queues.values()):
            for c in sorted(queues):
                if queues[c]:
                    order.append(queues[c].pop())
        n = int(cfg.get("n_sites", 20))
        sep = float(cfg.get("min_separation_um", 2 * float(self.half_um[0])))
        chosen, pts = [], []
        for i in order:
            p = self.phys[i]
            if pts and np.min(np.linalg.norm(np.array(pts) - p, axis=1)) < sep:
                continue
            chosen.append(i)
            pts.append(p)
            if len(chosen) >= n:
                break
        return sel.loc[chosen].reset_index(drop=False).rename(columns={"index": "cell_index"})

    # -- per site ------------------------------------------------------------

    def site_phys(self, k):
        row = self.sites.iloc[k]
        return np.array([row["x"], row["y"], row["z"]], float) * self.cell_voxel_um

    def cut_site(self, k):
        """-> dict with the image box, the label box and the cells inside."""
        c = self.site_phys(k)
        lo, hi = c - self.half_um, c + self.half_um
        box = self.source.cut(lo, hi)
        lab, lab_origin = self.labels.box(lo, hi)

        inside = np.all((self.phys >= lo) & (self.phys <= hi), axis=1)
        idx = np.flatnonzero(inside)
        near = self.cells.iloc[idx].copy()
        near["px"], near["py"], near["pz"] = self.phys[idx].T
        near["selected"] = np.isin(idx, self.selected_index)
        near["is_target"] = near["cell_id"] == self.sites.iloc[k]["cell_id"]

        ratio = float("nan")
        anchor = self.cfg.get("signal_check_channel") or next(iter(box.arrays))
        if anchor in box.arrays and len(near):
            radius = np.maximum(1, np.rint(np.array([3.0, 3.0, 0.0]) / box.voxel_um)).astype(int)
            radius[2] = 0 if box.voxel_um[2] > 4 else radius[2]
            soma = near[~near["class_name"].str.contains("Sox9", case=False)] \
                if box.source == "tiles" else near
            if len(soma):
                pts = box.local(soma[["px", "py", "pz"]].to_numpy())
                ratio, _ = geom.signal_check(box.arrays[anchor], pts,
                                             np.random.default_rng(k), radius_px=radius)
        return {"box": box, "labels": lab, "labels_origin": lab_origin,
                "cells": near, "signal_ratio": ratio, "center": c}

    def region_label(self, sid):
        sid = int(sid)
        if sid == 0:
            return "background / 未归区"
        info = self.structures.get(sid)
        if info is None:
            return f"id {sid} (本体里没有)"
        return f"{info.get('acronym')} · {info.get('name')}"

    def describe_site(self, k):
        row = self.sites.iloc[k]
        d = row["depth_um"]
        depth = "far outside" if not np.isfinite(d) else f"{d:+.0f} µm"
        return {
            "site": k,
            "cell_id": row["cell_id"],
            "class_name": row["class_name"],
            "xyz_px": f"{row['x']:.0f}, {row['y']:.0f}, {int(row['z'])}",
            "table_region": self.region_label(row["region_id"]),
            "labels_region": self.region_label(row["lookup_id"]),
            "agrees": bool(row["lookup_agrees"]),
            "depth": depth,
            "tile": row.get("tile_name", ""),
        }

    # -- summary -------------------------------------------------------------

    def summary_lines(self):
        sel = self.selected
        roots = ", ".join(f"{a} ({n})" for _, n, a in self.region_roots)
        lines = [
            f"run      {self.run_dir}",
            f"regions  {roots}  —— 含后代共 {len(self.region_ids)} 个 id",
            f"cells    选中 {len(sel)} / 全部 {len(self.cells)}",
        ]
        if len(sel):
            d = sel["depth_um"].to_numpy()
            fin = d[np.isfinite(d)]
            lines.append(
                f"labels_in_sample 回查：{100 * sel['lookup_agrees'].mean():.1f}% 落在所选区域内；"
                f"不一致的 {int((~sel['lookup_agrees']).sum())} 个，"
                f"其中离边界 ≤20 µm 的 {int(((~sel['lookup_agrees']) & (d >= -20)).sum())} 个")
            if fin.size:
                q = np.percentile(fin, [5, 25, 50, 75, 95])
                lines.append("离区域边界深度 (µm, 正=在区内)  "
                             + "  ".join(f"p{p} {v:+.0f}" for p, v in zip((5, 25, 50, 75, 95), q)))
            by = sel.groupby("class_name").agg(
                n=("cell_id", "size"),
                agree=("lookup_agrees", "mean"),
                depth_med=("depth_um", lambda s: np.nanmedian(s[np.isfinite(s)]) if np.isfinite(s).any() else np.nan))
            lines.append("按类别：")
            for name, r in by.iterrows():
                lines.append(f"  {name:28s} n={int(r.n):7d}  回查一致 {100 * r.agree:5.1f}%  "
                             f"深度中位 {r.depth_med:+6.0f} µm")
        lines.append(f"source   {self.source.kind}，体素 {self.source.voxel_um.tolist()} µm，"
                     f"半框 {self.half_um.tolist()} µm")
        lines.append(f"sites    {len(self.sites)} 个")
        lines += [f"⚠️  {w}" for w in self.warnings]
        return lines


# ── Frame check ───────────────────────────────────────────────────────────────

def frame_offset_scan(cells, phys, vol, voxel_um, marker="RFP", dz_um=None,
                      xy_shift_um=(0.0, 0.0), n=15000, by_tile=False,
                      min_cells=300, seed=0):
    """Is the image under the cells the tissue they were detected in?

    Takes a whole-brain volume in which one marker is bright (the 561 nm
    registration tiff shows RFP; a 730 nm one shows Sox9) and compares the
    mean intensity under marker+ cells with that under marker- cells, both
    sampled at the same shift.  Both groups sit in tissue, so unlike "cells vs
    random positions" the tissue/background contrast cancels and only the
    marker contrast is left: the ratio peaks where the cells line up with
    their own nuclei.

    -> DataFrame, one row per group (the whole run, or each tile), with
       best_dz_um, ratio at the best dz and at dz 0.

    Measured on the TSC runs (RFP, 555 registration tiffs): runs whose tiles
    agree peak at dz = -20 um in every tile, ratio ~3 at the peak -- consistent
    with registration-tiff slice k being the average of global slices
    4k+1..4k+4 (inferred from the offset, not from the downsampling code).  A per-tile best dz that scatters
    over tens of microns is a tile z-origin error in the centroids.
    """
    if dz_um is None:
        dz_um = np.arange(-200, 81, 4)
    dz_um = np.asarray(dz_um, float)
    voxel_um = np.asarray(voxel_um, float)
    Z, Y, X = vol.shape
    pos = cells["class_name"].str.split("_").apply(lambda p: marker in p).to_numpy()
    groups = cells.groupby(cells["tile_name"].fillna("NA").astype(str)) if by_tile \
        else [("all", cells)]
    rng = np.random.default_rng(seed)

    def mean_at(P, dz):
        k = np.rint((P + (xy_shift_um[0], xy_shift_um[1], dz)) / voxel_um).astype(np.int64)
        ok = np.all((k >= 0) & (k < (X, Y, Z)), axis=1)
        k = k[ok]
        if len(k) == 0:
            return np.nan
        order = np.lexsort((k[:, 0], k[:, 1], k[:, 2]))   # page-ordered reads
        k = k[order]
        return float(np.asarray(vol[k[:, 2], k[:, 1], k[:, 0]], dtype=float).mean())

    rows = []
    for name, g in groups:
        where = np.flatnonzero(cells.index.isin(g.index))
        a, b = where[pos[where]], where[~pos[where]]
        if len(a) < min_cells or len(b) < min_cells:
            continue
        a = rng.choice(a, min(n, len(a)), replace=False)
        b = rng.choice(b, min(n, len(b)), replace=False)
        curve = np.array([mean_at(phys[a], d) / mean_at(phys[b], d) for d in dz_um])
        if not np.isfinite(curve).any():
            continue
        j = int(np.nanargmax(curve))
        at0 = curve[np.argmin(np.abs(dz_um))]
        rows.append({"group": name, "n_cells": len(g), "best_dz_um": float(dz_um[j]),
                     "ratio_best": float(curve[j]), "ratio_dz0": float(at0)})
    return pd.DataFrame(rows)


def frame_check_lines(session, marker="RFP"):
    """Run-level and per-tile frame_offset_scan on the session's volume."""
    src = session.source
    if not isinstance(src, VolumeSource):
        return ["frame check 只在 source: volume 下可用（要整脑图，tile 模式读不起）。"]
    name, vol = next(iter(src.vols.items()))
    shift = session.cfg.get("frame_check_xy_shift_um", (-1.3, -1.3))
    whole = frame_offset_scan(session.cells, session.phys, vol, src.voxel_um, marker,
                              xy_shift_um=shift)
    if whole.empty:
        return [f"frame check：{marker}+ 或 {marker}- 细胞太少，做不了。"]
    w = whole.iloc[0]
    lines = [f"frame check（{marker}+ / {marker}- 在 {name} 上的强度比）："
             f"dz=0 时 {w.ratio_dz0:.2f}，最佳 dz {w.best_dz_um:+.0f} µm 时 {w.ratio_best:.2f}"]
    tiles = frame_offset_scan(session.cells, session.phys, vol, src.voxel_um, marker,
                              xy_shift_um=shift, by_tile=True, n=3000)
    if len(tiles) > 1:
        counts = tiles["best_dz_um"].value_counts().sort_index()
        spread = np.percentile(np.repeat(tiles.best_dz_um, tiles.n_cells), [10, 90])
        lines.append(f"  逐 tile 最佳 dz（{len(tiles)} 个 tile）："
                     + "  ".join(f"{int(k):+d}×{v}" for k, v in counts.items()))
        lines.append(f"  按细胞数加权 p10~p90：{spread[0]:+.0f} ~ {spread[1]:+.0f} µm")
    elif tiles.empty or tiles.iloc[0]["group"] == "NA":
        lines.append("  细胞表里没有 tile_name，只能给整体偏移。")
    if w.ratio_best < 1.2:
        lines.append(f"  ⚠️ 峰值比值太低 —— 这张图上 {marker} 可能不亮，换 frame_check_marker。")
    elif abs(w.best_dz_um + 20) > 12 or (len(tiles) > 1 and spread[1] - spread[0] > 16):
        lines.append("  ⚠️ 偏移不是正常的 -20 µm 或各 tile 不一致：细胞坐标和原图对不上，"
                     "先查 cell_centroids 的 z 约定，再看任何按层/按区的结果。")
    return lines


# ── Verdicts ──────────────────────────────────────────────────────────────────

VERDICTS = {
    "ok": "区域正确",
    "wrong_region": "区域错误",
    "not_cell": "不是细胞",
    "unsure": "看不清",
}


class VerdictStore:
    """out_dir/verdicts.csv, one row per (run_dir, cell_id), rewritten on
    every change so a crash loses at most the current click."""

    COLUMNS = ["run_dir", "cell_id", "class_name", "x", "y", "z", "table_region_id",
               "lookup_region_id", "depth_um", "source", "verdict", "note", "time"]

    def __init__(self, path, run_dir):
        self.path = Path(path)
        self.run_dir = str(run_dir)
        if self.path.exists():
            self.df = pd.read_csv(self.path, dtype={"cell_id": str, "note": str})
        else:
            self.df = pd.DataFrame(columns=self.COLUMNS)

    def get(self, cell_id):
        m = (self.df["run_dir"] == self.run_dir) & (self.df["cell_id"] == cell_id)
        if not m.any():
            return None, ""
        r = self.df.loc[m].iloc[-1]
        note = r.get("note", "")
        return r["verdict"], "" if pd.isna(note) else str(note)

    def set(self, site_row, verdict, note, source):
        from datetime import datetime
        if verdict not in VERDICTS:
            raise ValueError(verdict)
        m = (self.df["run_dir"] == self.run_dir) & (self.df["cell_id"] == site_row["cell_id"])
        self.df = self.df.loc[~m]
        new = pd.DataFrame([{
            "run_dir": self.run_dir, "cell_id": site_row["cell_id"],
            "class_name": site_row["class_name"],
            "x": site_row["x"], "y": site_row["y"], "z": site_row["z"],
            "table_region_id": int(site_row["region_id"]),
            "lookup_region_id": int(site_row["lookup_id"]),
            "depth_um": float(site_row["depth_um"]),
            "source": source, "verdict": verdict, "note": note,
            "time": datetime.now().isoformat(timespec="seconds"),
        }])
        self.df = new if self.df.empty else pd.concat([self.df, new], ignore_index=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.path, index=False)

    def tally(self):
        sub = self.df[self.df["run_dir"] == self.run_dir]
        return sub["verdict"].value_counts().to_dict()
