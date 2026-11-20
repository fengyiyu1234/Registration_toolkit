"""Per-cell-type density heatmaps in standard atlas space, one PNG per cell type.

    python visualization/cell_density_png.py local_config/single_sample.yaml
    python visualization/cell_density_png.py local_config/single_sample.yaml --sigma 3 --out my_dir

Reads <sample_dir>/cell_registration/<class>/cell_registration.csv (cols 6-8 = atlas-space
voxel x,y,z, the same ones the atlas viewer plots), projects the cells along each axis,
smooths and draws three views (horizontal / sagittal / coronal) over the atlas brain
outline. Axes are in mm on the atlas grid, with the anatomical direction spelled out.
Output goes to <sample_dir>/cell_density_png/.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import tifffile
import yaml
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# One hue, light -> dark (magnitude). Zero density is left unpainted so the grey brain
# silhouette shows through instead of reading as "a little".
CMAP = LinearSegmentedColormap.from_list(
    "density", ["#deebf7", "#9ecae1", "#4292c6", "#08519c", "#08306b"])
BRAIN_FILL, BRAIN_EDGE, INK, MUTED = "#eceff1", "#78909c", "#1f2933", "#52606d"

# (title, projected-away axis, horizontal axis, vertical axis, x label, y label)
# Atlas array order is (z, y, x); axes below are 0=z (DV), 1=y (AP), 2=x (ML).
VIEWS = [
    ("Horizontal", 0, 2, 1, "Medial  ←  →  Lateral", "Anterior  ←  →  Posterior"),
    ("Sagittal",   2, 1, 0, "Anterior  ←  →  Posterior", "Ventral  ←  →  Dorsal"),
    ("Coronal",    1, 2, 0, "Medial  ←  →  Lateral", "Ventral  ←  →  Dorsal"),
]


def load_atlas(path):
    # The configured name may carry a .tif the file on disk does not (or vice versa).
    for p in (path, path[:-4] if path.lower().endswith(".tif") else path + ".tif"):
        if os.path.exists(p):
            # extensionless files are tifs; only the ITK formats go through SimpleITK
            if p.lower().endswith((".nii", ".nii.gz", ".mhd", ".mha", ".nrrd")):
                return sitk.GetArrayFromImage(sitk.ReadImage(p))
            return tifffile.imread(p)
    raise FileNotFoundError(f"atlas label volume not found: {path}")


def voxel_um(sample_dir, default=20.0):
    hits = glob.glob(os.path.join(sample_dir, "*_in_atlas.nii.gz"))
    return float(sitk.ReadImage(hits[0]).GetSpacing()[0]) if hits else default


def load_cells(csv_path, shape):
    df = pd.read_csv(csv_path, header=None, usecols=[6, 7, 8])
    xyz = df.dropna().to_numpy(float)
    zyx = np.rint(xyz[:, [2, 1, 0]]).astype(int)
    inside = ((zyx >= 0) & (zyx < np.array(shape))).all(axis=1)
    return zyx[inside], len(zyx)


def draw(cls, zyx, n_total, brain, um, sigma, out_png):
    mm = um / 1000.0
    ratios = [brain.shape[h] / brain.shape[v] for _, _, h, v, _, _ in VIEWS]
    fig = plt.figure(figsize=(15, 5.8), dpi=200, facecolor="white")
    gs = fig.add_gridspec(1, 4, width_ratios=ratios + [0.06], wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    dens = []
    for ax, (title, drop, ax_h, ax_v, lab_h, lab_v) in zip(axes, VIEWS):
        # rows = ax_v, cols = ax_h; for all three views that is also the order
        # brain.any(axis=drop) leaves the remaining axes in.
        h2, _, _ = np.histogram2d(zyx[:, ax_v], zyx[:, ax_h],
                                  bins=[brain.shape[ax_v], brain.shape[ax_h]],
                                  range=[[0, brain.shape[ax_v]], [0, brain.shape[ax_h]]])
        # cells per mm^2 of the projection plane
        d = gaussian_filter(h2, sigma) / (mm * mm)
        dens.append((ax, title, d, brain.any(axis=drop), lab_h, lab_v, ax_h, ax_v))
    vmax = max(np.percentile(d[d > 0], 99.5) if (d > 0).any() else 1 for _, _, d, *_ in dens)

    im = None
    for ax, title, d, sil, lab_h, lab_v, ax_h, ax_v in dens:
        ext = [0, brain.shape[ax_h] * mm, 0, brain.shape[ax_v] * mm]
        ax.imshow(sil, origin="lower", extent=ext, cmap=matplotlib.colors.ListedColormap([BRAIN_FILL]),
                  vmin=0, vmax=1, alpha=sil.astype(float), interpolation="nearest")
        shown = np.ma.masked_where(d < vmax * 0.01, d)
        im = ax.imshow(shown, origin="lower", extent=ext, cmap=CMAP, vmin=0, vmax=vmax,
                       interpolation="bilinear")
        ax.contour(np.linspace(0, ext[1], sil.shape[1]), np.linspace(0, ext[3], sil.shape[0]),
                   sil.astype(float), levels=[0.5], colors=BRAIN_EDGE, linewidths=0.8)
        ax.set_aspect("equal", anchor="N")
        ax.set_title(title, color=INK, fontsize=12, loc="left")
        ax.set_xlabel(lab_h + "  (mm)", color=MUTED, fontsize=9)
        ax.set_ylabel(lab_v + "  (mm)", color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label("Cell density (cells / mm², projected)", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)
    fig.suptitle(f"{cls}   ·   n = {len(zyx):,} cells in atlas grid"
                 + (f" ({n_total - len(zyx):,} outside grid not shown)" if n_total > len(zyx) else ""),
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.savefig(out_png, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="local_config/single_sample.yaml")
    ap.add_argument("--sigma", type=float, default=3.0, help="Gaussian smoothing, in atlas voxels")
    ap.add_argument("--out", default=None, help="output dir (default <sample_dir>/cell_density_png)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    sample_dir = cfg["sample_dir"]
    out_dir = args.out or os.path.join(sample_dir, "cell_density_png")
    os.makedirs(out_dir, exist_ok=True)

    brain = load_atlas(cfg["std_atlas_path"]) > 0
    um = voxel_um(sample_dir)
    print(f"atlas grid (z,y,x) = {brain.shape}, {um:g} µm/voxel")

    for csv in sorted(glob.glob(os.path.join(sample_dir, "cell_registration", "*", "cell_registration.csv"))):
        cls = os.path.basename(os.path.dirname(csv))
        zyx, n_total = load_cells(csv, brain.shape)
        if len(zyx) == 0:
            print(f"skip {cls}: no cells inside the atlas grid")
            continue
        out_png = os.path.join(out_dir, f"{cls}.png")
        draw(cls, zyx, n_total, brain, um, args.sigma, out_png)
        print(f"{cls}: {len(zyx)}/{n_total} cells -> {out_png}")


if __name__ == "__main__":
    main()
