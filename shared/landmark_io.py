"""Shared helper: read and write the landmark CSVs that pass between the tools
in this repo, in one place with one set of checks.

FORMAT
------
`index,axis-0,axis-1,axis-2` -- the layout napari's own Points-layer writer
produces. Landmarks are placed in napari itself (open the image, add a Points
layer, save it), so a CSV exported from there can be fed straight to any tool
here, and one written by this module can be dragged back into napari.

The three coordinate columns are VOXEL indices in the image's own numpy array
order, `(z, y, x)`, axis 0 being the real imaging/atlas planes. That is what
every napari/SimpleITK tool in this repo works in, and deliberately NOT
`ants.image_read().numpy()`'s reversed order for the same file.

CONVERSIONS
-----------
Two steps, kept separate because different consumers stop at different points:

    to_xyz(voxel_zyx)                       -> voxel indices in (x,y,z) order
    voxels_to_physical(voxel_zyx, sp, org)  -> microns in (x,y,z) order

Every image in this project has an identity direction matrix and carries any
cropping offset in its origin, so `physical = voxel_xyz * spacing + origin`
holds exactly. Points placed on one image must be converted with THAT image's
geometry -- mixing a sample CSV with the atlas's spacing/origin produces
coordinates that are wrong everywhere and flagged nowhere.

ROW ORDER IS THE PAIRING
------------------------
Nothing in the file names a landmark. Row i in a sample CSV and row i in the
atlas CSV are "the same anatomical location" purely by convention, so
registration_eval.py's compute_tre is only as correct as that ordering.
read_landmark_csv() cannot check it -- a mis-ordered row shows up as one
landmark with a wildly larger TRE than the rest, and nowhere else.

Not runnable on its own -- imported by registration_eval.py and the tests.
"""
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["axis-0", "axis-1", "axis-2"]


def read_landmark_csv(csv_path, shape_zyx=None, role="landmark"):
    """Parse a landmark CSV into an (N,3) array of voxel coordinates, (z,y,x).

    `shape_zyx`, when given, is the numpy shape of the image the points are
    supposed to belong to; points outside it are a hard error, which is the
    cheapest available check that a CSV was placed on the image it is now being
    used with. `role` ("sample"/"atlas"/...) only ever appears in messages.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{role} landmarks CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing column(s) {missing}; found {list(df.columns)}.\n"
            f"Expected the index,axis-0,axis-1,axis-2 layout napari's own "
            f"Points-layer writer produces.")
    if len(df) == 0:
        raise ValueError(f"{csv_path} has no rows.")

    zyx = df[COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(zyx).all():
        bad = [int(i) for i in np.flatnonzero(~np.isfinite(zyx).all(axis=1))]
        raise ValueError(f"{csv_path} has non-numeric/NaN coordinates on row(s) {bad}.")

    if shape_zyx is not None:
        shape_zyx = np.asarray(shape_zyx, dtype=float)
        outside = np.flatnonzero(((zyx < -0.5) | (zyx > shape_zyx - 0.5)).any(axis=1))
        if len(outside):
            raise ValueError(
                f"{len(outside)} of {len(zyx)} {role} landmark(s) fall outside the "
                f"{role} image, whose (z,y,x) voxel shape is "
                f"{tuple(int(v) for v in shape_zyx)}:\n"
                + "\n".join(f"    row {i}: (z,y,x) = "
                            + "(" + ", ".join(f"{c:.1f}" for c in zyx[i]) + ")"
                            for i in outside[:10])
                + ("\n    ..." if len(outside) > 10 else "")
                + f"\nThese points were placed on a DIFFERENT image than the one they "
                  f"are being used with.")
    return zyx


def write_landmark_csv(csv_path, voxel_zyx):
    """Write voxel coordinates in the format read_landmark_csv() expects."""
    voxel_zyx = np.asarray(voxel_zyx, dtype=float)
    if voxel_zyx.ndim != 2 or voxel_zyx.shape[1] != 3:
        raise ValueError(f"expected an (N,3) array of (z,y,x) voxels, got {voxel_zyx.shape}")
    df = pd.DataFrame(voxel_zyx, columns=COLUMNS)
    df.insert(0, "index", np.arange(len(df)))
    df.to_csv(csv_path, index=False)
    return Path(csv_path)


def to_xyz(voxel_zyx):
    """(z,y,x) -> (x,y,z), the order the physical-space conventions use."""
    return np.asarray(voxel_zyx, dtype=float)[:, ::-1]


def voxels_to_physical(voxel_zyx, spacing_xyz, origin_xyz):
    """(z,y,x) voxels -> (x,y,z) physical microns in that image's own space."""
    return to_xyz(voxel_zyx) * np.asarray(spacing_xyz, dtype=float) \
        + np.asarray(origin_xyz, dtype=float)
