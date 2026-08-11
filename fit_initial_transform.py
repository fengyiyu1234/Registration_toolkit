"""Turn two hand-placed landmark CSVs into a pair of initial deformation fields
for the ANTs pipeline in ../Registration_ants.

WHY THIS EXISTS
---------------
The P5 sample (right hemisphere, iDISCO-cleared, flattened to 61% of the
atlas's dorsoventral extent) is far enough from DevCCF P04 that intensity-driven
registration cannot establish correspondence in some regions on its own -- the
deep midline structures in particular, where there is no unambiguous intensity
landmark to lock onto. This script lets a human state the correspondence
directly: place matching points in both images, fit a deformation field through
them, and hand that field to `antsRegistration -r` so the optimizer starts from
the anatomy you specified instead of from a centre-of-mass guess.

    place_landmarks.py  ->  two CSVs  ->  THIS SCRIPT  ->  two warp fields
                                                       ->  Registration_ants
                                                           (registration.initial_transform)

DIRECTION SEMANTICS (measured, not assumed)
-------------------------------------------
`ants.fit_transform_to_paired_points(moving_points, fixed_points, ...)` returns
a transform in the RESAMPLING direction: applying it to a point in FIXED space
yields the corresponding point in MOVING space. Verified on a phantom with a
known analytic deformation -- apply(fixed point) landed 5.8um from the true
moving point, versus 106um from where it started. That is exactly the direction
`antsRegistration -r` / `ants.apply_transforms` expect, so nothing here is
inverted.

TWO FIELDS, NOT ONE
-------------------
ANTs cannot invert a deformation field that it did not produce itself. Given
only the forward field, the reverse chain in the pipeline (labels_in_sample,
cell-to-region assignment) is missing a link -- on a phantom that dropped Dice
from 0.99 to 0.86. So the same landmark pairs are fitted twice, with the roles
of the two point sets swapped:

    <prefix>init_fwd.nii.gz   fit(moving=sample pts, fixed=atlas pts,  domain=atlas grid)
    <prefix>init_inv.nii.gz   fit(moving=atlas pts,  fixed=sample pts, domain=20um sample grid)

These are two independent fits of the same correspondence, not an inverse pair
computed from one another.

The inverse field's domain is the 20um sample grid, NOT the grid the sample
landmarks were placed on. Downstream it warps atlas labels into sample space,
and the reference grid for that step in the pipeline is
`{name}_fine_{N}um.nii.gz` -- a field on the raw acquisition grid would be the
wrong shape for it. Pass that file as --sample-domain-image; if the pipeline
has not produced it yet, this script synthesises an empty isotropic --target-um
grid spanning the same physical extent (only the grid is used, never the
intensities).

COORDINATE CONVENTION (fixed; do not change)
--------------------------------------------
place_landmarks.py writes napari/SimpleITK voxel coordinates in (z,y,x) array
order. Every image in this project has an identity direction matrix and carries
its cropping offset in the origin, so:

    zyx  = df[["axis-0", "axis-1", "axis-2"]]
    xyz  = zyx[:, ::-1]
    phys = xyz * voxel_size_xyz + origin_xyz

which is _landmark_io.voxels_to_physical(), shared with registration_eval.py
and place_landmarks.py. Sample points are converted with the SAMPLE image's
geometry and atlas points with the ATLAS image's -- mixing them produces a
plausible-looking field that is wrong everywhere.

THE TWO SIDES ARE AT DIFFERENT RESOLUTIONS, ON PURPOSE
------------------------------------------------------
Sample landmarks are placed on the RAW acquisition TIFF (registration.tif,
2273x3974x157 at [2.6, 2.6, 32.0] um), because the 20um resample drops xy by a
factor of 8 and the structures being pointed at stop being identifiable. Atlas
landmarks are placed on the prepared 20um atlas. That mismatch costs nothing:
the fit is done in physical coordinates (microns), so each side only has to
know its own voxel size.

Cropping does not disturb this either -- crop_for_registration shifts the
origin to keep physical space continuous, so a point placed on the uncropped
raw stack is in the same physical frame as the cropped volume the registration
actually runs on.

A RAW TIFF HAS NO VOXEL SIZE IN ITS HEADER
------------------------------------------
SimpleITK reads registration.tif with spacing (1.0, 1.0, 1.0). Believing that
would put every physical coordinate out by a factor of 2.6 to 32 with nothing
to signal it, so --sample-voxel-size is REQUIRED whenever --sample-image is a
.tif/.tiff, and its absence is a hard error rather than a default. For a
.nii.gz the header is trusted (and --sample-voxel-size, if given anyway,
overrides it and warns about the disagreement).

The values are in (x, y, z) order -- the same order as `sample.voxel_size_um`
in the Registration_ants config, and the REVERSE of the CSV's (z, y, x)
columns. Getting that backwards is the mistake selftest 5 exists to catch.

NO PIXEL DATA IS EVER READ
--------------------------
Only headers. The fit needs each image's grid (shape/spacing/origin), never its
intensities, so the domains are reconstructed as empty images from header info
alone. That is what keeps a 2.7 GB raw TIFF from being loaded to answer a
question about its shape, and it is why --sample-image may point at a file far
too large to open.

THE ATLAS IMAGE MUST BE THE CACHED, REORIENTED, CROPPED ONE
-----------------------------------------------------------
i.e. the file the pipeline actually registers against
(P04_LSFM_20um_1_-3_2__320-640_full_full.nii.gz), not the DevCCF release file --
those have a different axis order and a different origin, so points placed on
one mean nothing in the other. The bounds check below catches the common case
(voxel indices that fall outside the image), and the paired-distance summary
catches the rest.

FITTING LEVELS -- READ THIS BEFORE TRUSTING THE OUTPUT
------------------------------------------------------
`number_of_fitting_levels` sets how fine the B-spline control lattice is over
the WHOLE image domain, so the right value depends on the domain's physical
size, and the default of 4 is not always enough. Measured on this project's
actual atlas domain (6.4 x 16.0 x 11.2 mm) with 14 landmarks:

    levels=4    median residual  100.2 um   max 1756.6 um    <- field misses the landmarks
    levels=5    median residual    0.9 um   max   13.5 um
    levels=6    median residual    0.9 um   max    2.2 um

The per-point residual table below is what tells you which case you are in.
Uniformly large residuals mean the lattice is too coarse -- raise
--fitting-levels. A few large residuals among small ones mean the two CSVs
disagree about the order of those landmarks -- unless those rows sit near the
fitting domain's edge, where the lattice is unconstrained and no number of
levels helps (see DOMAIN_PAD_FRAC). The report distinguishes the two.

COST
----
Both fits cover their whole domain grid, so the cost is set by the images, not
by the number of landmarks. Measured end to end on this project's real files
(atlas 560x800x320, sample 251x517x295, both 20um, 14 landmarks):

    forward fit  305 s      init_fwd.nii.gz  1.56 GB   (atlas grid)
    inverse fit   83 s      init_inv.nii.gz  0.42 GB   (sample grid)
    total       7.5 min     peak RSS 14.5 GB

Higher --fitting-levels costs more of both. A derived inverse domain is padded
on both sides of three axes, so it costs (1 + 2*DOMAIN_PAD_FRAC)^3 = 1.7x the
grid it replaces -- 67 M voxels against the real fine grid's 38 M for s12t.
Passing the real --sample-domain-image avoids that as well as being the grid
the pipeline wants. Reading no pixel data means --sample-image's own size is
irrelevant: the 2.7 GB raw TIFF costs a header read (measured: 0.00 s).

USAGE
-----
    conda activate antsreg
    python fit_initial_transform.py --selftest      # synthetic data, ~6 s

    # landmarks placed on the raw acquisition stack (the normal case)
    python fit_initial_transform.py \
        --sample-landmarks   landmarks_sample.csv \
        --atlas-landmarks    landmarks_atlas.csv \
        --sample-image       registration.tif \
        --sample-voxel-size  2.6 2.6 32.0 \
        --sample-domain-image tsc12t_fine_20um.nii.gz \
        --atlas-image        P04_LSFM_20um_1_-3_2__320-640_full_full.nii.gz \
        --out-prefix         /path/to/s12t_ \
        --fitting-levels     5

    # landmarks placed on the 20um resample instead: the header carries the
    # voxel size, and the same image serves as the inverse field's domain
    python fit_initial_transform.py \
        --sample-landmarks landmarks_sample.csv --atlas-landmarks landmarks_atlas.csv \
        --sample-image tsc12t_fine_20um.nii.gz \
        --atlas-image  P04_LSFM_20um_1_-3_2__320-640_full_full.nii.gz \
        --out-prefix   /path/to/s12t_ --fitting-levels 5

    python fit_initial_transform.py                 # no args -> the same form
                                                    # window place_landmarks.py uses
    python fit_initial_transform.py --no-form       # straight from
                                                    # configs/fit_initial_transform.yaml

Either pass all five required paths on the command line, or none of them and
let configs/fit_initial_transform.yaml and/or the form supply them; half a
command line is refused rather than quietly topped up from a config.
--sample-voxel-size, --sample-domain-image and --target-um are qualifiers on
those five, so they may be given alongside either route and always win.

Only ants/numpy/pandas/tifffile/pyyaml are needed; `registration_ants` is
deliberately not imported. PyQt5 is imported lazily, and only when the form is
actually shown, so both the flag and --no-form paths run headless.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import NamedTuple

import ants
import numpy as np
import pandas as pd
import tifffile

import _landmark_io  # sibling module
import _local_config  # sibling module

# Fewer than this cannot constrain a 3D deformation field in any useful way --
# 6 points is already generous for something with three degrees of freedom per
# control point. 10-20 well-spread points is the working range.
MIN_LANDMARKS = 6
RECOMMENDED_LANDMARKS = 10
DEFAULT_FITTING_LEVELS = 4

# A point whose residual exceeds max(OUTLIER_FACTOR * median, one voxel) is
# flagged. The one-voxel floor matters: when every point fits to well under a
# voxel, 3x the median is still sub-voxel noise and would flag half the table
# for nothing. Sub-voxel residuals are not actionable -- the landmarks were
# clicked at voxel resolution in the first place.
OUTLIER_FACTOR = 3.0

# Isotropic voxel size of the grid the pipeline registers on, used only when no
# --sample-domain-image is available and the inverse field's domain has to be
# synthesised. Matches the pipeline's `{name}_fine_{N}um.nii.gz`.
DEFAULT_TARGET_UM = 20.0

# A B-spline fit is poorly constrained within about two control-point spans of
# its domain's edge, and unlike a coarse lattice that is NOT something more
# --fitting-levels fixes. Measured on the phantom, fitting the same 15 pairs on
# a domain whose edge sat ~90 um from the outermost landmark:
#
#     pad   levels=4 max resid   levels=5 max resid
#      0%        140.5 um             71.6 um
#      5%        101.5 um             23.5 um
#     10%         73.8 um              4.1 um     <- chosen
#     20%         24.7 um              0.2 um
#
# So a synthesised domain is padded. 10% and not 20% because the pad is applied
# on BOTH sides of all three axes -- it costs (1 + 2*pad)^3 in voxels, i.e. 1.7x
# at 10% and 2.7x at 20% -- and 4 um is already a fifth of a 20 um voxel, well
# under the precision anyone clicks landmarks with. On the real s12t stack that
# is a 67 M voxel domain rather than 105 M.
#
# A domain passed in with --sample-domain-image cannot be padded (it has to stay
# the grid the pipeline uses), which is why proximity to the edge is offered as
# an explanation for flagged rows instead.
DOMAIN_PAD_FRAC = 0.10
EDGE_SPANS = 2.0

TIFF_SUFFIXES = (".tif", ".tiff")


# =====================================================================================
# Image geometry (headers only -- see "NO PIXEL DATA IS EVER READ")
# =====================================================================================
class Geometry(NamedTuple):
    """Everything this script needs to know about an image: its grid.

    `shape_zyx` is numpy/napari/SimpleITK order, matching the CSV's axis-0..2.
    `spacing_xyz` and `origin_xyz` are physical (x,y,z), matching ANTs and
    `sample.voxel_size_um` in the Registration_ants config. The two orders are
    reverses of each other and conflating them is the failure this whole
    module is arranged to prevent, hence the suffixes on every name.
    """
    shape_zyx: tuple
    spacing_xyz: tuple
    origin_xyz: tuple
    path: str
    note: str          # where spacing came from; printed so it can be checked

    @property
    def shape_xyz(self):
        return tuple(reversed(self.shape_zyx))

    @property
    def extent_xyz(self):
        """Physical size of the grid in microns, (x,y,z)."""
        return tuple(float(n * s) for n, s in zip(self.shape_xyz, self.spacing_xyz))

    def describe(self):
        return (f"shape(z,y,x)={self.shape_zyx} spacing={_fmt_vec(self.spacing_xyz)} "
                f"origin={_fmt_vec(self.origin_xyz)}  [{self.note}]")


def parse_voxel_size(value, source="--sample-voxel-size"):
    """Accept the three (x,y,z) voxel sizes as a list (YAML) or a string (form).

    Returns a 3-tuple of floats, or None when nothing was given.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
    else:
        parts = list(value)
    try:
        nums = [float(p) for p in parts]
    except (TypeError, ValueError):
        raise ValueError(
            f"{source} must be three numbers in (x, y, z) order, e.g. "
            f"'2.6 2.6 32.0'; got {value!r}") from None
    if len(nums) != 3:
        raise ValueError(
            f"{source} needs exactly three values (x y z), got {len(nums)}: {value!r}")
    if any(n <= 0 for n in nums):
        raise ValueError(f"{source} must be positive, got {nums}")
    return tuple(nums)


def read_geometry(path, voxel_size_xyz=None, role="image"):
    """Read `path`'s grid from its header alone. Never touches pixel data.

    `voxel_size_xyz` is required for TIFFs (their headers carry no spacing) and
    optional -- but authoritative -- for anything SimpleITK/ITK can describe.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{role} image not found: {path}")

    if path.suffix.lower() in TIFF_SUFFIXES:
        with tifffile.TiffFile(str(path)) as tif:
            shape = tuple(int(v) for v in tif.series[0].shape)
        if len(shape) != 3:
            raise ValueError(
                f"{path} has shape {shape}; this script expects a 3D stack "
                f"(z, y, x) -- the same array place_landmarks.py showed you.")
        if voxel_size_xyz is None:
            raise ValueError(
                f"--sample-voxel-size is required when the {role} image is a TIFF.\n"
                f"  {path}\n"
                f"TIFF headers carry no voxel size -- SimpleITK reports (1.0, 1.0, 1.0) "
                f"for this file. Using that would scale every physical coordinate wrong "
                f"by a factor of 2.6 to 32 and nothing downstream would notice.\n"
                f"Pass the acquisition voxel size in (x, y, z) microns, the same values "
                f"as sample.voxel_size_um in the Registration_ants config, e.g.\n"
                f"    --sample-voxel-size 2.6 2.6 32.0")
        # A raw acquisition stack has no cropping offset to record, and nothing
        # writes one into a TIFF; physical space starts at 0.
        return Geometry(shape, tuple(voxel_size_xyz), (0.0, 0.0, 0.0), str(path),
                        "TIFF: shape from header, spacing given, origin 0")

    info = ants.image_header_info(str(path))
    shape_zyx = tuple(int(v) for v in reversed(info["dimensions"]))
    header_spacing = tuple(float(v) for v in info["spacing"])
    origin = tuple(float(v) for v in info["origin"])

    # voxels_to_physical() is `voxel_xyz * spacing + origin`, which is only the
    # right conversion when the direction matrix is identity. Every image in
    # this project satisfies that; one that does not would produce coordinates
    # that are wrong everywhere and flagged nowhere.
    direction = np.asarray(info["direction"], dtype=float)
    if not np.allclose(direction, np.eye(len(shape_zyx)), atol=1e-6):
        raise ValueError(
            f"{path} has a non-identity direction matrix:\n{direction}\n"
            f"Landmark voxel->physical conversion here assumes identity (see the "
            f"COORDINATE CONVENTION section). Reorient the file first -- the "
            f"pipeline's cached atlas and its resampled sample volumes are all "
            f"identity-direction.")

    if voxel_size_xyz is not None:
        if not np.allclose(header_spacing, voxel_size_xyz, rtol=1e-3):
            print(f"  WARNING: --sample-voxel-size {_fmt_vec(voxel_size_xyz)} disagrees "
                  f"with {path.name}'s header spacing {_fmt_vec(header_spacing)}. "
                  f"Using the value you passed.")
        return Geometry(shape_zyx, tuple(voxel_size_xyz), origin, str(path),
                        "header shape/origin, spacing overridden on the command line")
    return Geometry(shape_zyx, header_spacing, origin, str(path), "from the file header")


def blank_domain(geom):
    """An empty ANTsImage with `geom`'s grid, for use as a fit/write domain.

    `fit_transform_to_paired_points` and `transform_to_displacement_field` use
    the domain only for its grid, so there is no reason to read the real
    volume's intensities -- and every reason not to, when it is 2.7 GB.
    """
    arr = np.zeros(geom.shape_xyz, dtype="float32")   # ANTs arrays are (x,y,z)
    return ants.from_numpy(arr, spacing=tuple(float(v) for v in geom.spacing_xyz),
                           origin=tuple(float(v) for v in geom.origin_xyz))


def inverse_domain_geometry(sample_geom, domain_image_path, target_um):
    """The grid the inverse field must live on.

    Three ways to get it, in order of preference:
      1. --sample-domain-image: the pipeline's own {name}_fine_{N}um.nii.gz.
         Always right, because it is literally the file downstream uses.
      2. the sample image itself, when landmarks were placed on a .nii.gz --
         in that workflow that file already IS the resampled grid.
      3. synthesised: an empty isotropic `target_um` grid over the same
         physical extent, for when the pipeline has not been run yet.
    """
    if domain_image_path:
        geom = read_geometry(domain_image_path, role="sample domain")
        # Same physical object, so the two extents should agree to within a
        # voxel or two. A mismatch means a different sample, or a crop that the
        # landmarks were not placed relative to.
        ratio = np.asarray(geom.extent_xyz) / np.asarray(sample_geom.extent_xyz)
        if np.any(ratio < 0.8) or np.any(ratio > 1.25):
            print(f"  WARNING: --sample-domain-image covers "
                  f"{_fmt_vec(geom.extent_xyz)} um but the landmark image covers "
                  f"{_fmt_vec(sample_geom.extent_xyz)} um.\n"
                  f"  Those should be the same physical volume. Check that both refer "
                  f"to the same sample, and that --sample-voxel-size is right.")
        return geom

    if Path(sample_geom.path).suffix.lower() not in TIFF_SUFFIXES:
        return sample_geom._replace(
            note=f"{sample_geom.note}; reused as the inverse domain")

    # Padded outwards (see DOMAIN_PAD_FRAC): the raw stack is barely larger than
    # the brain, so an exactly-fitting grid puts surface landmarks in the band
    # where the B-spline fit is unconstrained.
    extent = np.asarray(sample_geom.extent_xyz)
    pad = DOMAIN_PAD_FRAC * extent
    origin = tuple(float(o - p) for o, p in zip(sample_geom.origin_xyz, pad))
    shape_xyz = tuple(max(1, int(np.ceil(e / target_um))) for e in extent + 2 * pad)
    return Geometry(tuple(reversed(shape_xyz)), (target_um,) * 3, origin, sample_geom.path,
                    f"synthesised {target_um:g} um isotropic grid, the sample's extent "
                    f"padded by {DOMAIN_PAD_FRAC:.0%}")


def landmarks_near_domain_edge(points_phys, geom, fitting_levels):
    """Rows lying close enough to `geom`'s edge for the fit to be unconstrained.

    A cubic B-spline control point influences two spans either side, so a
    landmark within ~two spans of the boundary is partly supported by control
    points outside the domain, which are never solved for.

    Reported only as an explanation for rows the residual table ALREADY
    flagged, never on its own: the band is a generous fraction of the domain
    (a quarter of it at --fitting-levels 4) and most landmarks inside it fit
    perfectly well, so a standalone warning here would fire constantly and mean
    nothing. What it is good for is telling a genuine outlier caused by the
    domain edge -- which more levels will NOT fix -- apart from one caused by a
    mis-ordered CSV.
    """
    extent = np.asarray(geom.extent_xyz, dtype=float)
    lower = np.asarray(geom.origin_xyz, dtype=float)
    margin = EDGE_SPANS * extent / (2 ** max(fitting_levels - 1, 1))
    slack = np.minimum(points_phys - lower, (lower + extent) - points_phys)
    return {int(i) for i in np.flatnonzero((slack < margin).any(axis=1))}


# =====================================================================================
# Loading landmarks
# =====================================================================================
def load_landmarks(csv_path, geom, role):
    """Read a place_landmarks.py CSV into `geom`'s physical coordinates.

    Returns (phys_xyz, voxel_zyx), both (N,3) float arrays. `phys_xyz` is in
    microns; `voxel_zyx` is kept only so the residual table can print
    coordinates you can type back into napari.

    Parsing, the format checks and the out-of-bounds check are _landmark_io's;
    only the role-specific advice on top of the bounds error is this script's,
    because the two ways to land points outside an image have two different
    causes worth naming.
    """
    try:
        zyx = _landmark_io.read_landmark_csv(csv_path, shape_zyx=geom.shape_zyx, role=role)
    except ValueError as exc:
        if role == "atlas" and "outside" in str(exc):
            raise ValueError(
                f"{exc}\nFor the atlas side this is usually a raw DevCCF release file "
                f"passed where the pipeline's reoriented+cropped cache belongs "
                f"(different axis order, different origin).") from None
        if role == "sample" and "outside" in str(exc):
            raise ValueError(
                f"{exc}\nFor the sample side this usually means --sample-image is not "
                f"the image the points were placed on: landmarks clicked on the raw "
                f"acquisition stack do not fit inside the 20 um resample, whose voxel "
                f"indices are ~8x smaller in x and y.") from None
        raise
    return _landmark_io.voxels_to_physical(zyx, geom.spacing_xyz, geom.origin_xyz), zyx


def check_landmark_sets(sample_phys, atlas_phys, sample_vox, atlas_vox,
                        sample_geom, atlas_geom):
    """Everything that can be checked before spending 10 minutes on two fits.

    Hard errors on counts (a fit built from mis-paired rows is worthless);
    warnings on the things that are merely suspicious.
    """
    n_s, n_a = len(sample_phys), len(atlas_phys)
    if n_s != n_a:
        raise ValueError(
            f"Landmark counts differ: sample has {n_s}, atlas has {n_a}.\n"
            f"The two CSVs are paired by ROW ORDER -- there are no point names in "
            f"the files -- so they must contain the same anatomical locations, in "
            f"the same order, and therefore the same number of rows.")
    if n_s < MIN_LANDMARKS:
        raise ValueError(
            f"Only {n_s} landmark pair(s); at least {MIN_LANDMARKS} are required "
            f"and {RECOMMENDED_LANDMARKS}-20 well-spread pairs are recommended. "
            f"Fewer than that cannot constrain a 3D deformation field.")

    print(f"landmarks: {n_s} pairs")
    if n_s < RECOMMENDED_LANDMARKS:
        print(f"  NOTE: {n_s} pairs is above the hard minimum but below the "
              f"recommended {RECOMMENDED_LANDMARKS}-20.")

    # Duplicate points inside one CSV: usually a double-click in napari, which
    # leaves the two CSVs the same length but shifts every later row by one.
    for name, vox in (("sample", sample_vox), ("atlas", atlas_vox)):
        d = np.linalg.norm(vox[:, None, :] - vox[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        dup = np.argwhere(d < 1.0)
        dup = {tuple(sorted(p)) for p in dup}
        if dup:
            pairs = ", ".join(f"rows {i} and {j}" for i, j in sorted(dup))
            print(f"  WARNING: {name} CSV has points less than one voxel apart ({pairs}) "
                  f"-- a stray double-click here shifts every later row by one.")

    raw = np.linalg.norm(sample_phys - atlas_phys, axis=1)
    print(f"raw paired distance (before any fitting): "
          f"median {np.median(raw):.1f} um, max {raw.max():.1f} um")
    print(f"  sample: {sample_geom.describe()}")
    print(f"  atlas : {atlas_geom.describe()}")

    # Both extremes are informative. Near-zero means the same CSV was given
    # twice; enormous means an axis-order or wrong-image mistake -- including a
    # transposed --sample-voxel-size, which is why this check exists at all.
    sample_extent = np.asarray(sample_geom.extent_xyz)
    if np.median(raw) < max(sample_geom.spacing_xyz):
        print("  WARNING: the two point sets are less than one voxel apart on average. "
              "Are both --sample-landmarks and --atlas-landmarks pointing at the same file?")
    if raw.max() > 1.5 * np.linalg.norm(sample_extent):
        print("  WARNING: paired distances exceed the sample image's own diagonal. "
              "That usually means one CSV was placed on a different image, or "
              "--sample-voxel-size is wrong or in the wrong order (it is x y z, the "
              "reverse of the CSV columns), or the two CSVs are not in the same "
              "anatomical order.")
    return raw


# =====================================================================================
# Fitting
# =====================================================================================
def fit_field(moving_phys, fixed_phys, domain_img, fitting_levels, label):
    """One call to ants.fit_transform_to_paired_points, with timing.

    The returned transform maps FIXED-space points to MOVING-space points (see
    the module docstring) -- i.e. it is already in the direction
    antsRegistration's -r and ants.apply_transforms want, and must not be
    inverted.
    """
    # flush: on the real atlas grid this call takes minutes, and a redirected
    # stdout would otherwise show nothing at all until it returns.
    print(f"\n[{label}] fitting diffeo field over "
          f"{_shape_zyx(domain_img)} voxels, "
          f"number_of_fitting_levels={fitting_levels} ...", flush=True)
    t0 = time.time()
    xf = ants.fit_transform_to_paired_points(
        moving_phys, fixed_phys, transform_type="diffeo",
        domain_image=domain_img, number_of_fitting_levels=fitting_levels)
    print(f"[{label}] fitted in {time.time() - t0:.1f} s")
    return xf


def warp_points(xf, fixed_phys):
    """Apply `xf` to each fixed-space point, giving its moving-space image."""
    return np.array([ants.apply_ants_transform_to_point(xf, list(p)) for p in fixed_phys])


def residual_report(xf, fixed_phys, moving_phys, fixed_vox, moving_vox, raw,
                    voxel_um, fixed_name, moving_name, label, full=True,
                    near_edge=(), edge_remedy=""):
    """Per-landmark fitting residual: how far apply(fixed point) lands from the
    moving point it is supposed to reach.

    This is the single most useful output of this script. A field is fitted
    through whatever pairs it is given, so a pair that names two DIFFERENT
    anatomical locations does not fail -- it silently drags the field. Its
    residual is the only place it shows up.

    Returns (resid, flagged_indices).
    """
    warped = warp_points(xf, fixed_phys)
    resid = np.linalg.norm(warped - moving_phys, axis=1)
    median = float(np.median(resid))
    threshold = max(OUTLIER_FACTOR * median, voxel_um)
    flagged = [int(i) for i in np.flatnonzero(resid > threshold)]

    # full=False (the second, reverse fit -- the same landmarks, refitted with
    # the roles swapped) prints only the rows that need attention, since the
    # full table would otherwise be printed twice for one set of landmarks.
    rows = range(len(resid)) if full else flagged
    print(f"\n[{label}] per-landmark fitting residual"
          + ("" if full else " (flagged rows only)"))
    if len(rows):
        print(f"  {'idx':>4}  {fixed_name + ' voxel (z,y,x)':>26}  "
              f"{moving_name + ' voxel (z,y,x)':>26}  {'paired':>10}  {'residual':>10}")
        for i in rows:
            mark = "  <-- CHECK" if i in flagged else ""
            print(f"  {i:>4}  {_fmt_vec(fixed_vox[i]):>26}  {_fmt_vec(moving_vox[i]):>26}  "
                  f"{raw[i]:>7.1f} um  {resid[i]:>7.1f} um{mark}")
    print(f"  residual: median {median:.1f} um, max {resid.max():.1f} um "
          f"(flag threshold {threshold:.1f} um = max({OUTLIER_FACTOR:g}x median, one voxel))")

    if flagged:
        print(f"  {len(flagged)} point(s) flagged: {flagged}")
        edge_hits = sorted(set(flagged) & set(near_edge))
        if edge_hits:
            print(f"  Row(s) {edge_hits} sit near this fit's domain edge, where the "
                  f"B-spline lattice is\n  unconstrained. That is the likely cause for "
                  f"those, and raising --fitting-levels\n  will NOT fix it.")
            if edge_remedy:
                print(f"  {edge_remedy}")
        if sorted(set(flagged) - set(near_edge)):
            print(f"  A few outliers among otherwise small residuals almost always means "
                  f"those rows are\n  in a different order in the two CSVs -- open both in "
                  f"place_landmarks.py (preload the\n  points CSV) and check that row i is "
                  f"the same anatomical location in each.")
    else:
        print("  no outliers")

    if median > voxel_um:
        print(f"  WARNING: the median residual is above one voxel ({voxel_um:.1f} um), i.e. "
              f"EVERY point\n  fits badly, not just a few. That is an under-resolved "
              f"B-spline lattice rather than a\n  landmark problem -- re-run with a higher "
              f"--fitting-levels (each level halves the\n  control-point spacing; on this "
              f"project's atlas domain 4 -> 5 took the median\n  residual from 100 um to "
              f"0.9 um).")
    return resid, flagged


# =====================================================================================
# Writing
# =====================================================================================
def write_field(xf, domain_img, path, label):
    """Save `xf` as a displacement-field NIfTI on `domain_img`'s grid.

    The displacement magnitude summary is the same measurement used to judge
    whether a registration's own 1Warp.nii.gz actually deformed anything (see
    Registration_ants' PROGRESS_LOG) -- printed here so the initialisation can
    be compared against it directly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{label}] writing {path} ...", flush=True)
    t0 = time.time()
    field = ants.transform_to_displacement_field(xf, domain_img)
    ants.image_write(field, str(path))
    size_mb = path.stat().st_size / 1e6
    size = f"{size_mb / 1000:.2f} GB" if size_mb >= 1000 else f"{size_mb:.1f} MB"
    print(f"[{label}] wrote {size} in {time.time() - t0:.1f} s")

    mag = np.sqrt(np.einsum("...i,...i->...", field.numpy(), field.numpy()))
    print(f"[{label}] displacement magnitude over the domain: "
          f"median {np.median(mag):.1f} um, 90th pct {np.percentile(mag, 90):.1f} um, "
          f"max {mag.max():.1f} um")
    return path


def verify_written_field(path, xf, fixed_phys, voxel_um, label):
    """Round-trip check: the field read back from disk must move points the same
    way the fitted transform does."""
    reloaded = ants.transform_from_displacement_field(ants.image_read(str(path)))
    deviation = np.linalg.norm(warp_points(reloaded, fixed_phys) - warp_points(xf, fixed_phys), axis=1)
    print(f"[{label}] disk round-trip: max deviation from the in-memory transform "
          f"{deviation.max():.3f} um")
    if deviation.max() > 0.5 * voxel_um:
        print(f"  WARNING: the file on disk does not reproduce the fitted transform "
              f"(more than half a voxel).\n  Do not use it -- something went wrong in "
              f"transform_to_displacement_field/image_write.")
    return deviation


def print_config_snippet(fwd_path, inv_path):
    print("\n" + "=" * 78)
    print("Paste into the Registration_ants sample config:\n")
    print("registration:")
    print("  initial_transform:")
    print(f"    path: {Path(fwd_path).resolve()}")
    print(f"    inverse_path: {Path(inv_path).resolve()}")
    print("=" * 78)


def _fmt_vec(values):
    return "(" + ", ".join(f"{v:.1f}" for v in np.asarray(values, dtype=float)) + ")"


def _shape_zyx(img):
    """ANTs reports shape as (x,y,z); everything a human reads in this project
    -- napari, the CSVs, SimpleITK -- is (z,y,x)."""
    return tuple(int(v) for v in reversed(img.shape))


# =====================================================================================
# Driver
# =====================================================================================
def run(sample_landmarks, atlas_landmarks, sample_image, atlas_image, out_prefix,
        fitting_levels=DEFAULT_FITTING_LEVELS, sample_voxel_size=None,
        sample_domain_image=None, target_um=DEFAULT_TARGET_UM):
    """Fit both fields and write them. Returns (fwd_path, inv_path).

    `sample_voxel_size` is (x,y,z) microns, required when `sample_image` is a
    TIFF. `sample_domain_image` is the 20 um grid the INVERSE field must live
    on; without it one is derived (see inverse_domain_geometry).
    """
    sample_voxel_size = parse_voxel_size(sample_voxel_size)
    sample_geom = read_geometry(sample_image, sample_voxel_size, role="sample")
    atlas_geom = read_geometry(atlas_image, role="atlas")
    inv_geom = inverse_domain_geometry(sample_geom, sample_domain_image, target_um)

    sample_phys, sample_vox = load_landmarks(sample_landmarks, sample_geom, "sample")
    atlas_phys, atlas_vox = load_landmarks(atlas_landmarks, atlas_geom, "atlas")
    raw = check_landmark_sets(sample_phys, atlas_phys, sample_vox, atlas_vox,
                              sample_geom, atlas_geom)
    print(f"  inv domain: {inv_geom.describe()}")

    # The residual flag threshold's floor is "one voxel" of the space the
    # residual is measured in -- landmarks were clicked at voxel resolution, so
    # anything finer than that is not actionable. On the raw stack that is the
    # 32 um z step, deliberately: z is where the clicking is least precise.
    sample_voxel_um = float(max(sample_geom.spacing_xyz))
    atlas_voxel_um = float(max(atlas_geom.spacing_xyz))

    # Only used to explain rows the residual tables flag; see the function's
    # docstring for why this is never reported on its own.
    fwd_near_edge = landmarks_near_domain_edge(atlas_phys, atlas_geom, fitting_levels)
    inv_near_edge = landmarks_near_domain_edge(sample_phys, inv_geom, fitting_levels)

    atlas_domain = blank_domain(atlas_geom)
    inv_domain = blank_domain(inv_geom)

    # Forward: registration's own direction (fixed=atlas, moving=sample), so the
    # field lives on the atlas grid and can be handed straight to -r.
    fwd = fit_field(sample_phys, atlas_phys, atlas_domain, fitting_levels, "fwd")
    _, fwd_flagged = residual_report(
        fwd, atlas_phys, sample_phys, atlas_vox, sample_vox, raw,
        sample_voxel_um, "atlas", "sample", "fwd", full=True, near_edge=fwd_near_edge,
        edge_remedy="Move those landmarks inwards: this close to the atlas cache's "
                    "border they are\n  outside the region the registration covers "
                    "anyway.")

    # Reverse: the same correspondence fitted with the roles swapped, because
    # ANTs cannot invert a field it did not produce (see module docstring). Its
    # domain is the 20 um sample grid, not the grid the landmarks were placed on.
    inv = fit_field(atlas_phys, sample_phys, inv_domain, fitting_levels, "inv")
    _, inv_flagged = residual_report(
        inv, sample_phys, atlas_phys, sample_vox, atlas_vox, raw,
        atlas_voxel_um, "sample", "atlas", "inv", full=False, near_edge=inv_near_edge,
        edge_remedy="A --sample-domain-image cropped tight to the brain does this; a "
                    "derived domain\n  is padded instead. Re-run without it to compare.")

    print()
    fwd_path = write_field(fwd, atlas_domain, f"{out_prefix}init_fwd.nii.gz", "fwd")
    verify_written_field(fwd_path, fwd, atlas_phys, sample_voxel_um, "fwd")
    inv_path = write_field(inv, inv_domain, f"{out_prefix}init_inv.nii.gz", "inv")
    verify_written_field(inv_path, inv, sample_phys, atlas_voxel_um, "inv")

    flagged = sorted(set(fwd_flagged) | set(inv_flagged))
    if flagged:
        print(f"\nNOTE: landmark row(s) {flagged} were flagged above. The fields were "
              f"written anyway --\nthey are usable, but fix the pairing and re-run if "
              f"those rows are genuinely mismatched.")

    print_config_snippet(fwd_path, inv_path)
    return fwd_path, inv_path


# =====================================================================================
# Selftests (synthetic data only)
# =====================================================================================
_PH_SPACING = 20.0
_PH_CZ = 800.0          # z about which the phantom is compressed, in um
_PH_SCALE = 0.65        # the sample's real dorsoventral ratio to the atlas

# The real acquisition's (x,y,z) voxel size. Anisotropic by more than 12x, on
# purpose: under isotropic spacing a transposed (z,y,x) vector still gives the
# right answer by luck, so only these numbers can prove the axis order.
_RAW_VOXEL = (2.6, 2.6, 32.0)


def _phantom_blob(phys_xyz):
    """Indicator of an ellipsoid defined in ATLAS physical space."""
    x, y, z = phys_xyz[..., 0], phys_xyz[..., 1], phys_xyz[..., 2]
    return (((x - 600) / 350.0) ** 2 + ((y - 700) / 420.0) ** 2
            + ((z - 800) / 500.0) ** 2) < 1.0


def _phantom_grid_phys(shape_xyz, origin_xyz):
    idx = np.indices(shape_xyz).astype(float)
    return np.stack([idx[k] * _PH_SPACING + origin_xyz[k] for k in range(3)], axis=-1)


def _atlas_to_sample(p):
    """The phantom's known analytic deformation: flatten along z to 61-65%."""
    q = np.array(p, dtype=float)
    q[..., 2] = _PH_CZ + (q[..., 2] - _PH_CZ) * _PH_SCALE
    return q


def _sample_to_atlas(q):
    p = np.array(q, dtype=float)
    p[..., 2] = _PH_CZ + (p[..., 2] - _PH_CZ) / _PH_SCALE
    return p


def _phantom_images():
    """An atlas image and a sample image of the same object, the sample
    flattened along z by the known factor and on its own grid/origin (a
    different shape and a nonzero origin, so the origin term in the
    voxel->physical conversion is actually exercised)."""
    atlas_shape, atlas_origin = (60, 70, 80), (0.0, 0.0, 0.0)
    samp_shape, samp_origin = (64, 74, 76), (-40.0, -60.0, 120.0)
    atlas_arr = _phantom_blob(_phantom_grid_phys(atlas_shape, atlas_origin)).astype("float32")
    samp_arr = _phantom_blob(_sample_to_atlas(
        _phantom_grid_phys(samp_shape, samp_origin))).astype("float32")
    atlas_img = ants.from_numpy(atlas_arr, spacing=(_PH_SPACING,) * 3, origin=atlas_origin)
    samp_img = ants.from_numpy(samp_arr, spacing=(_PH_SPACING,) * 3, origin=samp_origin)
    return atlas_img, atlas_arr, samp_img, samp_arr


def _phantom_landmarks():
    """Landmarks on the ellipsoid's surface plus its centre -- the analogue of
    what a human actually clicks (identifiable boundary features), and what the
    field needs in order to constrain the outline rather than just the middle."""
    n = 14
    golden = np.pi * (3 - np.sqrt(5))
    pts = []
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(max(0.0, 1 - z * z))
        th = golden * i
        pts.append([600 + 350 * r * np.cos(th), 700 + 420 * r * np.sin(th), 800 + 500 * z])
    pts.append([600.0, 700.0, 800.0])
    atlas_pts = np.array(pts)
    return atlas_pts, _atlas_to_sample(atlas_pts)


def _phys_to_voxel_zyx(phys, ref):
    """Inverse of the load_landmarks() conversion, for writing phantom CSVs.

    `ref` is a Geometry or an ANTsImage -- the phantom tests hold both.
    """
    spacing = ref.spacing_xyz if isinstance(ref, Geometry) else ref.spacing
    origin = ref.origin_xyz if isinstance(ref, Geometry) else ref.origin
    xyz = (np.asarray(phys) - np.asarray(origin)) / np.asarray(spacing)
    return xyz[:, ::-1]


def _geom_of(img, path="<phantom>"):
    """Geometry of an in-memory phantom, without a file to read a header from."""
    return Geometry(_shape_zyx(img), tuple(float(v) for v in img.spacing),
                    tuple(float(v) for v in img.origin), path, "phantom")


def _dice(a, b):
    a, b = a > 0.5, b > 0.5
    return 2.0 * (a & b).sum() / (a.sum() + b.sum())


def _write_phantom_tiff(path, shape_zyx):
    """An empty stack standing in for registration.tif: no spacing in its
    header, and read for its shape alone."""
    tifffile.imwrite(str(path), np.zeros(shape_zyx, dtype="uint8"))
    return path


def selftest_direction_and_residual(tmp):
    print("1. known analytic deformation: fit through the points, check both directions")
    atlas_img, atlas_arr, samp_img, samp_arr = _phantom_images()
    atlas_pts, samp_pts = _phantom_landmarks()

    fwd = fit_field(samp_pts, atlas_pts, atlas_img, 4, "selftest-fwd")
    warped = warp_points(fwd, atlas_pts)

    # (1) every fixed point ends up closer to its moving partner than it started.
    #     Restricted to the points that actually have to move: the phantom's
    #     centre landmark sits on the compression axis, so its required
    #     displacement is exactly zero and "closer than before" is vacuous there
    #     (0 < 0 is false). Its residual is still covered by check (2).
    before = np.linalg.norm(atlas_pts - samp_pts, axis=1)
    after = np.linalg.norm(warped - samp_pts, axis=1)
    moves = before > 1.0
    assert moves.sum() == len(before) - 1 and before.max() > 100.0, \
        "phantom landmarks are not displaced the way this test assumes"
    assert (after[moves] < 0.1 * before[moves]).all(), (
        f"apply(fixed) did not land on the moving points -- the transform's direction "
        f"is not what this script assumes.\n"
        f"before={np.round(before, 1)}\nafter ={np.round(after, 1)}")
    print(f"   distance to the moving point: {np.median(before[moves]):.1f} um before -> "
          f"{np.median(after[moves]):.2f} um after (median over the {moves.sum()} "
          f"displaced landmarks)")

    # (2) the residual is small in absolute terms, not merely smaller.
    assert after.max() < 10.0, f"residual too large: max {after.max():.2f} um"

    # (3) the file on disk warps IMAGES in the direction the pipeline needs.
    inv = fit_field(atlas_pts, samp_pts, samp_img, 4, "selftest-inv")
    fwd_path = write_field(fwd, atlas_img, tmp / "st_init_fwd.nii.gz", "selftest-fwd")
    inv_path = write_field(inv, samp_img, tmp / "st_init_inv.nii.gz", "selftest-inv")
    verify_written_field(fwd_path, fwd, atlas_pts, _PH_SPACING, "selftest-fwd")

    d_none = _dice(ants.resample_image_to_target(samp_img, atlas_img).numpy(), atlas_arr)
    d_fwd = _dice(ants.apply_transforms(fixed=atlas_img, moving=samp_img,
                                        transformlist=[str(fwd_path)]).numpy(), atlas_arr)
    d_swapped = _dice(ants.apply_transforms(fixed=atlas_img, moving=samp_img,
                                            transformlist=[str(inv_path)]).numpy(), atlas_arr)
    d_back = _dice(ants.apply_transforms(fixed=samp_img, moving=atlas_img,
                                         transformlist=[str(inv_path)]).numpy(), samp_arr)
    print(f"   sample->atlas Dice: {d_none:.3f} untransformed, {d_fwd:.3f} with init_fwd, "
          f"{d_swapped:.3f} with the fields swapped")
    print(f"   atlas->sample Dice: {d_none:.3f} untransformed, {d_back:.3f} with init_inv")
    assert d_fwd > d_none + 0.10, f"init_fwd barely helped: {d_none:.3f} -> {d_fwd:.3f}"
    assert d_fwd > d_swapped + 0.20, (
        f"using init_inv where init_fwd belongs scored {d_swapped:.3f} vs {d_fwd:.3f} -- "
        f"too close to tell the two directions apart, so this test proves nothing")
    assert d_back > d_none + 0.10, f"init_inv barely helped: {d_none:.3f} -> {d_back:.3f}"
    print("   ok")


def selftest_swapped_pair_is_flagged(tmp):
    print("2. two landmarks swapped between the CSVs -> the residual table flags them")
    atlas_img, _, samp_img, _ = _phantom_images()
    atlas_pts, samp_pts = _phantom_landmarks()

    swapped = samp_pts.copy()
    i, j = 3, 9
    swapped[[i, j]] = swapped[[j, i]]

    xf = fit_field(swapped, atlas_pts, atlas_img, 4, "selftest-swap")
    raw = np.linalg.norm(swapped - atlas_pts, axis=1)
    resid, flagged = residual_report(
        xf, atlas_pts, swapped,
        _phys_to_voxel_zyx(atlas_pts, atlas_img), _phys_to_voxel_zyx(swapped, samp_img),
        raw, _PH_SPACING, "atlas", "sample", "selftest-swap", full=True)
    assert i in flagged and j in flagged, (
        f"the swapped rows {i} and {j} were not flagged; flagged={flagged}, "
        f"residuals={np.round(resid, 1)}")

    # And the clean fit of the same points must NOT flag anything, or the flag
    # carries no information.
    clean_xf = fit_field(samp_pts, atlas_pts, atlas_img, 4, "selftest-clean")
    _, clean_flagged = residual_report(
        clean_xf, atlas_pts, samp_pts,
        _phys_to_voxel_zyx(atlas_pts, atlas_img), _phys_to_voxel_zyx(samp_pts, samp_img),
        np.linalg.norm(samp_pts - atlas_pts, axis=1), _PH_SPACING,
        "atlas", "sample", "selftest-clean", full=False)
    assert not clean_flagged, f"correctly ordered landmarks were flagged: {clean_flagged}"
    print("   ok")


def _expect(exc_type, needle, fn):
    try:
        fn()
    except exc_type as exc:
        assert needle in str(exc), f"message does not mention {needle!r}: {exc}"
        return
    raise AssertionError(f"expected {exc_type.__name__} mentioning {needle!r}")


def selftest_input_validation(tmp):
    print("3. bad inputs are refused, with a message that names the problem")
    atlas_img, _, samp_img, _ = _phantom_images()
    atlas_geom, samp_geom = _geom_of(atlas_img), _geom_of(samp_img)
    atlas_pts, samp_pts = _phantom_landmarks()

    atlas_csv = tmp / "st_atlas.csv"
    samp_csv = tmp / "st_sample.csv"
    _landmark_io.write_landmark_csv(atlas_csv, _phys_to_voxel_zyx(atlas_pts, atlas_img))
    _landmark_io.write_landmark_csv(samp_csv, _phys_to_voxel_zyx(samp_pts, samp_img))

    # The CSV round trip must reproduce the physical coordinates it came from.
    reloaded, _ = load_landmarks(atlas_csv, atlas_geom, "atlas")
    assert np.allclose(reloaded, atlas_pts, atol=1e-6), "voxel<->physical round trip is lossy"
    reloaded_s, _ = load_landmarks(samp_csv, samp_geom, "sample")
    assert np.allclose(reloaded_s, samp_pts, atol=1e-6), \
        "sample-side round trip is lossy -- the origin term is probably being dropped"

    expect = _expect

    # Mismatched counts.
    short_csv = tmp / "st_short.csv"
    _landmark_io.write_landmark_csv(short_csv, _phys_to_voxel_zyx(samp_pts, samp_img)[:-1])
    short, short_vox = load_landmarks(short_csv, samp_geom, "sample")
    expect(ValueError, "counts differ",
           lambda: check_landmark_sets(short, atlas_pts, short_vox,
                                       _phys_to_voxel_zyx(atlas_pts, atlas_img),
                                       samp_geom, atlas_geom))

    # Too few points.
    few_csv = tmp / "st_few.csv"
    _landmark_io.write_landmark_csv(few_csv, _phys_to_voxel_zyx(samp_pts, samp_img)[:4])
    few, few_vox = load_landmarks(few_csv, samp_geom, "sample")
    expect(ValueError, str(MIN_LANDMARKS),
           lambda: check_landmark_sets(few, atlas_pts[:4], few_vox,
                                       _phys_to_voxel_zyx(atlas_pts, atlas_img)[:4],
                                       samp_geom, atlas_geom))

    # Atlas points read against the wrong (smaller) image -- the "raw DevCCF
    # file instead of the cropped cache" mistake, which lands out of bounds.
    small = _geom_of(ants.from_numpy(np.zeros((20, 20, 20), dtype="float32"),
                                     spacing=(_PH_SPACING,) * 3, origin=(0.0, 0.0, 0.0)))
    expect(ValueError, "outside", lambda: load_landmarks(atlas_csv, small, "atlas"))

    # Missing columns.
    junk_csv = tmp / "st_junk.csv"
    pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]}).to_csv(junk_csv, index=False)
    expect(ValueError, "axis-0", lambda: load_landmarks(junk_csv, atlas_geom, "atlas"))
    print("   ok")


def selftest_end_to_end(tmp):
    print("4. run() end to end on phantom CSVs -> two fields written, both usable")
    atlas_img, atlas_arr, samp_img, samp_arr = _phantom_images()
    atlas_pts, samp_pts = _phantom_landmarks()

    atlas_csv, samp_csv = tmp / "e2e_atlas.csv", tmp / "e2e_sample.csv"
    atlas_nii, samp_nii = tmp / "e2e_atlas.nii.gz", tmp / "e2e_sample.nii.gz"
    _landmark_io.write_landmark_csv(atlas_csv, _phys_to_voxel_zyx(atlas_pts, atlas_img))
    _landmark_io.write_landmark_csv(samp_csv, _phys_to_voxel_zyx(samp_pts, samp_img))
    ants.image_write(atlas_img, str(atlas_nii))
    ants.image_write(samp_img, str(samp_nii))

    fwd_path, inv_path = run(samp_csv, atlas_csv, samp_nii, atlas_nii,
                             str(tmp / "e2e_"), fitting_levels=4)
    assert fwd_path.name == "e2e_init_fwd.nii.gz", fwd_path
    assert inv_path.name == "e2e_init_inv.nii.gz", inv_path

    # Each field must be defined on ITS OWN domain's grid: fwd on the atlas
    # (that is what antsRegistration -r requires), inv on the sample.
    fwd_field, inv_field = ants.image_read(str(fwd_path)), ants.image_read(str(inv_path))
    assert fwd_field.shape == atlas_img.shape, \
        f"init_fwd is on the wrong grid: {fwd_field.shape} vs atlas {atlas_img.shape}"
    assert inv_field.shape == samp_img.shape, \
        f"init_inv is on the wrong grid: {inv_field.shape} vs sample {samp_img.shape}"
    assert np.allclose(fwd_field.origin, atlas_img.origin)
    assert np.allclose(inv_field.origin, samp_img.origin)

    d_none = _dice(ants.resample_image_to_target(samp_img, atlas_img).numpy(), atlas_arr)
    d_fwd = _dice(ants.apply_transforms(fixed=atlas_img, moving=samp_img,
                                        transformlist=[str(fwd_path)]).numpy(), atlas_arr)
    assert d_fwd > d_none + 0.10, f"end-to-end field barely helped: {d_none:.3f} -> {d_fwd:.3f}"
    print(f"   ok (Dice {d_none:.3f} -> {d_fwd:.3f} through files written by run())")


def selftest_anisotropic_voxel_size(tmp):
    print("5. anisotropic voxel size on a raw TIFF: (x,y,z) order, and never a silent (1,1,1)")
    tif = _write_phantom_tiff(tmp / "st_raw.tif", (10, 200, 150))

    # The whole point of the flag: a TIFF without one is refused, by name.
    _expect(ValueError, "--sample-voxel-size", lambda: read_geometry(tif, role="sample"))

    geom = read_geometry(tif, _RAW_VOXEL, role="sample")
    assert geom.shape_zyx == (10, 200, 150), geom.shape_zyx
    assert geom.spacing_xyz == _RAW_VOXEL, geom.spacing_xyz
    assert geom.origin_xyz == (0.0, 0.0, 0.0), geom.origin_xyz

    # The conversion, against hand arithmetic. The voxel is picked so that
    # applying the spacing in the CSV's (z,y,x) order instead changes the
    # answer: a reversed spacing cannot pass this.
    csv = tmp / "st_raw.csv"
    _landmark_io.write_landmark_csv(csv, np.array([[3.0, 2.0, 6.0]]))      # (z,y,x)
    phys, _ = load_landmarks(csv, geom, "sample")
    expected = np.array([[6 * 2.6, 2 * 2.6, 3 * 32.0]])                    # (x,y,z)
    assert np.allclose(phys, expected), f"got {phys}, hand-computed {expected}"
    assert not np.allclose(phys, [[6 * 32.0, 2 * 2.6, 3 * 2.6]]), \
        "this assertion cannot tell (x,y,z) spacing from (z,y,x) spacing"
    print(f"   voxel (z,y,x)=(3,2,6) with {_RAW_VOXEL} um -> {_fmt_vec(phys[0])} um (x,y,z)")

    # The flag's own parsing: YAML gives a list, the form gives a string.
    assert parse_voxel_size([2.6, 2.6, 32.0]) == _RAW_VOXEL
    assert parse_voxel_size("2.6 2.6 32.0") == _RAW_VOXEL
    assert parse_voxel_size("2.6, 2.6, 32.0") == _RAW_VOXEL
    assert parse_voxel_size(None) is None and parse_voxel_size("   ") is None
    _expect(ValueError, "exactly three", lambda: parse_voxel_size("2.6 2.6"))
    _expect(ValueError, "three numbers", lambda: parse_voxel_size("a b c"))
    _expect(ValueError, "positive", lambda: parse_voxel_size("2.6 0 32.0"))

    # A derived inverse domain: isotropic target_um, covering the raw's extent
    # with DOMAIN_PAD_FRAC of margin on every side.
    inv_geom = inverse_domain_geometry(geom, None, DEFAULT_TARGET_UM)
    assert inv_geom.spacing_xyz == (DEFAULT_TARGET_UM,) * 3, inv_geom.spacing_xyz
    assert inv_geom.shape_zyx != geom.shape_zyx, \
        "the inverse domain must not be the raw acquisition grid"

    raw_extent = np.asarray(geom.extent_xyz)                 # (390, 520, 320) um
    pad = DOMAIN_PAD_FRAC * raw_extent
    expected_xyz = tuple(int(np.ceil(e / DEFAULT_TARGET_UM)) for e in raw_extent + 2 * pad)
    assert inv_geom.shape_zyx == tuple(reversed(expected_xyz)), \
        (inv_geom.shape_zyx, tuple(reversed(expected_xyz)))
    assert np.allclose(inv_geom.origin_xyz, np.asarray(geom.origin_xyz) - pad), \
        "the padded grid must start before the sample's own origin"
    # It has to cover the raw's whole extent with room to spare on both sides.
    assert np.all(np.asarray(inv_geom.origin_xyz) < np.asarray(geom.origin_xyz))
    assert np.all(np.asarray(inv_geom.origin_xyz) + np.asarray(inv_geom.extent_xyz)
                  > np.asarray(geom.origin_xyz) + raw_extent)

    # The edge band narrows as the lattice gets finer, and is measured from the
    # domain's own origin/extent rather than from voxel counts.
    cube = Geometry((10, 10, 10), (20.0,) * 3, (0.0, 0.0, 0.0), "cube", "t")   # 200 um
    assert landmarks_near_domain_edge(np.array([[100.0, 100.0, 100.0]]), cube, 4) == set()
    assert landmarks_near_domain_edge(np.array([[10.0, 100.0, 100.0]]), cube, 4) == {0}
    assert landmarks_near_domain_edge(np.array([[30.0, 100.0, 100.0]]), cube, 4) == {0}
    assert landmarks_near_domain_edge(np.array([[30.0, 100.0, 100.0]]), cube, 6) == set()

    # blank_domain must reproduce a geometry exactly -- every domain the script
    # fits or writes on is now built this way rather than read from disk.
    dom = blank_domain(inv_geom)
    assert _shape_zyx(dom) == inv_geom.shape_zyx, (_shape_zyx(dom), inv_geom.shape_zyx)
    assert np.allclose(dom.spacing, inv_geom.spacing_xyz)
    assert np.allclose(dom.origin, inv_geom.origin_xyz)

    # When the sample side is a .nii.gz the header already describes the
    # resampled grid, so that file serves as its own inverse domain.
    _, _, samp_img, _ = _phantom_images()
    samp_nii = tmp / "st_dom_sample.nii.gz"
    ants.image_write(samp_img, str(samp_nii))
    nii_geom = read_geometry(samp_nii, role="sample")
    reused = inverse_domain_geometry(nii_geom, None, DEFAULT_TARGET_UM)
    assert reused.shape_zyx == nii_geom.shape_zyx, reused.shape_zyx
    assert np.allclose(reused.origin_xyz, nii_geom.origin_xyz), \
        "reusing the sample image as the domain dropped its origin"

    # An explicit --sample-domain-image wins over both. (The extent warning
    # printed here is expected: the phantom sample is not the raw stack.)
    print("   (the extent mismatch warning below is what this case is checking)")
    picked = inverse_domain_geometry(geom, samp_nii, DEFAULT_TARGET_UM)
    assert picked.shape_zyx == nii_geom.shape_zyx, picked.shape_zyx
    print("   ok")


def selftest_tiff_end_to_end(tmp):
    print("6. sample points on a raw TIFF + atlas points at 20 um -> right grids, small residual")
    atlas_img, _, _, _ = _phantom_images()
    atlas_pts, samp_pts = _phantom_landmarks()

    # A stand-in raw stack whose physical extent contains the sample points.
    raw = _write_phantom_tiff(tmp / "e2e_raw.tif", (40, 460, 400))
    raw_geom = read_geometry(raw, _RAW_VOXEL, role="sample")
    assert (samp_pts.min(axis=0) >= 0).all() and \
        (samp_pts.max(axis=0) < np.asarray(raw_geom.extent_xyz)).all(), \
        "the phantom's sample points do not fit inside the synthetic raw stack"

    atlas_nii = tmp / "e2e_t_atlas.nii.gz"
    ants.image_write(atlas_img, str(atlas_nii))
    atlas_csv, samp_csv = tmp / "e2e_t_atlas.csv", tmp / "e2e_t_sample.csv"
    _landmark_io.write_landmark_csv(atlas_csv, _phys_to_voxel_zyx(atlas_pts, atlas_img))
    _landmark_io.write_landmark_csv(samp_csv, _phys_to_voxel_zyx(samp_pts, raw_geom))

    # levels=5, the value configs/fit_initial_transform.example.yaml recommends
    # for real data: at 4 the derived domain's own residual table still flags
    # points, which is the behaviour DOMAIN_PAD_FRAC's table documents.
    fwd_path, inv_path = run(samp_csv, atlas_csv, raw, atlas_nii, str(tmp / "e2e_t_"),
                             fitting_levels=5, sample_voxel_size=_RAW_VOXEL,
                             target_um=DEFAULT_TARGET_UM)

    fwd_field = ants.image_read(str(fwd_path))
    inv_field = ants.image_read(str(inv_path))
    assert fwd_field.shape == atlas_img.shape, \
        f"init_fwd is not on the atlas grid: {fwd_field.shape} vs {atlas_img.shape}"

    expected_inv = inverse_domain_geometry(raw_geom, None, DEFAULT_TARGET_UM)
    assert _shape_zyx(inv_field) == expected_inv.shape_zyx, \
        (f"init_inv is on {_shape_zyx(inv_field)}, expected the derived 20 um grid "
         f"{expected_inv.shape_zyx}")
    assert np.allclose(inv_field.spacing, (DEFAULT_TARGET_UM,) * 3), inv_field.spacing
    assert _shape_zyx(inv_field) != raw_geom.shape_zyx, \
        "init_inv was written on the raw acquisition grid, which the pipeline cannot use"

    # The fit itself has to survive the two sides being at different
    # resolutions -- this is the assertion that says mixed resolution is fine.
    # BOTH directions: the inverse is the one fitted on a derived domain, so it
    # is the one that would show a padding or origin mistake.
    fwd_xf = ants.transform_from_displacement_field(fwd_field)
    fwd_resid = np.linalg.norm(warp_points(fwd_xf, atlas_pts) - samp_pts, axis=1)
    assert fwd_resid.max() < 10.0, \
        f"cross-resolution forward residual too large: max {fwd_resid.max():.1f} um"

    inv_xf = ants.transform_from_displacement_field(inv_field)
    inv_resid = np.linalg.norm(warp_points(inv_xf, samp_pts) - atlas_pts, axis=1)
    assert inv_resid.max() < 10.0, (
        f"inverse residual too large on the derived domain: max {inv_resid.max():.1f} um. "
        f"Unpadded, this landed at 71.6 um -- check DOMAIN_PAD_FRAC and the grid's origin.")
    print(f"   ok (residual max: fwd {fwd_resid.max():.2f} um, inv {inv_resid.max():.2f} um; "
          f"init_inv on {expected_inv.shape_zyx} at {DEFAULT_TARGET_UM:g} um, not the "
          f"{raw_geom.shape_zyx} raw grid)")


def run_selftests():
    import tempfile
    print("=== fit_initial_transform.py selftests (synthetic data only) ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        selftest_direction_and_residual(tmp)
        selftest_swapped_pair_is_flagged(tmp)
        selftest_input_validation(tmp)
        selftest_end_to_end(tmp)
        selftest_anisotropic_voxel_size(tmp)
        selftest_tiff_end_to_end(tmp)
    print("\n=== all selftests passed ===")
    return 0


# =====================================================================================
# CLI
# =====================================================================================
_CSV_FILTER = "CSV files (*.csv);;All files (*)"
_IMG_FILTER = "Images (*.nii *.nii.gz *.tif *.tiff);;All files (*)"

_FORM_FIELDS = [
    {"key": "sample_landmarks", "label": "Sample landmarks CSV (place_landmarks.py, role=sample)",
     "type": "open_file", "filter": _CSV_FILTER},
    {"key": "atlas_landmarks", "label": "Atlas landmarks CSV (same points, same order)",
     "type": "open_file", "filter": _CSV_FILTER},
    {"key": "sample_image", "label": "Sample image the sample points were placed on",
     "type": "open_file", "filter": _IMG_FILTER},
    {"key": "sample_voxel_size", "label": "Sample voxel size, x y z um (REQUIRED for TIFF)",
     "type": "text", "optional": True, "placeholder": "2.6 2.6 32.0"},
    {"key": "sample_domain_image",
     "label": "20 um sample grid for the inverse field (blank = derive)",
     "type": "open_file", "filter": _IMG_FILTER, "optional": True},
    {"key": "atlas_image", "label": "Atlas image the atlas points were placed on (cropped cache!)",
     "type": "open_file", "filter": _IMG_FILTER},
    {"key": "out_prefix", "label": "Output prefix (-> <prefix>init_fwd.nii.gz / init_inv.nii.gz)",
     "type": "save_file", "filter": "All files (*)"},
    {"key": "fitting_levels", "label": "Fitting levels (raise if all residuals are large)",
     "type": "int", "default": DEFAULT_FITTING_LEVELS, "minimum": 1, "maximum": 8},
    {"key": "target_um", "label": "Target voxel size for a derived inverse domain (um)",
     "type": "float", "default": DEFAULT_TARGET_UM, "minimum": 0.1, "maximum": 1000.0,
     "decimals": 1},
]


_REQUIRED = ["sample_landmarks", "atlas_landmarks", "sample_image", "atlas_image", "out_prefix"]

# Qualifiers on the five required paths rather than inputs in their own right:
# they may come from the command line even when the paths come from a config or
# the form, and the command line always wins.
_QUALIFIERS = ["sample_voxel_size", "sample_domain_image", "fitting_levels", "target_um"]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)

    parser = argparse.ArgumentParser(
        description="Fit initial forward/inverse deformation fields from paired landmark CSVs.")
    _local_config.add_config_arg(parser, "fit_initial_transform")
    parser.add_argument("--selftest", action="store_true",
                        help="run the built-in synthetic tests and exit")
    _local_config.add_no_form_arg(parser)
    parser.add_argument("--sample-landmarks", help="sample-side CSV from place_landmarks.py")
    parser.add_argument("--atlas-landmarks",
                        help="atlas-side CSV: the same anatomical points, in the same order")
    parser.add_argument("--sample-image",
                        help="the image the sample landmarks were placed on -- the raw "
                             "acquisition TIFF, or a resampled .nii.gz")
    parser.add_argument("--sample-voxel-size", nargs=3, type=float, default=None,
                        metavar=("X", "Y", "Z"),
                        help="sample voxel size in microns, (x,y,z) order -- the same "
                             "values as sample.voxel_size_um in the Registration_ants "
                             "config. REQUIRED when --sample-image is a TIFF (their "
                             "headers carry no spacing); overrides the header otherwise")
    parser.add_argument("--sample-domain-image", default=None,
                        help="the grid the INVERSE field must live on: the pipeline's "
                             "{name}_fine_{N}um.nii.gz. Omit it and one is derived -- "
                             "from --sample-image if that is already a .nii.gz, "
                             "otherwise a synthetic --target-um isotropic grid")
    parser.add_argument("--target-um", type=float, default=None,
                        help=f"voxel size of the synthesised inverse domain when there "
                             f"is no --sample-domain-image (default {DEFAULT_TARGET_UM:g}, "
                             f"matching the pipeline's fine grid)")
    parser.add_argument("--atlas-image",
                        help="the image the atlas landmarks were placed on -- the pipeline's "
                             "reoriented+cropped atlas cache, NOT a raw DevCCF file")
    parser.add_argument("--out-prefix",
                        help="output prefix; writes <prefix>init_fwd.nii.gz and "
                             "<prefix>init_inv.nii.gz")
    parser.add_argument("--fitting-levels", type=int, default=None,
                        help=f"number_of_fitting_levels for the B-spline lattice "
                             f"(default {DEFAULT_FITTING_LEVELS}; raise it if the residual "
                             f"table shows every point fitting badly)")
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftests()

    # Three ways in, same as the interactive tools: all five flags on the
    # command line, or configs/fit_initial_transform.yaml (+ --no-form), or the
    # form window pre-filled from whichever of those exist. Half a command line
    # is refused rather than silently topped up from a config the user may have
    # forgotten about.
    given = [k for k in _REQUIRED if getattr(args, k)]
    if given and len(given) < len(_REQUIRED):
        missing = [f"--{k.replace('_', '-')}" for k in _REQUIRED if not getattr(args, k)]
        parser.error(f"missing required argument(s): {', '.join(missing)}. Pass all of them, "
                     f"or none (then a config and/or the form supplies them), or --selftest.")
    if not given:
        values = _local_config.resolve_inputs(
            "fit_initial_transform", "Fit Initial Transform", _FORM_FIELDS,
            args.config, args.no_form)
        for key in _REQUIRED:
            setattr(args, key, values[key])
        # Qualifiers fall back to the config/form only where the command line
        # said nothing, so `--sample-voxel-size ... ` alone still overrides a
        # stale value remembered in .dialog_state/.
        for key in _QUALIFIERS:
            if getattr(args, key) is None and values.get(key) not in (None, ""):
                setattr(args, key, values[key])

    if args.fitting_levels is None:
        args.fitting_levels = DEFAULT_FITTING_LEVELS
    args.fitting_levels = int(args.fitting_levels)
    if args.fitting_levels < 1:
        parser.error("--fitting-levels must be >= 1")
    if args.target_um is None:
        args.target_um = DEFAULT_TARGET_UM
    if args.target_um <= 0:
        parser.error("--target-um must be > 0")

    try:
        run(args.sample_landmarks, args.atlas_landmarks, args.sample_image, args.atlas_image,
            args.out_prefix, fitting_levels=args.fitting_levels,
            sample_voxel_size=args.sample_voxel_size,
            sample_domain_image=args.sample_domain_image,
            target_um=float(args.target_um))
    except (ValueError, FileNotFoundError) as exc:
        # These are the input mistakes the checks above exist to name; a
        # traceback would bury the message that says which file to fix.
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
