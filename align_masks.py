"""Verify -- and if needed force -- that ground-truth masks live in exactly the
same voxel space as the image they will be scored against. Run this BEFORE
computing any registration metric.

WHY THIS EXISTS
---------------
Annotation tools (napari, ilastik, ITK-SNAP) each write NIfTI headers their own
way, and some drop or rewrite spacing / origin / direction on export. A mask
that is off by a single voxel looks pixel-perfect in a viewer but systematically
poisons every surface-distance metric downstream (registration_eval.py's HD95,
and the symmetric mean surface distance here). Nothing errors out -- you just
get plausible-looking wrong numbers. So: check first, and fail loudly.

TWO FUNCTIONS
-------------
1. unify_headers(): copy source-image geometry onto a batch of masks.
   - Refuses when the voxel SHAPE differs. A shape mismatch means something
     worse went wrong upstream (a bad resample, the wrong file); rewriting the
     header would only hide it. This is a hard stop, not a warning.
   - Only geometry metadata is touched. Voxel values are never modified.
   - Always writes to a new file. The original is left alone.
2. check_geometry(): itemized size / spacing / origin / direction comparison,
   reporting WHICH field differs and BY HOW MUCH -- not one opaque boolean.

HOW SURFACE DISTANCE IS COMPUTED IN PHYSICAL UNITS (not voxels)
---------------------------------------------------------------
This is the single most common way to get these metrics silently wrong, so it
is worth stating precisely. The trap is an axis-order mismatch:

  * `sitk.GetArrayFromImage(img)` returns a numpy array indexed [z, y, x].
    `np.argwhere(...)` therefore yields index triples in (z, y, x) order.
  * `img.GetSpacing()` is SimpleITK's own order, (x, y, z) -- the REVERSE.

So the voxel->micron conversion must reverse the spacing before multiplying:

    spacing_zyx = np.asarray(img.GetSpacing())[::-1]
    points_um   = np.argwhere(border) * spacing_zyx

Every coordinate that reaches the KD-tree is already in microns, so the
returned distance is in microns by construction -- "voxel count" never appears
anywhere downstream. Under ISOTROPIC spacing a transposed vector still gives
the right answer by luck, which is exactly why selftest_known_translation()
below uses deliberately ANISOTROPIC spacing and shifts along all three axes:
that is the only configuration where the bug actually shows up.

Origin and direction are ignored by the fast path above. That is legitimate
ONLY because both masks are on an identical grid (unify_headers/check_geometry
guarantee it) and this project's images are all origin=(0,0,0) + identity
direction (see registration_eval.py's module docstring). Distances are
invariant under a shared rigid transform either way, and
_surface_points_um_via_sitk() cross-checks that claim against SimpleITK's own
TransformIndexToPhysicalPoint().

USAGE
-----
    conda activate antsreg          # or gt_sam; needs only SimpleITK+numpy+scipy
    python align_masks.py --selftest
    python align_masks.py --source sample_fine_20um.nii.gz \
                          --masks gt_ventricle.nii.gz gt_cortex.nii.gz \
                          --out-dir aligned/ --report aligned/report.txt
    python align_masks.py --check-only --source ... --masks ...   # report, write nothing

Dice/HD95 for the actual evaluation live in registration_eval.py. The Dice and
surface-distance code here is a deliberate standalone minimal copy: this module
must stay importable from annotate_gt_sam.py in the `gt_sam` env, which has no
antspyx, while registration_eval.py imports registration_ants.transforms (and
therefore ants). selftest_cross_check_against_registration_eval() asserts the
two implementations agree numerically whenever antspyx IS available.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

# Tolerance for comparing float geometry metadata. NIfTI stores spacing/origin
# as float32, so a value that round-tripped through a file can differ from the
# in-memory double by ~1e-7 relative -- tighter than this produces false alarms.
DEFAULT_TOL = 1e-5


# =====================================================================================
# Geometry checking
# =====================================================================================
@dataclass
class GeometryCheck:
    """Itemized result of comparing one image's grid against a reference."""
    name: str
    ok: bool = True
    problems: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.ok = False
        self.problems.append(message)

    def __str__(self) -> str:
        if self.ok:
            return f"PASS  {self.name}"
        lines = [f"FAIL  {self.name}"]
        lines += [f"        - {p}" for p in self.problems]
        return "\n".join(lines)


def _fmt(values) -> str:
    return "(" + ", ".join(f"{v:.6g}" for v in values) + ")"


def check_geometry(reference: sitk.Image, other: sitk.Image, name: str = "mask",
                   tol: float = DEFAULT_TOL) -> GeometryCheck:
    """Compare `other`'s voxel grid against `reference`'s, field by field.

    Reports size / spacing / origin / direction separately, each with the actual
    numbers and the max absolute difference, so a failure says what to go fix
    instead of just "not aligned".
    """
    result = GeometryCheck(name=name)

    ref_size, oth_size = reference.GetSize(), other.GetSize()
    if ref_size != oth_size:
        result.fail(f"size: {oth_size} != source {ref_size} "
                    f"(voxel dimensions differ -- NOT a header problem, see unify_headers)")

    for label, ref_v, oth_v in (
        ("spacing", reference.GetSpacing(), other.GetSpacing()),
        ("origin", reference.GetOrigin(), other.GetOrigin()),
        ("direction", reference.GetDirection(), other.GetDirection()),
    ):
        ref_a, oth_a = np.asarray(ref_v, float), np.asarray(oth_v, float)
        if ref_a.shape != oth_a.shape:
            result.fail(f"{label}: length {oth_a.size} != source {ref_a.size}")
            continue
        max_diff = float(np.max(np.abs(ref_a - oth_a))) if ref_a.size else 0.0
        if max_diff > tol:
            result.fail(f"{label}: {_fmt(oth_a)} != source {_fmt(ref_a)} "
                        f"(max abs diff {max_diff:.6g} > tol {tol:g})")

    return result


# =====================================================================================
# Header unification
# =====================================================================================
def unify_headers(source_path, mask_paths, out_dir, tol: float = DEFAULT_TOL,
                  verbose: bool = True) -> list[tuple[Path, GeometryCheck]]:
    """Force every mask's spacing/origin/direction to match the source image.

    Voxel values are copied through untouched -- only geometry metadata is
    overwritten (sitk.Image.CopyInformation). Results go to NEW files under
    out_dir; inputs are never modified in place.

    Raises ValueError if any mask's voxel SHAPE differs from the source. That is
    a deliberate hard stop: a shape mismatch means a resampling/selection bug
    upstream, and silently stamping a new header on top would bury it. Nothing
    is written when this fires -- shapes for every mask are validated before the
    first file is written, so a bad mask late in the list can't leave a
    half-finished output directory behind.
    """
    source_path = Path(source_path)
    out_dir = Path(out_dir)
    source = sitk.ReadImage(str(source_path))

    mask_paths = [Path(p) for p in mask_paths]

    # Pass 1: validate every shape BEFORE writing anything.
    bad = []
    for mask_path in mask_paths:
        size = sitk.ReadImage(str(mask_path)).GetSize()
        if size != source.GetSize():
            bad.append(f"  {mask_path.name}: size {size} != source {source.GetSize()}")
    if bad:
        raise ValueError(
            "Refusing to unify headers -- voxel shape differs from the source image.\n"
            "A shape mismatch is a resampling/wrong-file problem, not a header problem;\n"
            "rewriting the header would hide it rather than fix it. Nothing was written.\n"
            f"Source: {source_path}\n" + "\n".join(bad))

    # Pass 2: write.
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for mask_path in mask_paths:
        out_path = out_dir / mask_path.name
        if out_path.resolve() == mask_path.resolve():
            raise ValueError(
                f"out_dir would overwrite the input in place: {mask_path}. "
                "Pick a different --out-dir; the originals are kept on purpose.")

        mask = sitk.ReadImage(str(mask_path))
        before = check_geometry(source, mask, name=f"{mask_path.name} (before)", tol=tol)
        mask.CopyInformation(source)      # geometry only; voxel buffer untouched
        sitk.WriteImage(mask, str(out_path))

        after = check_geometry(source, sitk.ReadImage(str(out_path)),
                               name=f"{mask_path.name} -> {out_path}", tol=tol)
        if not after.ok:
            raise RuntimeError(
                f"CopyInformation did not produce an aligned file for {mask_path}:\n{after}")

        if verbose:
            status = "already aligned" if before.ok else "geometry rewritten"
            print(f"[unify] {mask_path.name}: {status} -> {out_path}")
            for problem in before.problems:
                print(f"           was: {problem}")
        results.append((out_path, after))

    return results


# =====================================================================================
# Minimal metrics (physical units -- see module docstring)
# =====================================================================================
def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice overlap of two binary masks. NaN when both are empty."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Boundary voxels: those removed by one erosion. Same definition as
    registration_eval._surface_distances, so the two agree voxel-for-voxel."""
    m = np.asarray(mask, dtype=bool)
    return m ^ binary_erosion(m)


def _surface_points_um(mask: np.ndarray, spacing_zyx) -> np.ndarray:
    """Surface voxel indices converted to microns.

    `spacing_zyx` MUST already be in numpy array-axis order (z, y, x) -- i.e.
    `np.asarray(img.GetSpacing())[::-1]`, since GetSpacing() is (x, y, z). Get
    this backwards and every distance below is wrong by the ratio between two
    axes' spacings, with no error raised. See the module docstring.
    """
    spacing_zyx = np.asarray(spacing_zyx, dtype=float)
    idx = np.argwhere(surface_voxels(mask))
    if idx.shape[1] != spacing_zyx.size:
        raise ValueError(f"spacing has {spacing_zyx.size} entries but the mask is "
                         f"{idx.shape[1]}-D")
    return idx * spacing_zyx


def symmetric_mean_surface_distance(a: np.ndarray, b: np.ndarray, spacing_zyx) -> float:
    """Symmetric mean surface distance in MICRONS: the average of the
    bidirectional nearest-boundary-point Euclidean distances.

    Returns NaN if either mask is empty (no surface to measure against).
    """
    pa = _surface_points_um(a, spacing_zyx)
    pb = _surface_points_um(b, spacing_zyx)
    if len(pa) == 0 or len(pb) == 0:
        return float("nan")
    d_ab = cKDTree(pb).query(pa)[0]
    d_ba = cKDTree(pa).query(pb)[0]
    return float(np.concatenate([d_ab, d_ba]).mean())


def _surface_points_um_via_sitk(mask: np.ndarray, img: sitk.Image) -> np.ndarray:
    """Independent oracle for _surface_points_um: ask SimpleITK itself where
    each surface voxel physically sits, via TransformIndexToPhysicalPoint.

    Slower (a Python call per voxel) and only used in the selftest, but it goes
    through SimpleITK's own geometry engine rather than any assumption of mine
    about axis order -- so if `spacing[::-1]` were wrong, the two would disagree.
    It also folds in origin and direction, which the fast path drops; distances
    are invariant under that shared rigid transform, and the selftest asserts
    exactly that on an image with a nonzero origin.

    Note the returned points are in physical (x, y, z) order while
    _surface_points_um returns (z, y, x). Pairwise distances are invariant to a
    consistent axis permutation, so only distances -- never raw coordinates --
    may be compared between the two.
    """
    idx_zyx = np.argwhere(surface_voxels(mask))
    return np.array(
        [img.TransformIndexToPhysicalPoint((int(x), int(y), int(z))) for z, y, x in idx_zyx],
        dtype=float,
    ).reshape(len(idx_zyx), 3)


def _smsd_from_point_sets(pa: np.ndarray, pb: np.ndarray) -> float:
    if len(pa) == 0 or len(pb) == 0:
        return float("nan")
    d_ab = cKDTree(pb).query(pa)[0]
    d_ba = cKDTree(pa).query(pb)[0]
    return float(np.concatenate([d_ab, d_ba]).mean())


def compare_masks(image_a: sitk.Image, image_b: sitk.Image, tol: float = DEFAULT_TOL) -> dict:
    """Dice + symmetric mean surface distance (um) between two masks that are
    already on the same grid. Raises if they are not -- comparing masks across
    grids is exactly the mistake this module exists to prevent."""
    check = check_geometry(image_a, image_b, name="compare_masks operand", tol=tol)
    if not check.ok:
        raise ValueError(f"Cannot compare masks on different grids:\n{check}")

    a = sitk.GetArrayFromImage(image_a) > 0
    b = sitk.GetArrayFromImage(image_b) > 0
    spacing_zyx = np.asarray(image_a.GetSpacing(), dtype=float)[::-1]
    return {
        "dice": dice(a, b),
        "surface_distance_um": symmetric_mean_surface_distance(a, b, spacing_zyx),
    }


# =====================================================================================
# Selftests -- synthetic data only, no real files needed
# =====================================================================================
def _make_image(arr: np.ndarray, spacing_xyz, origin_xyz=(0.0, 0.0, 0.0)) -> sitk.Image:
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.SetSpacing(tuple(float(s) for s in spacing_xyz))
    img.SetOrigin(tuple(float(o) for o in origin_xyz))
    return img


def _plane_mask(shape_zyx, axis: int, index: int) -> np.ndarray:
    """A single-voxel-thick plane normal to `axis`.

    Chosen deliberately over a cube: for a one-voxel-thick plane shifted along
    its own normal by k voxels, the symmetric mean surface distance is EXACTLY
    k * spacing[axis]. Every surface point's nearest counterpart is the point
    directly across from it (all of the other mask's points share one coordinate
    on that axis, so any lateral offset only adds to the distance), and a
    one-voxel-thick plane erodes to nothing, making the whole plane its own
    surface. A shifted cube gives no such closed form -- its trailing face lands
    inside the other cube, so the mean is some shape-dependent number rather
    than the known translation, which would turn an exact assertion into a fuzzy
    one exactly where precision matters most.
    """
    mask = np.zeros(shape_zyx, dtype=np.uint8)
    idx = [slice(None)] * 3
    idx[axis] = index
    mask[tuple(idx)] = 1
    return mask


def selftest_identity() -> None:
    print("1. identity: a mask against itself -> dice 1, surface distance 0")
    shape = (24, 30, 36)
    spacing_xyz = (10.0, 25.0, 40.0)
    spacing_zyx = np.asarray(spacing_xyz)[::-1]

    for axis in (0, 1, 2):
        mask = _plane_mask(shape, axis, index=shape[axis] // 2)
        assert dice(mask, mask) == 1.0, f"axis {axis}: dice(m, m) != 1"
        d = symmetric_mean_surface_distance(mask, mask, spacing_zyx)
        assert d == 0.0, f"axis {axis}: surface distance to self is {d}, expected 0"

    # A blob too, so this isn't only true for the degenerate plane case.
    zz, yy, xx = np.indices(shape)
    blob = (((zz - 12) / 8.0) ** 2 + ((yy - 15) / 9.0) ** 2 + ((xx - 18) / 11.0) ** 2 < 1)
    blob = blob.astype(np.uint8)
    assert dice(blob, blob) == 1.0
    assert symmetric_mean_surface_distance(blob, blob, spacing_zyx) == 0.0
    print("   ok")


def selftest_known_translation() -> None:
    print("2. known translation: shift by a known physical distance, in microns")
    shape = (24, 30, 36)          # deliberately NOT cubic
    isotropic = (20.0, 20.0, 20.0)

    # 2a. isotropic -- the easy case, where an axis-order bug would NOT show up.
    spacing_zyx = np.asarray(isotropic)[::-1]
    a = _plane_mask(shape, axis=2, index=10)
    b = _plane_mask(shape, axis=2, index=15)
    d = symmetric_mean_surface_distance(a, b, spacing_zyx)
    assert abs(d - 5 * 20.0) < 1e-9, f"isotropic: got {d}, expected 100.0"
    print(f"   isotropic 20um, shift 5 voxels along x -> {d:.6f} um (expected 100.0)")

    # 2b. anisotropic, all three axes. This is the test that actually catches
    #     "forgot to convert voxels to physical units" and, critically, "used
    #     GetSpacing() without reversing it into numpy axis order".
    spacing_xyz = (10.0, 25.0, 40.0)      # x=10, y=25, z=40 um
    spacing_zyx = np.asarray(spacing_xyz)[::-1]
    # (numpy axis, name, spacing along it, shift in voxels)
    cases = [
        (2, "x", spacing_xyz[0], 10),     # 10 voxels * 10um = 100um  <- the user's example
        (1, "y", spacing_xyz[1], 4),      #  4 voxels * 25um = 100um
        (0, "z", spacing_xyz[2], 3),      #  3 voxels * 40um = 120um
    ]
    for axis, name, spacing_along_axis, shift in cases:
        start = shape[axis] // 3
        a = _plane_mask(shape, axis, start)
        b = _plane_mask(shape, axis, start + shift)
        expected = shift * spacing_along_axis
        d = symmetric_mean_surface_distance(a, b, spacing_zyx)
        assert abs(d - expected) < 1e-9, (
            f"anisotropic {name}: got {d} um, expected {expected} um. "
            f"If this is off by the ratio of two axes' spacings, GetSpacing() "
            f"was used without reversing it into numpy (z,y,x) order.")
        # Voxel-count answer, i.e. the bug we are guarding against.
        assert abs(d - shift) > 1e-6, f"{name}: distance equals the voxel count, not microns"
        print(f"   anisotropic {spacing_xyz} um, shift {shift} voxels along {name} "
              f"-> {d:.6f} um (expected {expected})")

        # Cross-check against SimpleITK's own geometry engine, on an image with
        # a nonzero origin so origin handling is exercised too.
        img = _make_image(a, spacing_xyz, origin_xyz=(7.0, -3.0, 11.0))
        d_sitk = _smsd_from_point_sets(_surface_points_um_via_sitk(a, img),
                                       _surface_points_um_via_sitk(b, img))
        assert abs(d_sitk - expected) < 1e-6, (
            f"{name}: SimpleITK oracle got {d_sitk}, expected {expected}")
        assert abs(d_sitk - d) < 1e-6, (
            f"{name}: fast path {d} disagrees with SimpleITK oracle {d_sitk}")
    print("   ok (all three axes, cross-checked against TransformIndexToPhysicalPoint)")


def selftest_shape_mismatch_is_an_error() -> None:
    print("3. shape mismatch: unify_headers must refuse, not silently overwrite")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = _make_image(np.zeros((24, 30, 36), dtype=np.uint8), (10.0, 25.0, 40.0))
        sitk.WriteImage(source, str(tmp / "source.nii.gz"))

        good = _make_image(_plane_mask((24, 30, 36), 2, 10), (1.0, 1.0, 1.0))
        sitk.WriteImage(good, str(tmp / "good.nii.gz"))
        wrong = _make_image(_plane_mask((24, 30, 35), 2, 10), (1.0, 1.0, 1.0))
        sitk.WriteImage(wrong, str(tmp / "wrong_shape.nii.gz"))

        out_dir = tmp / "out"
        raised = False
        try:
            # `good` comes first on purpose: nothing at all may be written, not
            # even the masks that would individually have been fine.
            unify_headers(tmp / "source.nii.gz",
                          [tmp / "good.nii.gz", tmp / "wrong_shape.nii.gz"],
                          out_dir, verbose=False)
        except ValueError as exc:
            raised = True
            assert "wrong_shape.nii.gz" in str(exc), f"error names the wrong file: {exc}"
        assert raised, "unify_headers accepted a shape mismatch instead of raising"
        assert not out_dir.exists() or not list(out_dir.iterdir()), \
            "unify_headers wrote output despite the shape mismatch"

        # Refuses to overwrite the inputs in place, too.
        raised = False
        try:
            unify_headers(tmp / "source.nii.gz", [tmp / "good.nii.gz"], tmp, verbose=False)
        except ValueError as exc:
            raised = True
            assert "in place" in str(exc)
        assert raised, "unify_headers overwrote its input in place"
    print("   ok")


def selftest_unify_and_report() -> None:
    print("4. unify_headers rewrites geometry, keeps voxels, and reports each field")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shape = (24, 30, 36)
        spacing_xyz = (10.0, 25.0, 40.0)
        origin_xyz = (5.0, -2.0, 8.0)

        source = _make_image(np.zeros(shape, dtype=np.uint8), spacing_xyz, origin_xyz)
        sitk.WriteImage(source, str(tmp / "source.nii.gz"))

        # A mask exported by a tool that lost the geometry: unit spacing, zero
        # origin. Same voxel content, wrong header.
        voxels = _plane_mask(shape, axis=2, index=10)
        broken = _make_image(voxels, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
        sitk.WriteImage(broken, str(tmp / "gt_region.nii.gz"))

        before = check_geometry(source, sitk.ReadImage(str(tmp / "gt_region.nii.gz")))
        assert not before.ok
        joined = " ".join(before.problems)
        assert "spacing" in joined and "origin" in joined, f"report missed a field: {before}"
        assert "size" not in joined, "size flagged even though shapes match"

        out_dir = tmp / "aligned"
        results = unify_headers(tmp / "source.nii.gz", [tmp / "gt_region.nii.gz"],
                                out_dir, verbose=False)
        out_path, after = results[0]
        assert after.ok, f"still misaligned after unify_headers:\n{after}"

        fixed = sitk.ReadImage(str(out_path))
        assert np.array_equal(sitk.GetArrayFromImage(fixed), voxels), \
            "unify_headers changed voxel values -- it must only touch geometry"
        # Original untouched.
        original = sitk.ReadImage(str(tmp / "gt_region.nii.gz"))
        assert np.allclose(original.GetSpacing(), (1.0, 1.0, 1.0)), \
            "unify_headers modified the input file in place"

        # And now that geometry agrees, the metrics run.
        shifted = _make_image(_plane_mask(shape, axis=2, index=20), spacing_xyz, origin_xyz)
        metrics = compare_masks(fixed, shifted)
        assert abs(metrics["surface_distance_um"] - 10 * spacing_xyz[0]) < 1e-9
        assert metrics["dice"] == 0.0
        # compare_masks must refuse mismatched grids rather than return a number.
        raised = False
        try:
            compare_masks(fixed, broken)
        except ValueError:
            raised = True
        assert raised, "compare_masks compared masks on different grids"
    print("   ok")


def selftest_cross_check_against_registration_eval() -> None:
    """Assert this module's metrics match registration_eval.py's, so the two
    can't drift. Skipped when antspyx is missing (registration_eval imports
    registration_ants.transforms, which needs ants) -- e.g. in the gt_sam env."""
    print("5. cross-check against registration_eval.py (skipped without antspyx)")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import registration_eval as ev
    except Exception as exc:
        print(f"   skipped: {type(exc).__name__}: {exc}")
        return

    shape = (24, 30, 36)
    spacing_zyx = np.asarray((10.0, 25.0, 40.0))[::-1]
    a = _plane_mask(shape, axis=2, index=10)
    b = _plane_mask(shape, axis=2, index=20)

    assert ev.dice(a, b) == dice(a, b), "Dice differs from registration_eval's"
    assert ev.dice(a, a) == dice(a, a) == 1.0

    mine = symmetric_mean_surface_distance(a, b, spacing_zyx)
    theirs = float(ev._surface_distances(a.astype(bool), b.astype(bool), tuple(spacing_zyx)).mean())
    assert abs(mine - theirs) < 1e-9, (
        f"surface distance differs from registration_eval's: {mine} vs {theirs}")
    print(f"   ok (dice and surface distance identical; {mine:.6f} um)")


def run_selftests() -> int:
    print("=== align_masks.py selftests (synthetic data only) ===")
    selftest_identity()
    selftest_known_translation()
    selftest_shape_mismatch_is_an_error()
    selftest_unify_and_report()
    selftest_cross_check_against_registration_eval()
    print("=== all selftests passed ===")
    return 0


# =====================================================================================
# CLI
# =====================================================================================
def _write_report(lines: list[str], report_path) -> None:
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(text)
        print(f"[report] written to {report_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Check (and optionally force) that masks share the source image's voxel space.")
    parser.add_argument("--selftest", action="store_true",
                        help="run the built-in synthetic tests and exit")
    parser.add_argument("--source", help="source image whose geometry is authoritative")
    parser.add_argument("--masks", nargs="+", default=[], help="one or more mask files")
    parser.add_argument("--out-dir", help="where aligned copies are written (required unless --check-only)")
    parser.add_argument("--check-only", action="store_true",
                        help="report alignment without writing anything")
    parser.add_argument("--report", help="also write the report to this file")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL,
                        help=f"tolerance for float geometry comparison (default {DEFAULT_TOL:g})")
    args = parser.parse_args(argv)

    if args.selftest:
        return run_selftests()

    if not args.source or not args.masks:
        parser.error("--source and --masks are required (or use --selftest)")
    if not args.check_only and not args.out_dir:
        parser.error("--out-dir is required unless --check-only is given")

    source = sitk.ReadImage(str(args.source))
    lines = [f"source: {args.source}",
             f"        size={source.GetSize()} spacing={_fmt(source.GetSpacing())} "
             f"origin={_fmt(source.GetOrigin())}",
             ""]

    if args.check_only:
        checks = [check_geometry(source, sitk.ReadImage(str(m)), name=str(m), tol=args.tol)
                  for m in args.masks]
        lines += [str(c) for c in checks]
        n_failed = sum(not c.ok for c in checks)
        lines += ["", f"{len(checks) - n_failed}/{len(checks)} masks already aligned."]
        _write_report(lines, args.report)
        return 1 if n_failed else 0

    try:
        results = unify_headers(args.source, args.masks, args.out_dir, tol=args.tol, verbose=False)
    except ValueError as exc:
        lines += ["ERROR: " + str(exc)]
        _write_report(lines, args.report)
        return 2

    lines += [str(check) for _, check in results]
    lines += ["", f"{len(results)}/{len(args.masks)} masks written aligned to {args.out_dir}."]
    _write_report(lines, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
