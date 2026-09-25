"""
Registration evaluation metrics for P5 whole-brain LSFM -> synthetic P5 atlas.

Computes, per sample, and separately for each coarse region + whole-brain
outline + per group (TSC / normal):
    M1  TRE   - target registration error from hand-placed landmarks (um)
    M2  Dice  - overlap of warped-atlas structure vs sample structure
    M3  HD95  - 95th-percentile surface distance (um)
    M5  neg-Jacobian fraction - folding / invertibility check (%)
    M6  inverse-consistency error (um)
    M7  region Jacobian determinant stats - volume compression check per
        region (the TSC-cortex-expansion confound this eval exists to catch)

Wired to the real ants pipeline in ../Registration_ants/src/registration_ants/ (see PROGRESS_LOG.md
for its design history). Two key simplifications vs. a generic skeleton:

  * No anisotropic-voxel handling needed here. This pipeline resamples every
    sample to an ISOTROPIC grid before registering (io_utils.resample_to_isotropic,
    matched to the atlas's own isotropic resolution) -- so every grid a metric
    here touches (sample_fine, labels_in_sample, the atlas) is isotropic, and
    physical distance = voxel distance * one scalar per grid. The raw LSFM
    data's anisotropy (~0.65x0.65x8um) never reaches this file.
  * M5/M7 (Jacobian) are computed in ATLAS space, not sample space. ANTs'
    forward warp field (<prefix>1Warp.nii.gz) is defined on the FIXED/atlas
    grid (see ants.create_jacobian_determinant_image's own docstring example,
    which passes the fixed image as domain_image) -- so these metrics use the
    atlas's own region masks (same ROI for every sample) rather than a
    per-sample sample-space mask. Standard practice for deformation-based
    morphometry (compare Jacobians in a common template space).

Ground truth for M2/M3 comes from a semi-manual workflow already built in
this repo (not new tooling): run the ants pipeline once, then hand-correct
whichever regions came out wrong --
  tools/edit_sample_labels.py ("Save This Region")
                                        -> <name>_<region_slug>_corrected_mask.nii.gz,
                                           one per region, wired up via dice_region_masks
  paint_mask.py (KIND="mask")   -> sample_brain_mask_corrected.nii.gz
Landmarks for M1 are placed in napari by hand: open the sample and the atlas
template, add a Points layer to each, click the SAME anatomical points in the
SAME order in both, and save each layer to CSV (row i in one file and row i in
the other are one landmark -- see shared/landmark_io.py).

------------------------------------------------------------------------------------
CRITICAL AXIS-ORDER NOTE -- this file mixes TWO conventions on purpose, matching
how the rest of this codebase already reads each kind of file:
  * SAMPLE-space label volumes (labels_in_sample.nii.gz, the hand-corrected
    labels/brain-mask files) are read via SimpleITK here, giving array axis
    order (z,y,x) -- the same convention tools/edit_sample_labels.py and
    paint_mask.py already read/write with. z_axis=0 always
    for these arrays.
  * ATLAS-space arrays (the annotation used for Jacobian/region masks) come
    from an ants image (atlas_utils.load_custom_atlas), giving array axis
    order (x,y,z) -- the same convention the rest of ../Registration_ants/src/registration_ants/
    uses, and the convention that lines up index-for-index with
    ants.create_jacobian_determinant_image's output (same grid, same library).
  * Landmark points loaded via load_points() come from napari (SimpleITK-style
    (z,y,x) voxel order) and are reversed to (x,y,z) before being turned into
    physical microns -- matching cell_points.py's established
    "physical = voxel_index(x,y,z) * spacing(x,y,z)" convention (every ants
    image in this codebase has origin=(0,0,0)/identity direction).
Mixing these up silently produces meaningless numbers with no error -- see
cell_points.py / io_utils.py / the four existing napari tools' docstrings for
the same warning repeated at each place this matters.
------------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from registration_ants import atlas_utils, transforms

from shared import landmark_io  # the landmark CSV format


# =====================================================================================
# CONFIG
# =====================================================================================
@dataclass
class Config:
    # Coarse (high-level) regions to score Dice/HD95 on, matched by exact
    # name against the atlas ontology via atlas_utils.region_mask_by_exact_name
    # (includes all descendants). Add/remove freely.
    dice_regions: list[str] = field(
        default_factory=lambda: [
            "Isocortex", "Caudoputamen", "Hippocampal formation",
            "Brain stem", "Cerebellum", "Cerebellar cortex",
        ]
    )

    # Which z-slices (sample-space axis 0) are annotated, for sparse-slice
    # Dice/HD95. Explicit override applied to EVERY region regardless of any
    # per-region hint below; None (the default) lets use_annotation_hints
    # decide per region instead. Set this only if you want one fixed set of
    # slices for all regions no matter what tools/edit_sample_labels.py's
    # sidecar files say.
    annotated_slices: list[int] | None = None

    # When annotated_slices above is None, use each region's own
    # <mask>.annotated_slices.json sidecar (written by
    # tools/edit_sample_labels.py's "Save This Region") to restrict that
    # region's Dice/HD95 to just the z-planes actually hand-drawn, instead of
    # the full volume (which includes signed-distance-interpolated guesses
    # in between). Set False to ignore any hints and always use the full
    # volume.
    use_annotation_hints: bool = True

    # sample_id -> group name (e.g. "Control"/"Experimental"), loaded from
    # configs/eval_config.yaml's groups_manifest.
    groups: dict[str, str] = field(default_factory=dict)


def _load_groups_from_manifest(groups_manifest_path):
    """Flatten stats_config.yaml's groups.<a|b>.{name,samples} into a plain
    {sample_id: group_name} dict -- reusing the existing TSC/normal manifest
    instead of duplicating it here."""
    with open(groups_manifest_path) as f:
        manifest = yaml.safe_load(f)
    groups = {}
    for group in manifest["groups"].values():
        for sample_id in group["samples"]:
            groups[sample_id] = group["name"]
    return groups


def load_eval_config(path):
    """Parse configs/eval_config.yaml into (Config, per_sample_paths, atlas_cfg).

    per_sample_paths[sample_id] carries both the standard pipeline output
    paths (derived from output_dir/name using the exact naming pipeline.py
    writes) and the eval-only annotation paths given explicitly in the YAML
    (landmarks, hand-corrected labels/mask) -- the latter are optional, so a
    sample can be listed (and partially evaluated) before every annotation
    exists.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    groups = _load_groups_from_manifest(raw["groups_manifest"]) if raw.get("groups_manifest") else {}
    cfg = Config(
        dice_regions=raw.get("dice_regions", Config().dice_regions),
        annotated_slices=raw.get("annotated_slices"),
        use_annotation_hints=raw.get("use_annotation_hints", Config().use_annotation_hints),
        groups=groups,
    )

    per_sample_paths = {}
    for sample_id, s in raw.get("samples", {}).items():
        output_dir = Path(s["output_dir"])
        name = s["name"]
        sample_res = float(s["sample_resolution_um"])
        labels_in_sample = output_dir / f"{name}_labels_in_sample.nii.gz"
        transforms_prefix = str(output_dir / "transforms" / f"{name}_")

        if not labels_in_sample.exists():
            raise FileNotFoundError(
                f"sample '{sample_id}': expected pipeline output not found: {labels_in_sample} "
                "-- has this sample been run through registration_ants.pipeline yet?"
            )

        if "labels_in_sample_corrected" in s:
            raise ValueError(
                f"sample '{sample_id}': 'labels_in_sample_corrected' is no longer supported -- "
                "use 'dice_region_masks: {<Region Name>: <path>}' instead "
                "(see configs/eval_config.example.yaml)."
            )

        per_sample_paths[sample_id] = {
            "sample_resolution_um": sample_res,
            "atlas_resolution_um": float(s["atlas_resolution_um"]),
            "labels_in_sample": labels_in_sample,
            "transforms_prefix": transforms_prefix,
            "sample_landmarks_csv": s.get("sample_landmarks_csv"),
            "atlas_landmarks_csv": s.get("atlas_landmarks_csv"),
            "cortex_landmark_idx": s.get("cortex_landmark_idx") or None,
            "dice_region_masks": {name: Path(p) for name, p in (s.get("dice_region_masks") or {}).items()},
            "sample_brain_mask_corrected": s.get("sample_brain_mask_corrected"),
        }

    atlas_cfg = raw["atlas"]
    return cfg, per_sample_paths, atlas_cfg


# =====================================================================================
# M1 -- Landmark TRE
# =====================================================================================
def load_points(csv_path):
    """Parse a napari Points-layer CSV export (index,axis-0,axis-1,axis-2)
    into an (N,3) array of voxel coordinates in (x,y,z) order.

    napari/SimpleITK's array order for these tools is (z,y,x) (axis-0 = the
    actual imaging/atlas planes) -- reversed here to (x,y,z) to match this
    codebase's established physical-space convention (see module docstring).
    Parsing and validation live in shared/landmark_io.py so this tool and
    napari's own writer cannot drift apart on the format.
    """
    return landmark_io.to_xyz(landmark_io.read_landmark_csv(csv_path))


def apply_transform_to_points(sample_pts_xyz_um, transforms_prefix):
    """Warp sample-space landmark points (physical microns, (x,y,z) columns)
    into atlas physical space.

    Reuses registration_ants.transforms.transform_cell_points, which already
    handles ANTs' point-vs-image transform-direction convention correctly --
    it's the same function cell_points.assign_cell_regions uses for cell
    centroids, already validated end-to-end there. transforms.load_saved_transforms
    reconstructs the fwd/inv transform lists from the files a previous
    registration run wrote to disk (this script never holds the live `reg`
    dict from ants.registration() itself).
    """
    reg = transforms.load_saved_transforms(transforms_prefix)
    df = pd.DataFrame({
        "x": sample_pts_xyz_um[:, 0], "y": sample_pts_xyz_um[:, 1], "z": sample_pts_xyz_um[:, 2],
    })
    warped = transforms.transform_cell_points(df, reg, direction="sample_to_atlas")
    return warped[["x", "y", "z"]].to_numpy()


def compute_tre(warped_sample_pts_um: np.ndarray, atlas_pts_um: np.ndarray) -> np.ndarray:
    """M1: per-landmark target registration error in um. Returns (N,) distances.

    Both point sets are in ATLAS physical microns here (warped_sample_pts_um
    came out of apply_transform_to_points already in that space) -- so this
    is a plain isotropic Euclidean distance, no per-axis spacing needed.
    """
    assert warped_sample_pts_um.shape == atlas_pts_um.shape
    return np.linalg.norm(warped_sample_pts_um - atlas_pts_um, axis=1)


def summarize_tre(tre_um: np.ndarray, cortex_idx: list[int] | None = None) -> dict:
    """Report mean +/- std overall and (optionally) for the cortex subset."""
    out = {"tre_mean_um": float(tre_um.mean()),
           "tre_std_um": float(tre_um.std()),
           "tre_median_um": float(np.median(tre_um)),
           "n": int(tre_um.size)}
    if cortex_idx:
        c = tre_um[cortex_idx]
        out.update({"cortex_tre_mean_um": float(c.mean()),
                    "cortex_tre_std_um": float(c.std()),
                    "cortex_tre_median_um": float(np.median(c)),
                    "cortex_n": int(c.size)})
    return out


def annotator_reproducibility(pts_a_um: np.ndarray, pts_b_um: np.ndarray) -> float:
    """QC: RMS distance between two placements of the SAME landmark set (um).
    This is your localization noise floor. TRE cannot be better than this, so run it
    once (re-place points on another day, or a labmate re-places) before trusting
    small between-group TRE differences."""
    d = np.linalg.norm(pts_a_um - pts_b_um, axis=1)
    return float(np.sqrt((d ** 2).mean()))


# =====================================================================================
# M2 / M3 -- Dice and HD95 (overlap of warped-atlas structure vs sample structure)
# =====================================================================================
def load_sample_space_image(path):
    """Read a SAMPLE-space nii.gz via SimpleITK, keeping the full sitk.Image
    (size + spacing), not just its array -- needed wherever a mask might come
    from a DIFFERENT grid than the one it's being compared against (see
    resample_mask_to_reference): ground truth hand-drawn against one
    registration's output (one atlas, one fine_target_um) still needs scoring
    against another registration of the same sample (a different atlas, a
    different fine_target_um, or an entirely different tool -- ClearMap /
    VoxelMorph / brainreg / mBrainAligner -- with its own native resolution).
    """
    return sitk.ReadImage(str(path))


def resample_mask_to_reference(mask_img: sitk.Image, reference_img: sitk.Image) -> sitk.Image:
    """Nearest-neighbor-resample a binary/label mask onto another image's
    exact grid (size + spacing), so ground truth drawn against one
    registration's output can still be scored against a different
    registration of the same sample without requiring every method/atlas to
    share one fixed output resolution -- comparing labels_in_sample volumes
    across methods is exactly this repo's motivating use case (see
    PROGRESS_LOG.md). Nearest-neighbor keeps the result binary/integer
    (unlike linear, which would blur label ids at the boundary).

    A no-op (returns mask_img as-is) when the grids already match (same
    size+spacing) -- resampling an already-matching grid onto itself would
    still risk nudging border voxels by floating-point noise, for zero
    benefit.

    Both images must already follow this codebase's zero-origin/
    identity-direction convention (see module docstring) for the result to be
    physically meaningful -- a mask exported by a tool that writes its own
    arbitrary origin/direction needs converting to that convention first (the
    same conversion this project already had to do for DevCCF's own nii.gz
    files).
    """
    if (mask_img.GetSize() == reference_img.GetSize()
            and np.allclose(mask_img.GetSpacing(), reference_img.GetSpacing(), rtol=1e-6)):
        return mask_img
    return sitk.Resample(mask_img, reference_img, sitk.Transform(),
                          sitk.sitkNearestNeighbor, 0, mask_img.GetPixelID())


def remap_annotated_slices(slices, mask_img: sitk.Image, reference_img: sitk.Image):
    """Convert a hand-drawn z-slice index list (recorded against mask_img's
    own grid by load_region_annotation_hint) into the equivalent indices on
    reference_img's grid, for when resample_mask_to_reference actually had to
    resample (different z-spacing) -- otherwise the hint's indices would
    silently point at the wrong planes once the mask has been moved onto a
    different grid. SimpleITK's index order (GetSize()/GetSpacing()) has z at
    position 2, matching this module's SAMPLE-space numpy arrays' axis 0
    after GetArrayFromImage's index reversal (see module docstring).
    """
    if slices is None:
        return None
    if (mask_img.GetSize() == reference_img.GetSize()
            and np.allclose(mask_img.GetSpacing(), reference_img.GetSpacing(), rtol=1e-6)):
        return slices
    old_z_sp = mask_img.GetSpacing()[2]
    new_z_sp = reference_img.GetSpacing()[2]
    new_z_size = reference_img.GetSize()[2]
    remapped = sorted({int(round(z * old_z_sp / new_z_sp)) for z in slices})
    return [z for z in remapped if 0 <= z < new_z_size]


def load_region_mask(labels_arr, region_name, structures):
    """Boolean mask for one coarse region (+ all its ontology descendants)
    from an already-loaded SAMPLE-space label array ((z,y,x) order -- see
    load_sample_space_image + sitk.GetArrayFromImage), resolved by exact name
    (atlas_utils.region_mask_by_exact_name) -- used for the WARPED-ATLAS side
    of a Dice comparison (labels_in_sample, the registration output), which
    is always one combined multi-label volume. The ground-truth side uses
    load_binary_mask instead (see tools/edit_sample_labels.py's per-region
    "Save This Region" output).
    """
    return atlas_utils.region_mask_by_exact_name(labels_arr, structures, region_name)


def load_binary_mask(path):
    """Load an already-binary 0/1 mask (e.g. a hand-corrected brain outline
    from paint_mask.py's `mask` kind, or one region's corrected
    mask from tools/edit_sample_labels.py) as a bool array, SAMPLE-space
    (z,y,x) order."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0


def load_region_annotation_hint(mask_path):
    """Read the <mask_path>.annotated_slices.json sidecar
    tools/edit_sample_labels.py's "Save This Region" writes alongside a
    region's corrected mask -- {"hand_drawn_slices": [...], ...}. Returns the
    hand-drawn z-index list, or None if no sidecar exists (e.g. the mask was
    fully hand-corrected some other way, or predates this feature)."""
    mask_path = Path(mask_path)
    name = mask_path.name
    if name.endswith(".nii.gz"):
        name = name[: -len(".nii.gz")]
    sidecar_path = mask_path.parent / f"{name}.annotated_slices.json"
    if not sidecar_path.exists():
        return None
    return json.loads(sidecar_path.read_text())["hand_drawn_slices"]


def load_brain_mask_from_labels(labels_path):
    """Whole-brain outline implied by a SAMPLE-space label volume: any
    nonzero (non-background) label id counts as brain -- same 'background'
    convention as cell_points.py."""
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(labels_path)))
    return arr > 0


def restrict_to_slices(mask: np.ndarray, slices: list[int] | None, z_axis: int = 0) -> np.ndarray:
    """For sparse-slice evaluation: zero out everything except annotated z-slices,
    so Dice/HD95 are computed only where you actually annotated."""
    if slices is None:
        return mask
    keep = np.zeros_like(mask, dtype=bool)
    idx = [slice(None)] * mask.ndim
    for z in slices:
        idx[z_axis] = z
        keep[tuple(idx)] = mask[tuple(idx)]
    return keep


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """M2: Dice overlap of two binary masks."""
    a = a.astype(bool); b = b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def _surface_distances(a: np.ndarray, b: np.ndarray,
                       spacing_um: tuple[float, float, float]) -> np.ndarray:
    """Symmetric set of boundary-to-boundary nearest distances (um).
    Used by HD95. Boundary = voxels removed by one erosion."""
    a = a.astype(bool); b = b.astype(bool)
    a_border = a ^ binary_erosion(a)
    b_border = b ^ binary_erosion(b)
    sp = np.asarray(spacing_um, dtype=float)
    a_pts = np.argwhere(a_border) * sp
    b_pts = np.argwhere(b_border) * sp
    if len(a_pts) == 0 or len(b_pts) == 0:
        return np.array([np.nan])
    d_ab = cKDTree(b_pts).query(a_pts)[0]
    d_ba = cKDTree(a_pts).query(b_pts)[0]
    return np.concatenate([d_ab, d_ba])


def hd95(a: np.ndarray, b: np.ndarray, spacing_um: tuple[float, float, float]) -> float:
    """M3: 95th-percentile symmetric surface distance (um). Robust to a few outliers
    that would blow up the raw Hausdorff; sensitive to thin-cortex boundary errors."""
    d = _surface_distances(a, b, spacing_um)
    return float(np.percentile(d, 95))


def evaluate_structure(warped_atlas_mask: np.ndarray, sample_mask: np.ndarray,
                       sample_resolution_um: float, annotated_slices: list[int] | None = None) -> dict:
    """Run M2 + M3 for one structure (e.g. whole-brain outline, or one coarse
    region). Both masks are SAMPLE-space (z,y,x) arrays (see
    load_region_mask/load_brain_mask_from_labels/load_binary_mask), so
    z_axis=0 always; sample grid is isotropic so spacing_um is just
    sample_resolution_um repeated 3x (axis order doesn't matter for an
    isotropic spacing)."""
    a = restrict_to_slices(warped_atlas_mask, annotated_slices, z_axis=0)
    b = restrict_to_slices(sample_mask, annotated_slices, z_axis=0)
    spacing_um = (sample_resolution_um,) * 3
    return {"dice": dice(a, b), "hd95_um": hd95(a, b, spacing_um)}


# =====================================================================================
# M5 / M7 -- Jacobian-based metrics (ATLAS space)    M6 -- inverse consistency
# =====================================================================================
def load_jacobian(warp_path, atlas_template):
    """M5/M7 input: ANTs Jacobian determinant image, on the ATLAS grid.

    The forward warp field (<prefix>1Warp.nii.gz) is defined on the
    fixed/atlas domain (ants.create_jacobian_determinant_image's own
    docstring example passes the FIXED image as domain_image) -- so
    atlas_template must be the SAME atlas image object/grid the original
    registration used as `fixed=` (atlas_utils.load_custom_atlas reloads it
    the same way register.py originally loaded it). Returns an (x,y,z)-order
    numpy array (ants convention), matching atlas_template.numpy()/
    atlas_annotation.numpy()'s order -- do not mix this with the SAMPLE-space
    (z,y,x) arrays above.
    """
    import ants
    return ants.create_jacobian_determinant_image(atlas_template, str(warp_path), do_log=False).numpy()


def neg_jacobian_fraction(jac: np.ndarray, atlas_brain_mask: np.ndarray) -> float:
    """M5: fraction of voxels with det(J) <= 0 inside the atlas brain (folding). Should be ~0."""
    inside = jac[atlas_brain_mask.astype(bool)]
    return float((inside <= 0).mean())


def region_jacobian_stats(jac: np.ndarray, atlas_region_mask: np.ndarray, region_key: str) -> dict:
    """M7: det(J) distribution inside one atlas region (generalized to any
    region in cfg.dice_regions, not just cortex). det<1 = the registration
    COMPRESSED this tissue to fit the sample; det>1 = expanded. Compare TSC
    vs normal for a given region: if TSC's det is systematically smaller,
    the registration is squashing that region's real anatomical expansion
    (e.g. undercounting cortex cell density) -- the confound this eval
    exists to catch. Computed against the atlas's own (sample-independent)
    region mask -- same ROI for every sample, only the Jacobian values vary."""
    vals = jac[atlas_region_mask.astype(bool)]
    return {
        f"{region_key}_jac_mean": float(vals.mean()),
        f"{region_key}_jac_median": float(np.median(vals)),
        f"{region_key}_jac_p05": float(np.percentile(vals, 5)),
        f"{region_key}_jac_p95": float(np.percentile(vals, 95)),
    }


def inverse_consistency(reg, atlas_brain_mask, atlas_resolution_um, n_points=2000, seed=0) -> float:
    """M6: round-trip random atlas-space points through atlas->sample->atlas
    (invtransforms then fwdtransforms via transform_cell_points), and measure
    the mean residual displacement from the original point, in atlas physical
    microns. Cheap point-based check instead of composing dense warp fields.

    atlas_brain_mask/atlas_resolution_um: (x,y,z)-order atlas-space mask (see
    load_jacobian's docstring on why atlas arrays use that order here) and
    the atlas's isotropic voxel size.
    """
    rng = np.random.default_rng(seed)
    idx_xyz = np.argwhere(atlas_brain_mask)
    if len(idx_xyz) == 0:
        return float("nan")
    chosen = idx_xyz[rng.choice(len(idx_xyz), size=min(n_points, len(idx_xyz)), replace=False)]
    xyz_um = chosen.astype(float) * atlas_resolution_um

    df = pd.DataFrame({"x": xyz_um[:, 0], "y": xyz_um[:, 1], "z": xyz_um[:, 2]})
    to_sample = transforms.transform_cell_points(df, reg, direction="atlas_to_sample")
    back_to_atlas = transforms.transform_cell_points(to_sample, reg, direction="sample_to_atlas")

    residual = np.linalg.norm(back_to_atlas[["x", "y", "z"]].to_numpy() - xyz_um, axis=1)
    return float(residual.mean())


# =====================================================================================
# MAIN
# =====================================================================================
def evaluate_sample(sample_id: str, paths: dict, cfg: Config, atlas_ctx: dict) -> dict:
    """Full metric set for one sample. Metrics whose required annotation
    files aren't configured yet are skipped (with a printed note) rather
    than raising -- annotation is a progressive, manual process, so a sample
    can be listed and partially scored before every ground-truth file exists.
    """
    row: dict = {"sample": sample_id, "group": cfg.groups.get(sample_id, "unknown")}

    # Loaded once and reused as the reference grid for every ground-truth
    # comparison below. Ground truth may have been hand-drawn against a
    # DIFFERENT registration of this sample (a different atlas, a different
    # fine_target_um, or a different tool entirely) -- every ground-truth
    # mask gets nearest-neighbor-resampled onto THIS grid before comparison
    # (see resample_mask_to_reference), so one set of hand corrections can
    # score any number of registration methods/atlases without re-drawing.
    labels_img = load_sample_space_image(paths["labels_in_sample"])
    labels_arr = sitk.GetArrayFromImage(labels_img)

    # ---- M1 TRE ----
    if paths.get("sample_landmarks_csv") and paths.get("atlas_landmarks_csv"):
        sample_pts_vox = load_points(paths["sample_landmarks_csv"])
        atlas_pts_vox = load_points(paths["atlas_landmarks_csv"])
        sample_pts_um = sample_pts_vox * paths["sample_resolution_um"]
        atlas_pts_um = atlas_pts_vox * paths["atlas_resolution_um"]
        warped_um = apply_transform_to_points(sample_pts_um, paths["transforms_prefix"])
        tre = compute_tre(warped_um, atlas_pts_um)
        row.update(summarize_tre(tre, paths.get("cortex_landmark_idx")))
    else:
        print(f"[{sample_id}] no landmark CSVs configured yet -- skipping M1 (TRE).")

    # ---- M2 / M3 whole-brain outline ----
    if paths.get("sample_brain_mask_corrected"):
        wa_brain = labels_arr > 0
        sm_brain_img = resample_mask_to_reference(
            load_sample_space_image(paths["sample_brain_mask_corrected"]), labels_img)
        sm_brain = sitk.GetArrayFromImage(sm_brain_img) > 0
        wb = evaluate_structure(wa_brain, sm_brain, paths["sample_resolution_um"], cfg.annotated_slices)
        row.update({"brain_dice": wb["dice"], "brain_hd95_um": wb["hd95_um"]})
    else:
        print(f"[{sample_id}] no sample_brain_mask_corrected configured yet -- skipping whole-brain Dice/HD95.")

    # ---- M2 / M3 coarse regions ----
    # Each region is annotated independently (see tools/edit_sample_labels.py's
    # "Save This Region"), so each is skipped on its own rather than all-or-nothing.
    region_masks = paths.get("dice_region_masks") or {}
    if not region_masks:
        print(f"[{sample_id}] no dice_region_masks configured yet -- skipping all region Dice/HD95.")
    for region in cfg.dice_regions:
        key = region.lower().replace(" ", "_")
        mask_path = region_masks.get(region)
        if not mask_path:
            print(f"[{sample_id}] no dice_region_masks['{region}'] yet -- skipping {key} Dice/HD95.")
            continue
        wa = load_region_mask(labels_arr, region, atlas_ctx["structures"])
        # Ground truth's own grid may differ from this registration's
        # labels_in_sample (drawn against a different atlas/fine_target_um/
        # tool) -- resample onto labels_img's grid rather than requiring an
        # exact shape match, so the same ground truth can score every
        # registration method/atlas being compared.
        sm_img_raw = load_sample_space_image(mask_path)
        sm_img = resample_mask_to_reference(sm_img_raw, labels_img)
        sm = sitk.GetArrayFromImage(sm_img) > 0
        # Explicit annotated_slices (if set) applies to every region as-is;
        # otherwise fall back to this region's own hand-drawn-slices sidecar
        # (see load_region_annotation_hint), remapped onto labels_img's grid
        # if it needed resampling above -- so sparse per-region ground truth
        # is scored on just the planes actually drawn without having to
        # retype z-indices per region -- None (no override, no hint) means
        # the full volume, same as before.
        slices = cfg.annotated_slices
        if slices is None and cfg.use_annotation_hints:
            hint = load_region_annotation_hint(mask_path)
            slices = remap_annotated_slices(hint, sm_img_raw, labels_img)
        r = evaluate_structure(wa, sm, paths["sample_resolution_um"], slices)
        row.update({f"{key}_dice": r["dice"], f"{key}_hd95_um": r["hd95_um"]})

    # ---- M5 / M6 / M7 Jacobian + inverse consistency (atlas space) ----
    warp_path = Path(paths["transforms_prefix"] + "1Warp.nii.gz")
    if warp_path.exists():
        jac = load_jacobian(warp_path, atlas_ctx["atlas_template"])
        row["neg_jac_frac"] = neg_jacobian_fraction(jac, atlas_ctx["atlas_brain_mask"])
        for region in cfg.dice_regions:
            key = region.lower().replace(" ", "_")
            row.update(region_jacobian_stats(jac, atlas_ctx["atlas_region_masks"][region], key))

        reg = transforms.load_saved_transforms(paths["transforms_prefix"])
        row["inv_consistency_um"] = inverse_consistency(
            reg, atlas_ctx["atlas_brain_mask"], atlas_ctx["atlas_resolution_um"]
        )
    else:
        print(f"[{sample_id}] no forward warp found at {warp_path} -- skipping Jacobian/inverse-consistency metrics.")

    return row


def main(eval_config_path: str, out_csv: str = "reg_metrics.csv"):
    """Run all samples listed in eval_config_path, save a tidy table, print a
    per-group summary."""
    cfg, per_sample_paths, atlas_cfg = load_eval_config(eval_config_path)

    atlas_template, atlas_annotation = atlas_utils.load_custom_atlas(
        atlas_cfg["template_path"], atlas_cfg["annotation_path"], atlas_cfg["resolution_um"],
    )
    structures = (
        atlas_utils.load_ccf_ontology_json(atlas_cfg["ontology_path"]) if atlas_cfg.get("ontology_path") else None
    )
    atlas_arr = atlas_annotation.numpy()
    atlas_ctx = {
        "atlas_template": atlas_template,
        "structures": structures,
        "atlas_brain_mask": atlas_arr > 0,
        "atlas_region_masks": {
            region: atlas_utils.region_mask_by_exact_name(atlas_arr, structures, region)
            for region in cfg.dice_regions
        },
        "atlas_resolution_um": float(atlas_cfg["resolution_um"]),
    }

    rows = [evaluate_sample(sid, paths, cfg, atlas_ctx) for sid, paths in per_sample_paths.items()]
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print("\n=== per-sample ===")
    print(df.to_string(index=False))

    key_cols = [c for c in df.columns if c not in ("sample", "group")]
    if key_cols:
        print("\n=== per-group (compare TSC vs normal by eye; small n -> describe, don't p-test) ===")
        print(df.groupby("group")[key_cols].agg(["median", "min", "max"]).to_string())
    return df


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/eval_config.yaml"
    main(config_path)
