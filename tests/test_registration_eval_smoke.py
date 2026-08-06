"""Smoke tests for registration_eval.py: landmark TRE, region-mask Dice/HD95,
atlas-space Jacobian metrics, and the transform-reload plumbing that lets
registration_eval.py run in a separate process from whatever ran
ants.registration() originally.

Same manual-assert style as the smoke tests in ../Registration_ants/tests/
(no pytest). All-synthetic data throughout -- including a synthetic atlas
built directly via ants.from_numpy (no brainglobe download, no real DeMBA
files), so this test has no external/network dependency and no assumption
about what's mounted at /data/... Run manually (needs antspyx, so the antsreg
env): `python tests/test_registration_eval_smoke.py`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import tifffile

# Repo root, so `import registration_eval` finds the module one level up.
# registration_ants itself comes from ../Registration_ants's editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from registration_ants import atlas_utils, io_utils, register, transforms  # noqa: E402
import registration_eval as ev  # noqa: E402


def make_synthetic_sample(path, shape_zyx=(30, 40, 40)):
    """A layered ellipsoid blob, standing in for a tiny 'brain' -- same
    pattern as test_pipeline_smoke.py's make_synthetic_stack."""
    z, y, x = shape_zyx
    zz, yy, xx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing="ij")
    cz, cy, cx = z / 2, y / 2, x / 2
    r = np.sqrt(((zz - cz) / (z * 0.35)) ** 2 + ((yy - cy) / (y * 0.35)) ** 2 + ((xx - cx) / (x * 0.35)) ** 2)
    blob = (r < 1).astype(np.float32) * 1000
    blob += (r < 0.5).astype(np.float32) * 500
    rng = np.random.default_rng(0)
    blob += rng.normal(0, 20, size=blob.shape)
    arr = np.clip(blob, 0, None).astype(np.uint16)
    tifffile.imwrite(path, arr)


def make_synthetic_atlas(shape_xyz=(24, 32, 32), resolution_um=25.0):
    """A tiny synthetic atlas template+annotation, built directly (no
    brainglobe/DeMBA files needed): two nested boxes standing in for a
    'cortex' shell and a 'hippocampus' core, inside a spherical 'brain'."""
    import ants

    x, y, z = shape_xyz
    xx, yy, zz = np.meshgrid(np.arange(x), np.arange(y), np.arange(z), indexing="ij")
    cx, cy, cz = x / 2, y / 2, z / 2
    r = np.sqrt(((xx - cx) / (x * 0.4)) ** 2 + ((yy - cy) / (y * 0.4)) ** 2 + ((zz - cz) / (z * 0.4)) ** 2)
    brain = r < 1.0
    core = r < 0.4

    template_arr = (brain.astype(np.float32) * 500 + core.astype(np.float32) * 500)
    annotation_arr = np.zeros(shape_xyz, dtype=np.float32)
    annotation_arr[brain & ~core] = 10.0   # "FakeCortex"
    annotation_arr[core] = 20.0            # "FakeHippo"

    spacing = (resolution_um,) * 3
    template = ants.from_numpy(template_arr, spacing=spacing)
    annotation = ants.from_numpy(annotation_arr, spacing=spacing)

    structures = {
        1: {"id": 1, "name": "root", "acronym": "root", "structure_id_path": [1]},
        10: {"id": 10, "name": "FakeCortex", "acronym": "FCX", "structure_id_path": [1, 10]},
        20: {"id": 20, "name": "FakeHippo", "acronym": "FHP", "structure_id_path": [1, 20]},
    }
    return template, annotation, structures


def write_napari_points_csv(path, points_xyz_reversed_to_zyx):
    """Write a CSV in the exact format place_landmarks.py produces
    (index,axis-0,axis-1,axis-2, values in (z,y,x) voxel order)."""
    df = pd.DataFrame({
        "index": range(len(points_xyz_reversed_to_zyx)),
        "axis-0": points_xyz_reversed_to_zyx[:, 0],
        "axis-1": points_xyz_reversed_to_zyx[:, 1],
        "axis-2": points_xyz_reversed_to_zyx[:, 2],
    })
    df.to_csv(path, index=False)


def test_load_points():
    print("1. registration_eval.load_points (napari CSV axis-order reversal)...")
    tmp = Path("/tmp/test_landmarks.csv")
    # napari (z,y,x) voxel order: point 0 at z=1,y=2,x=3; point 1 at z=4,y=5,x=6
    write_napari_points_csv(tmp, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    pts_xyz = ev.load_points(tmp)
    assert pts_xyz.shape == (2, 3)
    assert np.allclose(pts_xyz[0], [3.0, 2.0, 1.0]), f"expected (x,y,z)=(3,2,1), got {pts_xyz[0]}"
    assert np.allclose(pts_xyz[1], [6.0, 5.0, 4.0]), f"expected (x,y,z)=(6,5,4), got {pts_xyz[1]}"
    print("   OK (napari (z,y,x) -> (x,y,z) reversal correct)")


def test_region_mask_dice_hd95(tmp_dir):
    print("2. load_region_mask / load_binary_mask / dice / hd95...")
    _, annotation, structures = make_synthetic_atlas()
    arr = annotation.numpy()  # (x,y,z), values in {0, 10, 20}

    labels_path = tmp_dir / "fake_labels_in_sample.nii.gz"
    # Write via sitk the same way the real pipeline's downstream tools do
    # (transposed to (z,y,x) on the way in, since sitk's GetImageFromArray
    # expects that -- mirrors ants.image_write -> sitk.ReadImage round trip).
    sitk.WriteImage(sitk.GetImageFromArray(np.transpose(arr, (2, 1, 0)).astype(np.uint32)), str(labels_path))

    labels_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(labels_path)))
    cortex_mask = ev.load_region_mask(labels_arr, "FakeCortex", structures)
    hippo_mask = ev.load_region_mask(labels_arr, "FakeHippo", structures)
    brain_mask = ev.load_brain_mask_from_labels(labels_path)

    assert cortex_mask.sum() == int((arr == 10).sum())
    assert hippo_mask.sum() == int((arr == 20).sum())
    assert brain_mask.sum() == int((arr > 0).sum())
    assert not np.any(cortex_mask & hippo_mask), "cortex/hippo masks should not overlap"
    print(f"   region voxel counts: cortex={cortex_mask.sum()}, hippo={hippo_mask.sum()}, brain={brain_mask.sum()}")

    # Dice of a mask against itself must be 1.0; HD95 must be ~0.
    d = ev.dice(cortex_mask, cortex_mask)
    assert abs(d - 1.0) < 1e-9, f"self-Dice should be 1.0, got {d}"
    h = ev.hd95(cortex_mask, cortex_mask, (25.0, 25.0, 25.0))
    assert h < 1e-6, f"self-HD95 should be ~0, got {h}"

    r = ev.evaluate_structure(cortex_mask, hippo_mask, sample_resolution_um=25.0)
    assert 0.0 <= r["dice"] <= 1.0 and r["dice"] < 1.0, "disjoint regions should have Dice < 1"
    print(f"   cortex-vs-hippo (disjoint) dice={r['dice']:.3f}, hd95={r['hd95_um']:.1f}um")
    print("   OK")


def test_resample_mask_to_reference():
    print("3. resample_mask_to_reference / remap_annotated_slices "
          "(cross-resolution ground truth, e.g. demba_p5 @25um vs devccf_p04 @20um)...")

    def make_box_image(spacing_um, size_um=400.0, box_um=(100.0, 300.0)):
        """A synthetic box with an EXACT physical bounding box (chosen as a
        multiple of both 20 and 25um so there's no rounding ambiguity in the
        test itself), rasterized at whatever isotropic spacing is passed in
        -- standing in for two registrations of the same sample at different
        resolutions (e.g. demba_p5 @25um vs devccf_p04 @20um)."""
        n = int(round(size_um / spacing_um))
        lo = int(round(box_um[0] / spacing_um))
        hi = int(round(box_um[1] / spacing_um))
        arr = np.zeros((n, n, n), dtype=np.uint8)
        arr[lo:hi, lo:hi, lo:hi] = 1
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((spacing_um,) * 3)
        return img

    reference_img = make_box_image(20.0)     # e.g. devccf_p04's own grid
    ground_truth_img = make_box_image(25.0)  # e.g. hand-drawn against demba_p5's grid

    same = ev.resample_mask_to_reference(reference_img, reference_img)
    assert same is reference_img, "same-grid input must be returned as-is (no-op)"

    resampled = ev.resample_mask_to_reference(ground_truth_img, reference_img)
    assert resampled.GetSize() == reference_img.GetSize()
    assert np.allclose(resampled.GetSpacing(), reference_img.GetSpacing())

    # The resampled mask's physical bounding box should land close to the
    # ORIGINAL ground-truth box's physical bounding box ([100,300]um), within
    # one reference voxel (20um) of slop -- i.e. resampling relocated the
    # mask onto the new grid without shifting *where in physical space* it
    # sits. (Checking bounding boxes directly, rather than Dice against an
    # independently-rasterized box, avoids conflating resample correctness
    # with this test's own box/grid quantization.)
    resampled_arr = sitk.GetArrayFromImage(resampled) > 0
    idx = np.argwhere(resampled_arr)
    assert idx.size > 0, "resampled mask should not be empty"
    resampled_lo_um = idx.min(axis=0) * 20.0
    resampled_hi_um = (idx.max(axis=0) + 1) * 20.0
    assert np.all(np.abs(resampled_lo_um - 100.0) <= 20.0), resampled_lo_um
    assert np.all(np.abs(resampled_hi_um - 300.0) <= 20.0), resampled_hi_um
    print(f"   resampled bbox (um): lo={resampled_lo_um}, hi={resampled_hi_um} "
          f"(original ground truth box was [100,300]um) -- shape {resampled_arr.shape}")

    # A z-plane hand-drawn at index 5 on the 25um ground truth grid (physical
    # z = 5*25 = 125um) should land at index round(125/20) = 6 on the 20um
    # reference grid.
    remapped = ev.remap_annotated_slices([5], ground_truth_img, reference_img)
    assert remapped == [6], f"expected z=5 @25um -> z=6 @20um, got {remapped}"
    # Same-grid case must be a no-op passthrough.
    assert ev.remap_annotated_slices([3, 7], reference_img, reference_img) == [3, 7]
    assert ev.remap_annotated_slices(None, ground_truth_img, reference_img) is None
    print("   OK (cross-resolution resample + slice-index remap correct)")


def test_eval_config_groups(tmp_dir):
    print("4. load_eval_config / _load_groups_from_manifest...")
    manifest_path = tmp_dir / "fake_stats_config.yaml"
    manifest_path.write_text(
        "groups:\n"
        "  a: {name: Control, samples: [s1, s2]}\n"
        "  b: {name: Experimental, samples: [s3]}\n"
    )
    groups = ev._load_groups_from_manifest(manifest_path)
    assert groups == {"s1": "Control", "s2": "Control", "s3": "Experimental"}
    print("   OK (groups.a/b flattened to {sample: group_name})")


def test_end_to_end_synthetic_registration(tmp_dir):
    print("5. end-to-end: synthetic registration -> transforms reload -> "
          "apply_transform_to_points / load_jacobian / inverse_consistency / evaluate_sample...")

    raw_tiff = tmp_dir / "synthetic_sample.tif"
    make_synthetic_sample(raw_tiff)
    voxel_size_xyz = (20.0, 20.0, 90.0)
    target_um = 25.0

    sample_img = io_utils.load_tiff_stack_as_ants(raw_tiff, voxel_size_xyz)
    sample_iso = io_utils.resample_to_isotropic(sample_img, target_um)

    atlas_template, atlas_annotation, structures = make_synthetic_atlas(resolution_um=target_um)

    transforms_dir = tmp_dir / "transforms"
    transforms_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(transforms_dir / "synth_")

    reg = register.register_to_atlas(
        sample_iso, atlas_template, atlas_annotation, atlas_structures=structures,
        type_of_transform="SyNRA", outprefix=prefix, verbose=False,
    )

    # --- load_saved_transforms must reconstruct exactly what ants.registration()
    # itself returned (this is the "guess the file order right" risk). ---
    reloaded = transforms.load_saved_transforms(prefix)
    assert reloaded["fwdtransforms"] == reg["fwdtransforms"], (
        f"fwdtransforms mismatch:\n  live={reg['fwdtransforms']}\n  reloaded={reloaded['fwdtransforms']}"
    )
    assert reloaded["invtransforms"] == reg["invtransforms"], (
        f"invtransforms mismatch:\n  live={reg['invtransforms']}\n  reloaded={reloaded['invtransforms']}"
    )
    print("   load_saved_transforms matches the live reg dict exactly -- OK")

    # --- labels_in_sample + a fabricated "corrected" version (ground truth stand-in) ---
    labels_in_sample_img = transforms.warp_labels_to_sample(sample_iso, reg)
    labels_path = tmp_dir / "sample_labels_in_sample.nii.gz"
    import ants
    ants.image_write(labels_in_sample_img, str(labels_path))

    # Re-read via sitk (as load_region_mask/load_brain_mask_from_labels do) and
    # verify the round trip through two different libraries didn't scramble
    # the label ids -- exactly the risk this whole file's axis-order docstring
    # warns about.
    arr_written = labels_in_sample_img.numpy()
    arr_reread = sitk.GetArrayFromImage(sitk.ReadImage(str(labels_path)))
    assert set(np.unique(arr_written)) == set(np.unique(arr_reread)), "label ids changed across ants<->sitk round trip"

    # Per-region ground-truth masks (fabricated identical to the auto labels),
    # matching edit_sample_labels.py's "Save This Region" output --
    # one independent binary mask per region, not one combined multi-label file.
    cortex_mask_path = tmp_dir / "sample_fakecortex_corrected_mask.nii.gz"
    hippo_mask_path = tmp_dir / "sample_fakehippo_corrected_mask.nii.gz"
    reference_sitk = sitk.ReadImage(str(labels_path))
    cortex_mask_sitk = sitk.GetImageFromArray((arr_reread == 10).astype(np.uint8))
    cortex_mask_sitk.CopyInformation(reference_sitk)
    sitk.WriteImage(cortex_mask_sitk, str(cortex_mask_path))
    hippo_mask_sitk = sitk.GetImageFromArray((arr_reread == 20).astype(np.uint8))
    hippo_mask_sitk.CopyInformation(reference_sitk)
    sitk.WriteImage(hippo_mask_sitk, str(hippo_mask_path))

    brain_mask_path = tmp_dir / "sample_brain_mask_corrected.nii.gz"
    brain_mask_sitk = sitk.GetImageFromArray((arr_reread > 0).astype(np.uint8))
    brain_mask_sitk.CopyInformation(sitk.ReadImage(str(labels_path)))
    sitk.WriteImage(brain_mask_sitk, str(brain_mask_path))

    # --- fabricate matching sample/atlas landmark CSVs ---
    sample_landmarks_path = tmp_dir / "sample_landmarks.csv"
    atlas_landmarks_path = tmp_dir / "atlas_landmarks.csv"
    sample_shape_zyx = np.array(sample_iso.shape)[::-1]  # ants (x,y,z) -> (z,y,x)
    atlas_shape_zyx = np.array(atlas_template.shape)[::-1]
    write_napari_points_csv(sample_landmarks_path, np.array([sample_shape_zyx / 2, sample_shape_zyx / 3]))
    write_napari_points_csv(atlas_landmarks_path, np.array([atlas_shape_zyx / 2, atlas_shape_zyx / 3]))

    # --- apply_transform_to_points / compute_tre ---
    sample_pts_vox = ev.load_points(sample_landmarks_path)
    atlas_pts_vox = ev.load_points(atlas_landmarks_path)
    sample_pts_um = sample_pts_vox * target_um
    atlas_pts_um = atlas_pts_vox * target_um
    warped_um = ev.apply_transform_to_points(sample_pts_um, prefix)
    assert np.isfinite(warped_um).all()
    tre = ev.compute_tre(warped_um, atlas_pts_um)
    assert np.isfinite(tre).all() and (tre >= 0).all()
    print(f"   TRE (meaningless numerically here, synthetic data): {tre}")

    # --- load_jacobian / neg_jacobian_fraction / region_jacobian_stats ---
    warp_path = Path(prefix + "1Warp.nii.gz")
    assert warp_path.exists()
    jac = ev.load_jacobian(warp_path, atlas_template)
    assert jac.shape == atlas_template.shape
    assert np.isfinite(jac).all()

    atlas_arr = atlas_annotation.numpy()
    atlas_brain_mask = atlas_arr > 0
    atlas_cortex_mask = atlas_utils.region_mask_by_exact_name(atlas_arr, structures, "FakeCortex")
    neg_frac = ev.neg_jacobian_fraction(jac, atlas_brain_mask)
    assert 0.0 <= neg_frac <= 1.0
    jac_stats = ev.region_jacobian_stats(jac, atlas_cortex_mask, "fakecortex")
    assert set(jac_stats) == {"fakecortex_jac_mean", "fakecortex_jac_median", "fakecortex_jac_p05", "fakecortex_jac_p95"}
    assert all(np.isfinite(v) for v in jac_stats.values())
    print(f"   neg_jac_frac={neg_frac:.4f}, {jac_stats}")

    # --- inverse_consistency ---
    inv = ev.inverse_consistency(reg, atlas_brain_mask, target_um, n_points=200)
    assert np.isfinite(inv) and inv >= 0
    print(f"   inverse_consistency_um={inv:.2f}")

    # --- evaluate_sample, fully configured ---
    cfg = ev.Config(dice_regions=["FakeCortex", "FakeHippo"], groups={"synth01": "Control"})
    paths_full = {
        "sample_resolution_um": target_um,
        "atlas_resolution_um": target_um,
        "labels_in_sample": labels_path,
        "transforms_prefix": prefix,
        "sample_landmarks_csv": sample_landmarks_path,
        "atlas_landmarks_csv": atlas_landmarks_path,
        "cortex_landmark_idx": None,
        "dice_region_masks": {"FakeCortex": cortex_mask_path, "FakeHippo": hippo_mask_path},
        "sample_brain_mask_corrected": brain_mask_path,
    }
    atlas_ctx = {
        "atlas_template": atlas_template,
        "structures": structures,
        "atlas_brain_mask": atlas_brain_mask,
        "atlas_region_masks": {
            r: atlas_utils.region_mask_by_exact_name(atlas_arr, structures, r)
            for r in cfg.dice_regions
        },
        "atlas_resolution_um": target_um,
    }
    row_full = ev.evaluate_sample("synth01", paths_full, cfg, atlas_ctx)
    expected_keys = {
        "sample", "group", "tre_mean_um", "tre_std_um", "tre_median_um", "n",
        "brain_dice", "brain_hd95_um",
        "fakecortex_dice", "fakecortex_hd95_um", "fakehippo_dice", "fakehippo_hd95_um",
        "neg_jac_frac", "inv_consistency_um",
        "fakecortex_jac_mean", "fakecortex_jac_median", "fakecortex_jac_p05", "fakecortex_jac_p95",
        "fakehippo_jac_mean", "fakehippo_jac_median", "fakehippo_jac_p05", "fakehippo_jac_p95",
    }
    missing = expected_keys - set(row_full)
    assert not missing, f"evaluate_sample missing expected keys: {missing}"
    # Ground truth was fabricated identical to the auto labels -> region Dice should be 1.0.
    assert abs(row_full["fakecortex_dice"] - 1.0) < 1e-9, row_full["fakecortex_dice"]
    assert abs(row_full["brain_dice"] - 1.0) < 1e-9, row_full["brain_dice"]
    print(f"   evaluate_sample (fully configured) row keys OK, "
          f"fakecortex_dice={row_full['fakecortex_dice']}, brain_dice={row_full['brain_dice']}")

    # --- evaluate_sample, minimal config (no ground truth yet) must not crash,
    # and must still run the Jacobian/inverse-consistency metrics that only
    # need the base pipeline outputs (labels_in_sample + transforms). ---
    paths_minimal = {
        "sample_resolution_um": target_um,
        "atlas_resolution_um": target_um,
        "labels_in_sample": labels_path,
        "transforms_prefix": prefix,
        "sample_landmarks_csv": None,
        "atlas_landmarks_csv": None,
        "cortex_landmark_idx": None,
        "dice_region_masks": {},
        "sample_brain_mask_corrected": None,
    }
    row_minimal = ev.evaluate_sample("synth01", paths_minimal, cfg, atlas_ctx)
    assert "tre_mean_um" not in row_minimal
    assert "brain_dice" not in row_minimal
    assert "neg_jac_frac" in row_minimal and np.isfinite(row_minimal["neg_jac_frac"])
    assert "inv_consistency_um" in row_minimal
    print("   evaluate_sample (minimal config) correctly skips TRE/Dice, still computes Jacobian metrics -- OK")

    print("   OK (full end-to-end synthetic registration)")


def main():
    tmp_dir = Path(
        "/tmp/claude-1004/-home-fyu7-My-project-Registration-ants/a41c76dc-5afa-4d34-b512-d61d021bf86f/"
        "scratchpad/registration_eval_smoketest"
    )
    tmp_dir.mkdir(parents=True, exist_ok=True)

    test_load_points()
    test_region_mask_dice_hd95(tmp_dir)
    test_resample_mask_to_reference()
    test_eval_config_groups(tmp_dir)
    test_end_to_end_synthetic_registration(tmp_dir)

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
