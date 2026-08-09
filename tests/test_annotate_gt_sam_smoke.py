"""Smoke tests for annotate_gt_sam.py's non-GUI half: config validation, the
manifest/sidecar pair, mask export geometry, and the --verify acceptance checks.

Same manual-assert style as the other smoke tests here (no pytest). All
synthetic -- no real volumes, no SAM model, no display. The interactive part
(point prompts -> segment -> commit -> save) cannot be exercised headlessly and
is covered by the manual checklist in README.md instead.

Runs in either env; needs only numpy + SimpleITK + PyYAML (importing
annotate_gt_sam does NOT import micro_sam or napari -- those are deferred into
the functions that open a GUI, which is what makes this test possible).

    python tests/test_annotate_gt_sam_smoke.py
"""
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import align_masks  # noqa: E402
import annotate_gt_sam as gt  # noqa: E402


SHAPE_ZYX = (24, 30, 36)
SPACING_XYZ = (10.0, 25.0, 40.0)
ORIGIN_XYZ = (5.0, -2.0, 8.0)


def make_volume(path):
    """A volume with deliberately anisotropic spacing and a nonzero origin, so
    any geometry that fails to propagate through the export shows up."""
    rng = np.random.default_rng(0)
    arr = (rng.random(SHAPE_ZYX) * 1000).astype(np.uint16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(SPACING_XYZ)
    img.SetOrigin(ORIGIN_XYZ)
    sitk.WriteImage(img, str(path))
    return img


def write_config(tmp, regions, **overrides):
    cfg = {
        "brain_id": "brain01",
        "volume": str(tmp / "volume.nii.gz"),
        "output_dir": str(tmp / "gt_masks"),
        "regions": regions,
    }
    cfg.update(overrides)
    path = tmp / "gt_annotation.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def blob_plane(seed):
    """A small filled rectangle, standing in for one committed_objects plane."""
    plane = np.zeros(SHAPE_ZYX[1:], dtype=np.uint8)
    plane[8 + seed % 3: 18, 10: 22 + seed % 4] = 1
    return plane


def test_config_validation(tmp):
    print("1. load_config rejects bad configs before anything expensive happens")
    make_volume(tmp / "volume.nii.gz")

    good = write_config(tmp, {"ventricle": [5, 10, 15]})
    cfg = gt.load_config(good)
    assert cfg["brain_id"] == "brain01"
    assert cfg["regions"]["ventricle"] == [5, 10, 15]
    assert cfg["model_type"] == "vit_b_lm"
    assert cfg["manifest"] == cfg["output_dir"] / "annotated_slices_manifest.json"
    assert cfg["embedding_cache_dir"] == cfg["output_dir"] / "_sam_embeddings"

    for regions, expect in [
        ({"ventricle": []}, "non-empty list"),
        ({"ventricle": [5, 5, 9]}, "duplicate"),
        ({"ventricle": [5, -1]}, "non-negative"),
        ({"ventricle": {"start": 15, "end": 5, "step": 5}}, "must be <="),
        ({"ventricle": {"start": 5, "end": 15, "step": 0}}, "positive int"),
        ({"ventricle": {"start": 5, "end": 15}}, "needs step"),
    ]:
        raised = False
        try:
            gt.load_config(write_config(tmp, regions))
        except ValueError as exc:
            raised = True
            assert expect in str(exc), f"unexpected message for {regions}: {exc}"
        assert raised, f"load_config accepted {regions}"

    print("   ok (rejections)")
    print("1b. regions also accepts {start, end, step}, mixed with explicit lists per region")
    range_cfg = gt.load_config(write_config(tmp, {
        "ventricle": {"start": 5, "end": 15, "step": 5},
        "cortex_surface": [6, 12],
    }))
    assert range_cfg["regions"]["ventricle"] == [5, 10, 15], range_cfg["regions"]["ventricle"]
    assert range_cfg["regions"]["cortex_surface"] == [6, 12], range_cfg["regions"]["cortex_surface"]

    raised = False
    try:
        gt.load_config(write_config(tmp, {"ventricle": [5]}, volume=str(tmp / "nope.nii.gz")))
    except FileNotFoundError:
        raised = True
    assert raised, "load_config accepted a missing volume"

    raised = False
    try:
        cfg_path = write_config(tmp, {"ventricle": [5]})
        raw = yaml.safe_load(cfg_path.read_text())
        del raw["output_dir"]
        cfg_path.write_text(yaml.safe_dump(raw))
        gt.load_config(cfg_path)
    except ValueError as exc:
        raised = True
        assert "output_dir" in str(exc)
    assert raised, "load_config accepted a config with no output_dir"
    print("   ok")


def annotate(cfg, region, z_list):
    """Simulate what a GUI session does on save: write planes into the region's
    3D mask, then update mask + manifest together."""
    reference = sitk.ReadImage(str(cfg["volume"]))
    mask = gt.load_region_mask(cfg, region, reference)
    for z in z_list:
        mask[z] = blob_plane(z)
    gt.save_region_mask(cfg, region, mask, reference)

    manifest = gt.load_manifest(cfg)
    manifest.setdefault(cfg["brain_id"], {})
    recorded = set(gt.manifest_slices(manifest, cfg["brain_id"], region)) | set(z_list)
    manifest[cfg["brain_id"]][region] = sorted(recorded)
    gt.save_manifest(cfg, manifest)


def annotate_absent(cfg, region, z_list):
    """Simulate 'Mark slice absent' clicks: no mask content written, just the
    manifest's confirmed-absent bookkeeping."""
    manifest = gt.load_manifest(cfg)
    for z in z_list:
        gt.record_region_slice(cfg, manifest, region, z, present=False)


def test_export_geometry_and_manifest(tmp):
    print("2. exported masks copy the volume's geometry; manifest + sidecars agree")
    reference = make_volume(tmp / "volume.nii.gz")
    cfg = gt.load_config(write_config(tmp, {"ventricle": [5, 10, 15],
                                            "cortex_surface": [6, 12]}))

    annotate(cfg, "ventricle", [5, 10, 15])
    annotate(cfg, "cortex_surface", [6, 12])

    for region, expected in (("ventricle", [5, 10, 15]), ("cortex_surface", [6, 12])):
        path = gt.mask_path_for(cfg, region)
        assert path.name == f"brain01_{region}.nii.gz", path.name
        img = sitk.ReadImage(str(path))

        check = align_masks.check_geometry(reference, img, name=region)
        assert check.ok, f"{region} geometry drifted from the volume:\n{check}"

        arr = sitk.GetArrayFromImage(img)
        assert arr.dtype == np.uint8, f"{region}: dtype {arr.dtype}"
        assert set(np.unique(arr).tolist()) <= {0, 1}, f"{region}: values {np.unique(arr)}"
        nonempty = sorted(int(z) for z in np.flatnonzero(arr.any(axis=(1, 2))))
        assert nonempty == expected, f"{region}: non-empty planes {nonempty} != {expected}"

        sidecar = json.loads(gt.sidecar_path_for(path).read_text())
        assert sidecar["hand_drawn_slices"] == expected, sidecar

    manifest = json.loads(cfg["manifest"].read_text())
    assert manifest == {"brain01": {"ventricle": [5, 10, 15], "cortex_surface": [6, 12]}}, manifest
    print("   ok")


def test_resume_does_not_lose_work(tmp):
    print("3. resuming keeps earlier slices and appends to the manifest")
    make_volume(tmp / "volume.nii.gz")
    cfg = gt.load_config(write_config(tmp, {"ventricle": [5, 10, 15]}))

    annotate(cfg, "ventricle", [5])
    annotate(cfg, "ventricle", [10])          # a later, separate "session"

    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(gt.mask_path_for(cfg, "ventricle"))))
    nonempty = sorted(int(z) for z in np.flatnonzero(arr.any(axis=(1, 2))))
    assert nonempty == [5, 10], f"resume lost work: {nonempty}"
    assert np.array_equal(arr[5], blob_plane(5)), "the first session's plane was altered"
    assert gt.manifest_slices(gt.load_manifest(cfg), "brain01", "ventricle") == [5, 10]

    # A mask whose grid no longer matches the volume must not be silently
    # appended to -- that is how a misaligned ground truth would sneak in.
    bad = sitk.GetImageFromArray(np.zeros(SHAPE_ZYX, dtype=np.uint8))
    bad.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(bad, str(gt.mask_path_for(cfg, "ventricle")))
    raised = False
    try:
        gt.load_region_mask(cfg, "ventricle", sitk.ReadImage(str(cfg["volume"])))
    except ValueError as exc:
        raised = True
        assert "spacing" in str(exc)
    assert raised, "load_region_mask accepted a mask on a different grid"
    print("   ok")


def test_verify_catches_manifest_mask_mismatch(tmp):
    print("4. --verify fails when the manifest and the mask disagree")
    make_volume(tmp / "volume.nii.gz")
    cfg = gt.load_config(write_config(tmp, {"ventricle": [5, 10, 15]}))
    annotate(cfg, "ventricle", [5, 10])
    assert gt.verify(cfg), "verify failed on consistent output"

    # (a) manifest claims a plane the mask does not have.
    manifest = gt.load_manifest(cfg)
    manifest["brain01"]["ventricle"] = [5, 10, 15]
    gt.save_manifest(cfg, manifest)
    assert not gt.verify(cfg), "verify passed with a manifest entry for an empty plane"

    # (b) mask has a plane the manifest does not list.
    manifest["brain01"]["ventricle"] = [5, 10]
    gt.save_manifest(cfg, manifest)
    path = gt.mask_path_for(cfg, "ventricle")
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    arr[15] = blob_plane(15)
    patched = sitk.GetImageFromArray(arr)
    patched.CopyInformation(img)
    sitk.WriteImage(patched, str(path))
    assert not gt.verify(cfg), "verify passed with an unrecorded annotated plane"

    # (c) values outside {0, 1}.
    manifest["brain01"]["ventricle"] = [5, 10, 15]
    gt.save_manifest(cfg, manifest)
    arr[15] = blob_plane(15) * 7
    patched = sitk.GetImageFromArray(arr)
    patched.CopyInformation(img)
    sitk.WriteImage(patched, str(path))
    assert not gt.verify(cfg), "verify passed with label values outside {0, 1}"

    # (d) a plane outside the config's locked slice list.
    arr[15] = blob_plane(15)
    arr[20] = blob_plane(20)
    patched = sitk.GetImageFromArray(arr)
    patched.CopyInformation(img)
    sitk.WriteImage(patched, str(path))
    manifest["brain01"]["ventricle"] = [5, 10, 15, 20]
    gt.save_manifest(cfg, manifest)
    assert not gt.verify(cfg), "verify passed with a z outside the locked slice list"
    print("   ok")


def test_confirmed_absent(tmp):
    print("5. confirmed-absent slices: recorded without mask content, verify passes, "
          "sidecar carries both present and absent")
    make_volume(tmp / "volume.nii.gz")
    cfg = gt.load_config(write_config(tmp, {"hippocampus": [2, 4, 10, 14, 18]}))

    annotate(cfg, "hippocampus", [10, 14, 18])          # drawn present (legacy flat-list write)
    annotate_absent(cfg, "hippocampus", [2, 4])         # confirmed absent (structure not present yet)

    assert gt.verify(cfg), "verify failed on a mix of present + confirmed-absent slices"

    sidecar = json.loads(gt.sidecar_path_for(gt.mask_path_for(cfg, "hippocampus")).read_text())
    assert sidecar["hand_drawn_slices"] == [2, 4, 10, 14, 18], sidecar
    assert sidecar["confirmed_absent_slices"] == [2, 4], sidecar

    manifest = gt.load_manifest(cfg)
    assert gt.manifest_slices(manifest, "brain01", "hippocampus") == [2, 4, 10, 14, 18]
    assert gt.manifest_absent_slices(manifest, "brain01", "hippocampus") == [2, 4]

    # Contradiction: manifest says a plane is confirmed absent but the mask has
    # content there -- verify() must catch it, not silently accept.
    path = gt.mask_path_for(cfg, "hippocampus")
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    arr[2] = blob_plane(2)
    patched = sitk.GetImageFromArray(arr)
    patched.CopyInformation(img)
    sitk.WriteImage(patched, str(path))
    assert not gt.verify(cfg), "verify passed despite a confirmed-absent plane having mask content"
    print("   ok")


def test_no_z_propagation_anywhere():
    print("6. no z-propagation API is referenced anywhere in annotate_gt_sam.py")
    source = (Path(__file__).resolve().parents[1] / "annotate_gt_sam.py").read_text()
    # Strip the docstrings/comments that explain WHY these are banned, so the
    # explanation itself doesn't trip the check.
    code = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))
    code = re.sub(r'""".*?"""', "", code, flags=re.S)

    banned = ["annotator_3d", "segment_nd", "segment_all_slices",
              "multi_dimensional_segmentation", "segment_mask_in_volume",
              "merge_instance_segmentation_3d", "propagat"]
    hits = [pattern for pattern in banned if pattern in code]
    assert not hits, f"z-propagation API referenced in executable code: {hits}"

    # And the only micro_sam entry point used is the 2D one.
    assert "annotator_2d" in code, "expected annotator_2d to be the entry point"
    print("   ok")


def main():
    # Each test gets its own temp dir so leftover masks/manifests can't leak
    # between them (test 4 deliberately corrupts its output).
    for fn in (test_config_validation,
               test_export_geometry_and_manifest,
               test_resume_does_not_lose_work,
               test_verify_catches_manifest_mask_mismatch,
               test_confirmed_absent):
        with tempfile.TemporaryDirectory() as tmp:
            fn(Path(tmp))
    test_no_z_propagation_anywhere()
    print("=== all annotate_gt_sam smoke tests passed ===")


if __name__ == "__main__":
    main()
