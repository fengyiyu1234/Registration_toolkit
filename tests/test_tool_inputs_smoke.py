"""Smoke tests for the two shared input helpers every tool in this repo now
goes through: _landmark_io (the landmark CSV format) and _local_config (paths
from configs/<tool>.yaml, with or without the form window).

Same manual-assert style as the other tests here, no pytest. Headless on
purpose -- nothing below opens a Qt window, so this runs over ssh:

    conda activate antsreg
    python tests/test_tool_inputs_smoke.py

Needs numpy/pandas/pyyaml/antspyx (the last only for the end-to-end
fit_initial_transform.py --no-form run at the end).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import _landmark_io  # noqa: E402
import _local_config  # noqa: E402


# =====================================================================================
# _landmark_io
# =====================================================================================
def test_landmark_csv_roundtrip(tmp):
    print("1. _landmark_io: CSV round trip, axis order, physical conversion...")
    csv = tmp / "pts.csv"
    voxels = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    _landmark_io.write_landmark_csv(csv, voxels)

    header = csv.read_text().splitlines()[0]
    assert header == "index,axis-0,axis-1,axis-2", f"format drifted: {header!r}"

    back = _landmark_io.read_landmark_csv(csv)
    assert np.allclose(back, voxels), f"round trip changed the values: {back}"

    # (z,y,x) -> (x,y,z). This is the reversal every physical-space convention
    # in the repo depends on; getting it wrong is silent.
    xyz = _landmark_io.to_xyz(back)
    assert np.allclose(xyz[0], [3.0, 2.0, 1.0]), xyz[0]

    # physical = voxel_xyz * spacing + origin, with the origin carrying the crop
    # offset. Anisotropic spacing and a nonzero origin on purpose: under
    # isotropic spacing at origin 0 an axis-order bug still gives the right
    # answer by luck.
    phys = _landmark_io.voxels_to_physical(back, (10.0, 25.0, 40.0), (5.0, -2.0, 8.0))
    assert np.allclose(phys[0], [3 * 10 + 5, 2 * 25 - 2, 1 * 40 + 8]), phys[0]
    print("   OK")


def test_landmark_csv_rejects_bad_files(tmp):
    print("2. _landmark_io: bad CSVs are refused, with the reason named...")

    def expect(needle, fn, exc=ValueError):
        try:
            fn()
        except exc as e:
            assert needle in str(e), f"message does not mention {needle!r}: {e}"
            return
        raise AssertionError(f"expected {exc.__name__} mentioning {needle!r}")

    wrong_cols = tmp / "wrong.csv"
    wrong_cols.write_text("x,y,z\n1,2,3\n")
    expect("axis-0", lambda: _landmark_io.read_landmark_csv(wrong_cols))

    empty = tmp / "empty.csv"
    empty.write_text("index,axis-0,axis-1,axis-2\n")
    expect("no rows", lambda: _landmark_io.read_landmark_csv(empty))

    nan = tmp / "nan.csv"
    nan.write_text("index,axis-0,axis-1,axis-2\n0,1,2,3\n1,4,,6\n")
    expect("row(s) [1]", lambda: _landmark_io.read_landmark_csv(nan))

    expect("not found", lambda: _landmark_io.read_landmark_csv(tmp / "nope.csv"),
           exc=FileNotFoundError)

    # The check that catches "these points were placed on a different image".
    ok = tmp / "ok.csv"
    _landmark_io.write_landmark_csv(ok, np.array([[1.0, 2.0, 3.0], [40.0, 5.0, 6.0]]))
    _landmark_io.read_landmark_csv(ok, shape_zyx=(50, 50, 50))       # inside: fine
    expect("outside", lambda: _landmark_io.read_landmark_csv(ok, shape_zyx=(10, 10, 10)))
    print("   OK")


def test_load_points_agrees_with_landmark_io(tmp):
    print("3. registration_eval.load_points still returns (x,y,z) voxels...")
    try:
        import registration_eval as ev
    except Exception as exc:      # needs antspyx + registration_ants
        print(f"   skipped: {type(exc).__name__}: {exc}")
        return
    csv = tmp / "eval_pts.csv"
    voxels = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    _landmark_io.write_landmark_csv(csv, voxels)
    pts = ev.load_points(csv)
    assert np.allclose(pts, voxels[:, ::-1]), f"load_points changed behaviour: {pts}"
    print("   OK (unchanged by the move into _landmark_io)")


# =====================================================================================
# _local_config
# =====================================================================================
_FIELDS = [
    {"key": "image_path", "type": "open_file"},
    {"key": "output_csv", "type": "save_file"},
    {"key": "use_alt", "type": "checkbox", "default": False},
    {"key": "alt_path", "type": "open_file", "enabled_when": ("use_alt", True)},
    {"key": "levels", "type": "int", "default": 4},
    {"key": "note", "type": "open_file", "optional": True},
]


def _write_cfg(path, **values):
    path.write_text(yaml.safe_dump(values))
    return path


def test_values_from_config(tmp):
    print("4. _local_config.values_from_config (the --no-form path)...")
    cfg = {"image_path": "/a/img.nii.gz", "output_csv": "/a/out.csv"}
    values = _local_config.values_from_config("dummy", _FIELDS, cfg)
    assert values["image_path"] == "/a/img.nii.gz"
    assert values["levels"] == 4, "a field's own default should fill in"
    assert values["use_alt"] is False
    # enabled_when not satisfied -> None, exactly like a greyed-out form field,
    # and NOT a missing-value error.
    assert values["alt_path"] is None, values["alt_path"]
    assert values["note"] is None, "optional fields may stay empty"

    values = _local_config.values_from_config(
        "dummy", _FIELDS, {**cfg, "use_alt": True, "alt_path": "/a/alt.nii.gz"})
    assert values["alt_path"] == "/a/alt.nii.gz", "enabled field should be kept"

    def expect(needle, fn, exc=ValueError):
        try:
            fn()
        except exc as e:
            assert needle in str(e), f"message does not mention {needle!r}: {e}"
            return
        raise AssertionError(f"expected {exc.__name__} mentioning {needle!r}")

    expect("output_csv", lambda: _local_config.values_from_config(
        "dummy", _FIELDS, {"image_path": "/a/img.nii.gz"}))
    # A required field that is enabled only conditionally must be demanded too.
    expect("alt_path", lambda: _local_config.values_from_config(
        "dummy", _FIELDS, {**cfg, "use_alt": True}))
    # --no-form with no config at all is an error, not an empty run.
    expect("找不到配置文件", lambda: _local_config.values_from_config("dummy", _FIELDS, {}),
           exc=FileNotFoundError)
    print("   OK")


def test_resolve_inputs_no_form(tmp):
    print("5. _local_config.resolve_inputs(no_form=True) reads an explicit config...")
    cfg_path = _write_cfg(tmp / "tool.yaml", image_path="~/img.nii.gz",
                          output_csv="${HOME}/out.csv", levels=6)
    values = _local_config.resolve_inputs("dummy", "Dummy", _FIELDS,
                                          cli_path=cfg_path, no_form=True)
    home = str(Path.home())
    assert values["image_path"] == f"{home}/img.nii.gz", values["image_path"]
    assert values["output_csv"] == f"{home}/out.csv", "${VAR} should expand"
    assert values["levels"] == 6, "config must win over the field default"

    try:
        _local_config.resolve_inputs("dummy", "Dummy", _FIELDS,
                                     cli_path=tmp / "missing.yaml", no_form=True)
    except FileNotFoundError as e:
        assert "不存在" in str(e), e
    else:
        raise AssertionError("an explicitly named missing config must be an error")
    print("   OK (no Qt import on this path)")


# =====================================================================================
# End to end through the CLI
# =====================================================================================
def test_fit_initial_transform_no_form(tmp):
    print("6. fit_initial_transform.py --no-form, end to end on phantom files...")
    try:
        import ants  # noqa: F401
        import fit_initial_transform as fit
    except Exception as exc:
        print(f"   skipped: {type(exc).__name__}: {exc}")
        return
    import ants

    atlas_img, _, samp_img, _ = fit._phantom_images()
    atlas_pts, samp_pts = fit._phantom_landmarks()
    _landmark_io.write_landmark_csv(tmp / "a.csv", fit._phys_to_voxel_zyx(atlas_pts, atlas_img))
    _landmark_io.write_landmark_csv(tmp / "s.csv", fit._phys_to_voxel_zyx(samp_pts, samp_img))
    ants.image_write(atlas_img, str(tmp / "atlas.nii.gz"))
    ants.image_write(samp_img, str(tmp / "sample.nii.gz"))

    cfg_path = _write_cfg(
        tmp / "fit.yaml",
        sample_landmarks=str(tmp / "s.csv"), atlas_landmarks=str(tmp / "a.csv"),
        sample_image=str(tmp / "sample.nii.gz"), atlas_image=str(tmp / "atlas.nii.gz"),
        out_prefix=str(tmp / "cfg_"), fitting_levels=4)

    proc = subprocess.run(
        [sys.executable, str(ROOT / "fit_initial_transform.py"), str(cfg_path), "--no-form"],
        capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, f"--no-form run failed:\n{proc.stdout}\n{proc.stderr}"
    assert (tmp / "cfg_init_fwd.nii.gz").exists(), proc.stdout
    assert (tmp / "cfg_init_inv.nii.gz").exists(), proc.stdout

    # Half a command line is refused rather than topped up from the config.
    proc = subprocess.run(
        [sys.executable, str(ROOT / "fit_initial_transform.py"),
         "--sample-landmarks", str(tmp / "s.csv")],
        capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode != 0, "a partial command line was accepted"
    assert "--atlas-landmarks" in proc.stderr, proc.stderr

    # The real workflow: sample landmarks placed on the raw acquisition TIFF,
    # whose voxel size can only come from the config (as a YAML list) and whose
    # grid must NOT become the inverse field's domain.
    raw = fit._write_phantom_tiff(tmp / "raw.tif", (40, 460, 400))
    raw_geom = fit.read_geometry(raw, fit._RAW_VOXEL, role="sample")
    _landmark_io.write_landmark_csv(tmp / "s_raw.csv",
                                    fit._phys_to_voxel_zyx(samp_pts, raw_geom))
    tiff_cfg = dict(
        sample_landmarks=str(tmp / "s_raw.csv"), atlas_landmarks=str(tmp / "a.csv"),
        sample_image=str(raw), atlas_image=str(tmp / "atlas.nii.gz"),
        out_prefix=str(tmp / "tif_"), fitting_levels=5)

    proc = subprocess.run(
        [sys.executable, str(ROOT / "fit_initial_transform.py"),
         str(_write_cfg(tmp / "fit_tiff.yaml", sample_voxel_size=list(fit._RAW_VOXEL),
                        **tiff_cfg)), "--no-form"],
        capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, f"TIFF run failed:\n{proc.stdout}\n{proc.stderr}"
    inv = ants.image_read(str(tmp / "tif_init_inv.nii.gz"))
    assert fit._shape_zyx(inv) != raw_geom.shape_zyx, \
        "the inverse field was written on the raw acquisition grid"
    assert np.allclose(inv.spacing, (fit.DEFAULT_TARGET_UM,) * 3), inv.spacing

    # ...and the same config with the voxel size left out must fail loudly
    # rather than fall back to the TIFF header's (1, 1, 1).
    proc = subprocess.run(
        [sys.executable, str(ROOT / "fit_initial_transform.py"),
         str(_write_cfg(tmp / "fit_novox.yaml", **{**tiff_cfg,
                                                   "out_prefix": str(tmp / "novox_")})),
         "--no-form"],
        capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode != 0, "a TIFF with no voxel size was accepted"
    assert "--sample-voxel-size" in proc.stderr, proc.stderr
    assert not (tmp / "novox_init_fwd.nii.gz").exists(), "wrote output despite the error"
    print("   OK (config -> run, partial flags refused, raw TIFF needs a voxel size)")


def main():
    print("=== tests/test_tool_inputs_smoke.py ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_landmark_csv_roundtrip(tmp)
        test_landmark_csv_rejects_bad_files(tmp)
        test_load_points_agrees_with_landmark_io(tmp)
        test_values_from_config(tmp)
        test_resolve_inputs_no_form(tmp)
        test_fit_initial_transform_no_form(tmp)
    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
