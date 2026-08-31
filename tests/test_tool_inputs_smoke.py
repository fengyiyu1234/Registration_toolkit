"""Smoke tests for the two shared input helpers every tool in this repo goes
through: shared.landmark_io (the landmark CSV format) and shared.local_config
(paths from configs/<tool>.yaml, with or without the form window).

Same manual-assert style as the other tests here, no pytest. Headless on
purpose -- nothing below opens a Qt window, so this runs over ssh:

    conda activate antsreg
    python tests/test_tool_inputs_smoke.py

Needs numpy/pandas/pyyaml (antspyx only for the registration_eval check, which
skips itself without it).
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared import atlas_reference  # noqa: E402
from shared import landmark_io  # noqa: E402
from shared import local_config  # noqa: E402


# =====================================================================================
# shared.landmark_io
# =====================================================================================
def test_landmark_csv_roundtrip(tmp):
    print("1. landmark_io: CSV round trip, axis order, physical conversion...")
    csv = tmp / "pts.csv"
    voxels = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    landmark_io.write_landmark_csv(csv, voxels)

    header = csv.read_text().splitlines()[0]
    assert header == "index,axis-0,axis-1,axis-2", f"format drifted: {header!r}"

    back = landmark_io.read_landmark_csv(csv)
    assert np.allclose(back, voxels), f"round trip changed the values: {back}"

    # (z,y,x) -> (x,y,z). This is the reversal every physical-space convention
    # in the repo depends on; getting it wrong is silent.
    xyz = landmark_io.to_xyz(back)
    assert np.allclose(xyz[0], [3.0, 2.0, 1.0]), xyz[0]

    # physical = voxel_xyz * spacing + origin, with the origin carrying the crop
    # offset. Anisotropic spacing and a nonzero origin on purpose: under
    # isotropic spacing at origin 0 an axis-order bug still gives the right
    # answer by luck.
    phys = landmark_io.voxels_to_physical(back, (10.0, 25.0, 40.0), (5.0, -2.0, 8.0))
    assert np.allclose(phys[0], [3 * 10 + 5, 2 * 25 - 2, 1 * 40 + 8]), phys[0]
    print("   OK")


def test_landmark_csv_rejects_bad_files(tmp):
    print("2. landmark_io: bad CSVs are refused, with the reason named...")

    def expect(needle, fn, exc=ValueError):
        try:
            fn()
        except exc as e:
            assert needle in str(e), f"message does not mention {needle!r}: {e}"
            return
        raise AssertionError(f"expected {exc.__name__} mentioning {needle!r}")

    wrong_cols = tmp / "wrong.csv"
    wrong_cols.write_text("x,y,z\n1,2,3\n")
    expect("axis-0", lambda: landmark_io.read_landmark_csv(wrong_cols))

    empty = tmp / "empty.csv"
    empty.write_text("index,axis-0,axis-1,axis-2\n")
    expect("no rows", lambda: landmark_io.read_landmark_csv(empty))

    nan = tmp / "nan.csv"
    nan.write_text("index,axis-0,axis-1,axis-2\n0,1,2,3\n1,4,,6\n")
    expect("row(s) [1]", lambda: landmark_io.read_landmark_csv(nan))

    expect("not found", lambda: landmark_io.read_landmark_csv(tmp / "nope.csv"),
           exc=FileNotFoundError)

    # The check that catches "these points were placed on a different image".
    ok = tmp / "ok.csv"
    landmark_io.write_landmark_csv(ok, np.array([[1.0, 2.0, 3.0], [40.0, 5.0, 6.0]]))
    landmark_io.read_landmark_csv(ok, shape_zyx=(50, 50, 50))       # inside: fine
    expect("outside", lambda: landmark_io.read_landmark_csv(ok, shape_zyx=(10, 10, 10)))
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
    landmark_io.write_landmark_csv(csv, voxels)
    pts = ev.load_points(csv)
    assert np.allclose(pts, voxels[:, ::-1]), f"load_points changed behaviour: {pts}"
    print("   OK (unchanged by the move into shared.landmark_io)")


# =====================================================================================
# shared.local_config
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
    print("4. local_config.values_from_config (the --no-form path)...")
    cfg = {"image_path": "/a/img.nii.gz", "output_csv": "/a/out.csv"}
    values = local_config.values_from_config("dummy", _FIELDS, cfg)
    assert values["image_path"] == "/a/img.nii.gz"
    assert values["levels"] == 4, "a field's own default should fill in"
    assert values["use_alt"] is False
    # enabled_when not satisfied -> None, exactly like a greyed-out form field,
    # and NOT a missing-value error.
    assert values["alt_path"] is None, values["alt_path"]
    assert values["note"] is None, "optional fields may stay empty"

    values = local_config.values_from_config(
        "dummy", _FIELDS, {**cfg, "use_alt": True, "alt_path": "/a/alt.nii.gz"})
    assert values["alt_path"] == "/a/alt.nii.gz", "enabled field should be kept"

    def expect(needle, fn, exc=ValueError):
        try:
            fn()
        except exc as e:
            assert needle in str(e), f"message does not mention {needle!r}: {e}"
            return
        raise AssertionError(f"expected {exc.__name__} mentioning {needle!r}")

    expect("output_csv", lambda: local_config.values_from_config(
        "dummy", _FIELDS, {"image_path": "/a/img.nii.gz"}))
    # A required field that is enabled only conditionally must be demanded too.
    expect("alt_path", lambda: local_config.values_from_config(
        "dummy", _FIELDS, {**cfg, "use_alt": True}))
    # --no-form with no config at all is an error, not an empty run.
    expect("找不到配置文件", lambda: local_config.values_from_config("dummy", _FIELDS, {}),
           exc=FileNotFoundError)
    print("   OK")


def test_resolve_inputs_no_form(tmp):
    print("5. local_config.resolve_inputs(no_form=True) reads an explicit config...")
    cfg_path = _write_cfg(tmp / "tool.yaml", image_path="~/img.nii.gz",
                          output_csv="${HOME}/out.csv", levels=6)
    values = local_config.resolve_inputs("dummy", "Dummy", _FIELDS,
                                         cli_path=cfg_path, no_form=True)
    home = str(Path.home())
    assert values["image_path"] == f"{home}/img.nii.gz", values["image_path"]
    assert values["output_csv"] == f"{home}/out.csv", "${VAR} should expand"
    assert values["levels"] == 6, "config must win over the field default"

    try:
        local_config.resolve_inputs("dummy", "Dummy", _FIELDS,
                                    cli_path=tmp / "missing.yaml", no_form=True)
    except FileNotFoundError as e:
        assert "不存在" in str(e), e
    else:
        raise AssertionError("an explicitly named missing config must be an error")
    print("   OK (no Qt import on this path)")


# =====================================================================================
# shared.atlas_reference.check_label_dtype
# =====================================================================================
def test_check_label_dtype(tmp, capture=None):
    print("6. atlas_reference: a float32 annotation is called out, uint32 is not...")
    limit = atlas_reference.LOSSLESS_INT_LIMIT

    # The real shape of the bug: ids on both sides of the float32 integer
    # limit. 484682516 is `corpus callosum, body`, the id that actually went
    # missing on this data.
    ids = np.array([0, 776, 1009, 484682516, 614454277], dtype=np.uint32)
    assert atlas_reference.check_label_dtype(np.uint32, ids, "fixed.tif"), \
        "uint32 holds every CCF id and must not warn"
    assert not atlas_reference.check_label_dtype(np.float32, ids, "stale.tif"), \
        "float32 cannot hold ids past 2**24 and must warn"

    # ...and it is the DTYPE that decides, not whether these particular ids
    # happen to be large: a float32 annotation of a small-id atlas is still
    # the wrong container, and one whose ids all fit is still worth saying so
    # about, just without the "N labels above the line" detail.
    small = np.array([0, 1, 2, 997], dtype=np.uint32)
    assert not atlas_reference.check_label_dtype(np.float32, small, "small.tif"), \
        "the dtype decides, not the ids that happen to be present"

    # Guard the constant itself -- it is the whole basis of the check.
    assert float(np.float32(limit)) == limit, "2**24 must be exactly representable"
    assert float(np.float32(limit + 1)) != limit + 1, \
        f"{limit}+1 must NOT be representable, or the limit is wrong"
    print("   OK")



def test_qc_reposition_audit(tmp):
    print("7. qc_guide_mask.audit_reposition: numbers, warnings and a before/after strip...")
    import numpy as np
    import SimpleITK as sitk
    import tifffile
    from registration_ants import reposition as rp
    import tools.qc_guide_mask as qc

    d = tmp / "qcrep"
    d.mkdir(exist_ok=True)
    stack = np.full((10, 120, 140), 50, dtype=np.uint16)
    stack[:, 60:110, 15:130] = 3000                 # the brain
    stack[2:8, 15:45, 40:100] = 3000                # the flap
    tifffile.imwrite(str(d / "raw.tif"), stack)

    # SPARSE outline: grabbed on 2 and 7 only, as the GUI exports it.
    frag = np.zeros(stack.shape, dtype=np.uint8)
    frag[2, 15:45, 40:100] = 1
    frag[7, 15:45, 40:100] = 1
    sitk.WriteImage(sitk.GetImageFromArray(frag), str(d / "frag.nii.gz"))

    plan = rp.make_plan(
        stack.shape, (2.6, 2.6, 32.0),
        [rp.make_fragment(1, [rp.make_keyframe(2, tx_um=0.0),
                              rp.make_keyframe(7, tx_um=26.0, ty_um=52.0, theta_deg=6.0)],
                          "flap A")],
        labels_path=str(d / "frag.nii.gz"))
    rp.write_plan(d / "plan.json", plan)
    cfg = {"sample": {"raw_tiff": str(d / "raw.tif"),
                      "reposition_plan": str(d / "plan.json"),
                      "voxel_size_um": [2.6, 2.6, 32.0]}}

    written = qc.audit_reposition(cfg, d, n_planes=6)
    assert len(written) == 1, written
    out = Path(written[0])
    assert out.exists() and out.stat().st_size > 5000, out

    # The audit must see the FILLED span, not just the two grabbed planes --
    # that is the whole point of exporting sparse and filling on use.
    dense = rp.densify_fragments(qc.load_painted_mask(d / "frag.nii.gz"))
    assert sorted(int(z) for z in np.unique(np.nonzero(dense == 1)[0])) == [2, 3, 4, 5, 6, 7]

    # A plan whose outline volume has no voxels for its label is reported, not
    # crashed on: the two files are edited separately and can drift apart.
    empty = np.zeros(stack.shape, dtype=np.uint8)
    sitk.WriteImage(sitk.GetImageFromArray(empty), str(d / "empty.nii.gz"))
    rp.write_plan(d / "orphan.json", dict(plan, labels_path=str(d / "empty.nii.gz")))
    cfg2 = dict(cfg, sample=dict(cfg["sample"], reposition_plan=str(d / "orphan.json")))
    assert qc.audit_reposition(cfg2, d, n_planes=6) == []

    # No plan at all is simply nothing to do.
    assert qc.audit_reposition({"sample": {"raw_tiff": str(d / "raw.tif")}}, d) == []
    print("   OK")


def main():
    print("=== tests/test_tool_inputs_smoke.py ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_landmark_csv_roundtrip(tmp)
        test_landmark_csv_rejects_bad_files(tmp)
        test_load_points_agrees_with_landmark_io(tmp)
        test_values_from_config(tmp)
        test_resolve_inputs_no_form(tmp)
        test_check_label_dtype(tmp)
        test_qc_reposition_audit(tmp)
    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
