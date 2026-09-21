"""Smoke tests for qc/detection_boxes.py + qc/view_detection_qc.py (headless).

    conda activate antsreg
    python tests/test_detection_qc_smoke.py

Reuses tests/test_region_qc_smoke.py's synthetic run (ontology, 20 um label
volume, registration tiff, cell_registration/) and adds a synthetic
brain_detector result directory next to it.

The frame tests are the point.  Every stage of the detector writes boxes in a
slightly different frame, and a box read under the wrong one still lands on
real tissue and still looks like a cell -- so the asserts below pin the exact
arithmetic rather than "roughly the right place":

  * s2/s3/s4 global px -> microns is a plain multiply by cells.voxel_size_um
  * s1 tile-local -> global is  x + tile_x0  and  z - tile_z0
  * coloc_result.csv -> cell_registration.csv columns 0-2 is an EQUALITY,
    because run_inference.py writes cx = (x1+x2)/2 and cell_points.py copies
    it through untouched.  test_match_reconciliation checks both that an
    intact pair matches exactly and that a dropped row is reported.

Needs numpy, pandas, nibabel, tifffile, scipy, matplotlib, registration_ants.
"""
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from qc import detection_boxes as db  # noqa: E402
from test_region_qc_smoke import CELL_UM, build_run, make_cfg  # noqa: E402

GLOBAL_COLS = ["x1", "y1", "x2", "y2", "score", "mean", "class", "z"]


# ── Synthetic detector result directory ───────────────────────────────────────

def _boxes_from_cells(run, classes, jitter=0.0):
    """Rebuild detector-style boxes from the run's own cell table, so the two
    sides are guaranteed to describe the same cells and a mismatch in a test
    can only come from the code under test."""
    rows = []
    for cls in classes:
        df = pd.read_csv(run / "cell_registration" / cls / "cell_registration.csv",
                         header=None)
        cx, cy, z = df[0].to_numpy(float), df[1].to_numpy(float), df[2].to_numpy(float)
        half = 6.0
        rows.append(pd.DataFrame({
            "x1": cx - half + jitter, "y1": cy - half, "x2": cx + half + jitter,
            "y2": cy + half, "score": 0.5, "mean": 1000.0, "class": cls, "z": z,
        }))
    return pd.concat(rows, ignore_index=True)


def build_detection(root, run, drop_last=0):
    """A brain_detector pATHRESULT holding s2, s3 and s4 for the run's cells.

    s3 == the cells one-for-one; s2 repeats each cell on three consecutive z
    slices (what a real z-linker collapses), so the funnel's compression ratio
    has a known answer of 3.
    """
    det = root / "detection_results"
    classes = ["neuron_RFP", "neuron_GFP", "glia_GFP_Sox9"]
    s4 = _boxes_from_cells(run, classes)
    if drop_last:
        s4 = s4.iloc[:-drop_last]
    (det / "4_colocalization").mkdir(parents=True)
    s4[GLOBAL_COLS].to_csv(det / "4_colocalization" / "coloc_result.csv", index=False)

    # per-channel: a cell counts for every channel its class carries a marker for
    (det / "2_global_2d_raw").mkdir(parents=True)
    (det / "3_channel_3d").mkdir(parents=True)
    per_ch = {}
    for ch in ("RFP", "GFP", "Sox9"):
        sub = s4[s4["class"].map(lambda c, m=ch: m in db.class_markers(c))].copy()
        # the detector names single-channel boxes by that channel alone
        sub["class"] = sub["class"].map(lambda c, m=ch: f"{db.split_class(c)[0]}_{m}")
        sub[GLOBAL_COLS].to_csv(det / "3_channel_3d" / f"{ch}_3d_tracked.csv", index=False)
        spread = pd.concat([sub.assign(z=sub["z"] + dz) for dz in (-1, 0, 1)],
                           ignore_index=True)
        spread[GLOBAL_COLS].to_csv(det / "2_global_2d_raw" / f"{ch}_2d_global.csv",
                                   index=False)
        per_ch[ch] = sub
    return det, s4, per_ch


def build_tiles(root, tiles):
    """A channel directory with just a merging xml -- enough for TileGrid's
    offsets, which is all the s1 fallback needs."""
    ch = root / "tiles" / "RFP"
    ch.mkdir(parents=True)
    stacks = ET.Element("STACKS")
    for name, h, v, d in tiles:
        (ch / name).mkdir()
        ET.SubElement(stacks, "Stack", DIR_NAME=name, ABS_H=str(h), ABS_V=str(v),
                      ABS_D=str(d), ROW="0", COL="0")
    xml = ET.Element("TeraStitcher")
    xml.append(stacks)
    ET.ElementTree(xml).write(str(ch / "xml_merging.xml"))
    return ch


def det_cfg(root, **kw):
    cfg = make_cfg(root, source="volume", n_sites=3, half_extent_um=[60, 60, 40])
    cfg["detection_dir"] = str(root / "detection_results")
    cfg["detection_channels"] = ["RFP", "GFP", "Sox9"]
    cfg.update(kw)
    return cfg


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_class_parsing():
    assert db.split_class("neuron_GFP_Sox9") == ("neuron", ["GFP", "Sox9"])
    assert db.class_markers("glia_RFP") == ["RFP"]
    # the exposure-suffix artefact: 'GFP_3' must not become a marker called '3'
    assert db.class_markers("neuron_3_GFP_RFP") == ["GFP", "RFP"]
    # ... but a class whose markers are all digits keeps them rather than vanishing
    assert db.class_markers("neuron_3") == ["3"]
    groups = db.coloc_groups(["neuron_GFP", "glia_GFP_Sox9", "neuron_GFP_RFP_Sox9"])
    assert [g["name"] for g in groups] == ["GFP+Sox9", "GFP+RFP+Sox9"], groups
    assert groups[0]["markers"] == {"gfp", "sox9"}
    # neuron and glia get distinguishable shades of the same combination colour
    assert groups[0]["neuron_color"] != groups[0]["glia_color"]


def test_normalise_column_orders():
    """The per-tile files put class before score; the global ones after.  Both
    appear with and without a header row."""
    tile = pd.DataFrame([["s", 1, 2, 3, 4, "neuron_RFP", 0.5, 10, 7]],
                        columns=db.TILE_COLS)
    got = db._normalise(tile.rename(columns=lambda c: "?"))
    assert (got.iloc[0][["x1", "y1", "x2", "y2", "z"]].to_numpy() == [1, 2, 3, 4, 7]).all()
    assert got.iloc[0]["class"] == "neuron_RFP"
    glob = pd.DataFrame([[1, 2, 3, 4, 0.5, 10, "glia_GFP", 7]], columns=GLOBAL_COLS)
    got = db._normalise(glob)
    assert got.iloc[0]["class"] == "glia_GFP" and got.iloc[0]["z"] == 7


def test_global_frame_and_collect():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = build_run(root)
        det, s4, per_ch = build_detection(root, run)
        runner = db.DetectionRun(det, ["RFP", "GFP", "Sox9"], CELL_UM,
                                 stages=("s2", "s3", "s4"), log=lambda *a: None)
        assert not runner.warnings, runner.warnings

        # global px -> microns is a plain per-axis multiply, z included
        one = pd.DataFrame({"x1": [100.0], "y1": [200.0], "x2": [112.0], "y2": [212.0],
                            "z": [7.0], "class": ["glia_GFP_Sox9"]})
        p = runner.to_phys(one)
        assert p.iloc[0]["x1"] == 100 * CELL_UM[0]
        assert p.iloc[0]["y2"] == 212 * CELL_UM[1]
        assert p.iloc[0]["zc"] == 7 * CELL_UM[2]

        # a window around one known cell picks up that cell at every stage
        row = s4.iloc[0]
        c = np.array([(row.x1 + row.x2) / 2, (row.y1 + row.y2) / 2, row.z]) * CELL_UM
        half = np.array([30.0, 30.0, 24.0])
        got = runner.collect([(c - half, c + half)])
        assert ("s4", None) in got and ("s3", "RFP") in got and ("s2", "RFP") in got
        s4_hit = got[("s4", None)]
        assert (s4_hit["window"] == 0).all()
        assert row["class"] in set(s4_hit["class"])
        # the window is 3 slices deep and s2 repeats each cell on 3 slices,
        # so s2 must hold strictly more boxes than s3 here
        assert len(got[("s2", "RFP")]) > len(got[("s3", "RFP")])
        # nothing outside the window leaked in
        for df in got.values():
            assert (df["zc"] >= c[2] - half[2] - 1e-9).all()
            assert (df["zc"] <= c[2] + half[2] + 1e-9).all()


def test_tile_fallback_offsets():
    """No 2_global_2d_raw -> read 1_tile_2d_filtered and undo the tile offsets.

    tile_x0 = ABS_H - min(ABS_H), tile_z0 = max(ABS_D) - ABS_D, and the CSV's
    z is the tile-local 1-indexed slice, so global_z = z_csv - tile_z0.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ch_dir = build_tiles(root, [("tileA", 0, 0, 100), ("tileB", 500, 0, 40)])
        from qc.crop_geometry import TileGrid
        grid = TileGrid(ch_dir)
        by_name = {t["name"]: t for t in grid.tiles}
        assert by_name["tileB"]["x0"] == 500 and by_name["tileB"]["z0"] == 60

        det = root / "detection_results"
        (det / "1_tile_2d_filtered").mkdir(parents=True)
        for name in ("tileA", "tileB"):
            pd.DataFrame([["s", 10, 20, 22, 32, "neuron_RFP", 0.5, 1000, 5]],
                         columns=db.TILE_COLS).to_csv(
                det / "1_tile_2d_filtered" / f"{name}_RFP_result.csv", index=False)

        runner = db.DetectionRun(det, ["RFP"], CELL_UM, stages=("s2",),
                                 tile_grids={"RFP": grid}, log=lambda *a: None)
        assert any("1_tile_2d_filtered" in w for w in runner.warnings), runner.warnings
        got = pd.concat([c for _, _, c in runner.iter_boxes()], ignore_index=True)
        assert len(got) == 2, got
        rows = {float(r.x1): r for _, r in got.iterrows()}
        assert set(rows) == {10.0, 510.0}, rows           # x + tile_x0
        assert rows[10.0]["z"] == 5 - 0                   # tileA: z0 = 0
        assert rows[510.0]["z"] == 5 - 60                 # tileB: z0 = 60


def test_rasterize():
    """Boxes are drawn as outlines on their own slice: the border is set, the
    inside is not, and a box on another slice does not bleed through."""
    boxes = pd.DataFrame({
        "x1": [10.0], "y1": [20.0], "x2": [30.0], "y2": [40.0], "zc": [16.0],
        "class": ["glia_GFP_Sox9"],
    })
    vol, cmap = db.rasterize(boxes, origin_um=[0.0, 0.0, 0.0], voxel_um=[1.0, 1.0, 8.0],
                             shape_xyz=[60, 60, 4], outline_width=2)
    assert vol.shape == (4, 60, 60)
    assert len(cmap) == 1
    lid = next(iter(cmap))
    assert vol[2, 20, 10] == lid and vol[2, 40, 30] == lid      # corners, z = 16/8
    assert vol[2, 30, 20] == 0                                  # interior stays empty
    assert vol[1].sum() == 0 and vol[3].sum() == 0              # other slices untouched
    assert list(cmap[lid]) == db.CLASS_COLOR["glia"]

    # a box entirely outside the grid must be dropped, not clipped to the edge
    vol2, cmap2 = db.rasterize(boxes.assign(zc=999.0), [0.0, 0.0, 0.0],
                               [1.0, 1.0, 8.0], [60, 60, 4])
    assert vol2.sum() == 0 and not cmap2

    # dashed (glia) draws strictly fewer border pixels than solid
    dashed, _ = db.rasterize(boxes, [0.0, 0.0, 0.0], [1.0, 1.0, 8.0], [60, 60, 4],
                             dash_of=lambda r: 8)
    assert 0 < (dashed > 0).sum() < (vol > 0).sum()


def test_region_counts_and_funnel():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = build_run(root)
        build_detection(root, run)
        from qc.view_detection_qc import DetectionSession
        s = DetectionSession(det_cfg(root), log=lambda *a: None, load_boxes=False)
        counts, lines = s.funnel()

        assert set(counts["stage"]) == {"s2", "s3", "s4"}
        in_region = counts.groupby("stage")["n_in_region"].sum()
        # every cell is in the label volume, and only the CTX half is selected,
        # so the region holds some but not all of them
        assert 0 < in_region["s4"] < counts[counts.stage == "s4"]["n_total"].sum()
        # s2 repeats each box on 3 slices; the z jitter moves a few across the
        # region boundary, so this is a ratio check, not an equality
        ratio = in_region["s2"] / in_region["s3"]
        assert 2.5 < ratio < 3.5, ratio

        text = "\n".join(lines)
        assert "Sox9+ 占比" in text
        assert "完全一致" in text, text      # nothing was dropped, so s4 == cell table


def test_match_reconciliation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = build_run(root)
        det, _, _ = build_detection(root, run)
        runner = db.DetectionRun(det, ["RFP", "GFP", "Sox9"], CELL_UM,
                                 stages=("s4",), log=lambda *a: None)
        from qc import region_cells as rc
        cells = rc.load_cells(run)

        m = db.match_coloc_to_cells(runner, cells)
        assert m["available"] and m["n_matched"] == m["n_cells"] == len(cells)
        assert m["n_cells_unmatched"] == 0 and m["n_s4_unmatched"] == 0
        assert "✅" in "\n".join(db.match_lines(m))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = build_run(root)
        det, _, _ = build_detection(root, run, drop_last=7)
        runner = db.DetectionRun(det, ["RFP", "GFP", "Sox9"], CELL_UM,
                                 stages=("s4",), log=lambda *a: None)
        from qc import region_cells as rc
        cells = rc.load_cells(run)
        m = db.match_coloc_to_cells(runner, cells)
        assert m["n_cells_unmatched"] == 7, m
        assert "⚠️" in "\n".join(db.match_lines(m))

        # a shifted (not dropped) coloc file must also fail: same count, no match
        (root / "shift").mkdir()
        det2, _, _ = build_detection(root / "shift", build_run(root / "shift"))
        shifted = pd.read_csv(det2 / "4_colocalization" / "coloc_result.csv")
        shifted[["x1", "x2"]] += 1.0
        shifted.to_csv(det2 / "4_colocalization" / "coloc_result.csv", index=False)
        r2 = db.DetectionRun(det2, ["RFP"], CELL_UM, stages=("s4",), log=lambda *a: None)
        m2 = db.match_coloc_to_cells(r2, rc.load_cells(root / "shift" / "run"))
        assert m2["n_matched"] == 0 and m2["n_s4"] == m2["n_cells"], m2


def test_snapshot_and_verdicts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = build_run(root)
        build_detection(root, run)
        from qc import region_cells as rc
        from qc import view_detection_qc as vdq
        s = vdq.DetectionSession(det_cfg(root, n_sites=2), log=lambda *a: None)
        assert s.boxes, "site boxes should have been collected"
        layers = s.site_boxes(0)
        assert layers, "site 0 has no detection layers"
        assert any(n.startswith("[coloc]") for n, *_ in layers), [n for n, *_ in layers]

        vdq.run_snapshots(s)
        pngs = sorted((root / "out" / "snapshots").glob("*.png"))
        assert len(pngs) == 2 and all(p.stat().st_size > 1000 for p in pngs)
        cols = pd.read_csv(root / "out" / "snapshots" / "sites.csv").columns
        assert any(c.startswith("s4") for c in cols), list(cols)

        store = rc.VerdictStore(root / "out" / "verdicts.csv", s.run_dir,
                                vocab=vdq.VERDICTS)
        row = s.sites.iloc[0]
        store.set(row, "over_merge", "两个核一个框", "volume")
        again = rc.VerdictStore(root / "out" / "verdicts.csv", s.run_dir,
                                vocab=vdq.VERDICTS)
        assert again.get(row["cell_id"]) == ("over_merge", "两个核一个框")
        assert "该分没分" in again.tally_text()
        try:
            store.set(row, "wrong_region", "", "volume")   # region_qc's vocabulary
        except ValueError:
            pass
        else:
            raise AssertionError("a verdict outside this tool's vocabulary must raise")


def test_interactive_cell_location():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = build_run(root)
        build_detection(root, run)
        from qc.view_detection_qc import DetectionSession
        s = DetectionSession(det_cfg(root, n_sites=1), log=lambda *a: None)
        original = s.sites.iloc[0]["cell_id"]
        target = s.cells[~s.cells["cell_id"].isin(s.sites["cell_id"])].iloc[0]
        k, distance = s.locate_cell(target["cell_id"])
        assert k == 1 and distance == 0
        assert s.sites.iloc[0]["cell_id"] == original
        assert s.sites.iloc[k]["cell_id"] == target["cell_id"]
        assert any(len(frame) for _, frame, *_ in s.site_boxes(k))
        assert s.cut_site(k)["cells"]["is_target"].sum() == 1
        assert s.locate_cell(target["cell_id"]) == (k, 0.0)
        assert len(s.sites) == 2
        xyz = target[["x", "y", "z"]].to_numpy(float) + [0.1, 0, 0]
        found, distance = s.locate_cell(", ".join(map(str, xyz)))
        assert found == k
        assert np.isclose(distance, 0.1 * CELL_UM[0])
        for bad in ("unknown:999", "nan 0 0", "1 2", "inf 2 3"):
            try:
                s.locate_cell(bad)
            except ValueError:
                pass
            else:
                raise AssertionError(bad)
        assert len(s.sites) == 2


def test_direct_coordinate_target_keeps_requested_center():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = build_run(root)
        build_detection(root, run)
        from qc.view_detection_qc import DetectionSession
        s = DetectionSession(det_cfg(root, n_sites=1), log=lambda *a: None)
        xyz = np.array([12.5, 23.5, 3.0])
        k, distance = s.locate_coordinate(xyz)
        row = s.sites.iloc[k]
        assert row["target_kind"] == "coordinate"
        assert np.allclose(row[["x", "y", "z"]].to_numpy(float), xyz)
        cut = s.cut_site(k)
        assert np.allclose(cut["center"], xyz * CELL_UM)
        assert distance >= 0
        assert s.locate_coordinate(xyz)[0] == k
    print("ok  direct coordinate target keeps its own crop center")


def test_three_dimension_verdicts_share_legacy_csv():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = build_run(root)
        from qc import region_cells as rc
        s = rc.Session(make_cfg(root, n_sites=1), log=lambda *a: None)
        path = root / "out" / "verdicts.csv"
        store = rc.VerdictStore(path, s.run_dir)
        row = s.sites.iloc[0]
        store.set_dimensions(row, detection_verdict="correct",
                             colocalization_verdict="over_merge",
                             registration_verdict="boundary_uncertain",
                             note="three axes", source="tiles")
        again = rc.VerdictStore(path, s.run_dir)
        record = again.get_record(row["cell_id"])
        assert record["detection_verdict"] == "correct"
        assert record["colocalization_verdict"] == "over_merge"
        assert record["registration_verdict"] == "boundary_uncertain"
        assert again.get(row["cell_id"])[1] == "three axes"
        assert len(again.df) == 1
    print("ok  three-dimensional verdicts share legacy CSV")


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    main()
