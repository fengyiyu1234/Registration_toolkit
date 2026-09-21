"""Smoke tests for qc/: the tile-assembly coordinate math, crop site picking,
and the scoring arithmetic.

Same manual-assert style as the rest of tests/, no pytest, headless -- nothing
here opens a window, so it runs over ssh:

    conda activate antsreg
    python tests/test_qc_crops_smoke.py

The first test is the one that matters.  The real tiles live on another
machine, so the global-frame convention that qc/crop_geometry.TileGrid
implements cannot be checked against real data here.  Instead a synthetic
mosaic is built whose every pixel value is a function of its GLOBAL
coordinate; cutting a box and reading the values back therefore verifies the
x, y and z conventions at once, including the z base
(local_z_0idx = global_z - 1 + tile_z0) that has no second chance to be
noticed if it is wrong -- a crop cut one slice off still looks like tissue.

Needs numpy, tifffile, scipy.
"""
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qc import crop_geometry as geom  # noqa: E402
from qc import score_crops  # noqa: E402

TILE = 64
OVERLAP = 16
N_SLICES = 20


def encode(gx, gy, gz):
    """A value that depends on all three global coordinates, so a cut that is
    off on any single axis cannot accidentally pass."""
    return (np.asarray(gx) * 7 + np.asarray(gy) * 13 + np.asarray(gz) * 31) % 60000


def build_mosaic(root):
    """2x2 tiles with a 16 px overlap and two different ABS_D, written so that
    every pixel holds encode(global coords)."""
    import tifffile

    step = TILE - OVERLAP
    stacks = [
        {"name": "000_000", "H": 0, "V": 0, "D": 0, "ROW": 0, "COL": 0},
        {"name": "000_001", "H": step, "V": 0, "D": 5, "ROW": 0, "COL": 1},
        {"name": "001_000", "H": 0, "V": step, "D": 5, "ROW": 1, "COL": 0},
        {"name": "001_001", "H": step, "V": step, "D": 0, "ROW": 1, "COL": 1},
    ]
    z_start = max(s["D"] for s in stacks)
    x_min = min(s["H"] for s in stacks)
    y_min = min(s["V"] for s in stacks)

    xml = ET.Element("TeraStitcher")
    ET.SubElement(xml, "dimensions", stack_rows="2", stack_columns="2",
                  stack_slices=str(N_SLICES))
    stacks_el = ET.SubElement(xml, "STACKS")
    for s in stacks:
        ET.SubElement(stacks_el, "Stack", DIR_NAME=s["name"],
                      ABS_H=str(s["H"]), ABS_V=str(s["V"]), ABS_D=str(s["D"]),
                      ROW=str(s["ROW"]), COL=str(s["COL"]))
        tile_dir = root / s["name"]
        tile_dir.mkdir(parents=True, exist_ok=True)

        x0, y0 = s["H"] - x_min, s["V"] - y_min
        z0 = z_start - s["D"]
        gy, gx = np.meshgrid(np.arange(TILE) + y0, np.arange(TILE) + x0,
                             indexing="ij")
        for local_z in range(N_SLICES):
            global_z = local_z + 1 - z0
            plane = encode(gx, gy, global_z).astype(np.uint16)
            tifffile.imwrite(str(tile_dir / f"slice_{local_z:04d}.tif"), plane)

    ET.ElementTree(xml).write(root / "xml_merging.xml")


def test_tile_grid_cut_matches_global_frame():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_mosaic(root)
        grid = geom.TileGrid(root)
        assert grid.probe_tile_shape() == (TILE, TILE), grid.probe_tile_shape()

        origin = (30, 30, 3)          # spans all four tiles
        size = (40, 40, 6)            # global z 3..8, valid in every tile
        vol = grid.cut(origin, size)
        assert vol.shape == (size[2], size[1], size[0]), vol.shape

        gz, gy, gx = np.meshgrid(
            np.arange(size[2]) + origin[2],
            np.arange(size[1]) + origin[1],
            np.arange(size[0]) + origin[0], indexing="ij")
        expected = encode(gx, gy, gz).astype(np.uint16)
        bad = int((vol != expected).sum())
        assert bad == 0, f"{bad}/{vol.size} 个体素和全局坐标编码对不上"

        # A one-slice z error is the failure this is really guarding against,
        # so make sure the test could actually see one.
        shifted = grid.cut((origin[0], origin[1], origin[2] + 1), size)
        assert (shifted != expected).any(), "z 方向平移一层竟然没有差别，测试是空的"
    print("  ok  TileGrid.cut reproduces the global frame on x, y and z")


def test_source_slice_location():
    import tifffile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_mosaic(root)
        grid = geom.TileGrid(root)
        hits = grid.locate([50, 50, 3])
        assert len(hits) == 4  # overlap: retain every source, not an arbitrary tile
        for hit in hits:
            x, y, z = hit["local_xyz"]
            assert tifffile.imread(hit["path"])[int(y), int(x)] == encode(50, 50, 3)
            assert Path(hit["path"]).name == f"slice_{z:04d}.tif"
        shifted = geom.TileGrid(root, offset_px=(2, 0, 1)).locate([50, 50, 3])
        for hit in shifted:
            x, y, _ = hit["local_xyz"]
            assert tifffile.imread(hit["path"])[int(y), int(x)] == encode(52, 50, 4)
        assert not grid.locate([-100, 0, 3])
        assert not grid.locate([50, 50, 999])
    print("  ok  source paths and local pixels reproduce global coordinates")


def test_offset_px_shifts_the_read():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_mosaic(root)
        plain = geom.TileGrid(root).cut((30, 30, 3), (20, 20, 4))
        shifted = geom.TileGrid(root, offset_px=(2, 0, 0)).cut((30, 30, 3), (20, 20, 4))
        assert (plain[:, :, 2:] == shifted[:, :, :-2]).all(), \
            "offset_px 没有按 x 平移读取窗口"
    print("  ok  offset_px shifts which box is read")


def test_frame_conversion_round_trip():
    cell_um, label_um = (0.65, 0.65, 8.0), (20.0, 20.0, 20.0)
    px = np.array([8000.0, 12000.0, 400.0])
    vox = geom.global_px_to_label_voxel(px, cell_um, label_um)
    back = geom.label_voxel_to_global_px(vox, cell_um, label_um)
    assert np.allclose(px, back), (px, back)
    # The documented scale factors, spelled out so a change to either is loud.
    assert np.allclose(vox / px, [0.0325, 0.0325, 0.4]), vox / px
    print("  ok  global pixel <-> label voxel is the documented scalar multiply")


def test_pick_crop_sites_stays_inside_the_region():
    labels = np.zeros((60, 60, 60), dtype=np.int64)
    labels[10:50, 10:50, 6:54] = 315          # a slab of "Isocortex"
    cell_um, label_um = (0.65, 0.65, 8.0), (20.0, 20.0, 20.0)
    crop_px = np.array([200, 200, 50])

    sites = geom.pick_crop_sites(labels, {315}, crop_px, cell_um, label_um,
                                 n_crops=6, n_z_bands=3,
                                 rng=np.random.default_rng(0))
    assert len(sites) > 0, "一个 crop 都没选出来"
    for site in sites:
        lo = geom.global_px_to_label_voxel(site["origin_px"], cell_um, label_um)
        hi = geom.global_px_to_label_voxel(
            np.asarray(site["origin_px"]) + crop_px, cell_um, label_um)
        corners = np.stack(np.meshgrid(*zip(lo, hi), indexing="ij"), -1).reshape(-1, 3)
        for corner in np.round(corners).astype(int):
            assert labels[tuple(np.clip(corner, 0, np.array(labels.shape) - 1))] == 315, \
                f"crop 角点落到区外: {corner}"
        assert site["region_id"] == 315
    assert len({s["z_band"] for s in sites}) > 1, "所有 crop 落在同一个深度带"
    print(f"  ok  pick_crop_sites kept all {len(sites)} crops inside the region")


def test_descendant_ids_and_unknown_name():
    """Both dict shapes: structure_id_path, which is what
    registration_ants.atlas_utils.load_ccf_ontology_json really writes, and
    parent_structure_id as the fallback."""
    by_path = {
        997: {"name": "root", "structure_id_path": [997]},
        315: {"name": "Isocortex", "structure_id_path": [997, 315]},
        985: {"name": "MOp", "structure_id_path": [997, 315, 985]},
        320: {"name": "MOp1", "structure_id_path": [997, 315, 985, 320]},
        512: {"name": "Cerebellum", "structure_id_path": [997, 512]},
    }
    assert geom.descendant_ids(by_path, "Isocortex") == {315, 985, 320}
    assert geom.descendant_ids(by_path, "root") == set(by_path)

    by_parent = {
        1: {"name": "root", "parent_structure_id": None},
        2: {"name": "Isocortex", "parent_structure_id": 1},
        3: {"name": "MOp", "parent_structure_id": 2},
        4: {"name": "MOp1", "parent_structure_id": 3},
        5: {"name": "Cerebellum", "parent_structure_id": 1},
    }
    assert geom.descendant_ids(by_parent, "Isocortex") == {2, 3, 4}

    try:
        geom.descendant_ids(by_path, "Isocortx")
    except ValueError:
        pass
    else:
        raise AssertionError("拼错的脑区名应该报错，不能静默返回空集")
    print("  ok  descendant_ids handles both ontology dict shapes")


def test_signal_check_separates_aligned_from_misaligned():
    rng = np.random.default_rng(0)
    vol = np.full((20, 100, 100), 100.0)
    pts = np.array([[20, 30, 5], [60, 70, 10], [80, 20, 15]], float)
    for x, y, z in pts.astype(int):
        vol[z - 1:z + 2, y - 4:y + 5, x - 4:x + 5] = 4000.0
    good, n = geom.signal_check(vol, pts, rng)
    bad, _ = geom.signal_check(vol, pts + np.array([40, 40, 0]), rng)
    assert n == 3 and good > 5, (good, n)
    assert bad < 2, bad
    print(f"  ok  signal_check: aligned {good:.1f}x vs misaligned {bad:.1f}x")


def test_match_and_score():
    voxel = (0.65, 0.65, 8.0)
    ann = [{"call": "sox9_pos", "lx": 10, "ly": 10, "lz": 5},
           {"call": "sox9_neg", "lx": 50, "ly": 50, "lz": 5},
           {"call": "sox9_pos", "lx": 90, "ly": 10, "lz": 5},   # pipeline missed Sox9
           {"call": "uncertain", "lx": 20, "ly": 80, "lz": 5},
           {"call": "sox9_neg", "lx": 70, "ly": 70, "lz": 5}]   # soma missed entirely
    preds = [{"lx": 11, "ly": 10, "lz": 5, "class_name": "glia_GFP_Sox9"},
             {"lx": 50, "ly": 51, "lz": 5, "class_name": "neuron_GFP"},
             {"lx": 90, "ly": 10, "lz": 5, "class_name": "neuron_RFP"},
             {"lx": 20, "ly": 80, "lz": 5, "class_name": "neuron_GFP"},
             {"lx": 5, "ly": 95, "lz": 5, "class_name": "neuron_RFP"}]   # false positive
    record = {"predictions": preds, "voxel_um_xyz": list(voxel)}

    pairs, lone_a, lone_b = score_crops.match_points(ann, preds, voxel, radius_um=12.0)
    assert len(pairs) == 4, pairs
    assert lone_a == [4] and lone_b == [4], (lone_a, lone_b)

    with tempfile.TemporaryDirectory() as tmp:
        crop = Path(tmp)
        (crop / "annotation.csv").write_text(
            "crop_id,call,lx,ly,lz\n" + "".join(
                f"c,{a['call']},{a['lx']},{a['ly']},{a['lz']}\n" for a in ann),
            encoding="utf-8")
        counts = score_crops.score_crop(record, crop, radius_um=12.0)

    assert counts["tp"] == 3 and counts["fn"] == 1 and counts["fp"] == 1, counts
    assert counts["uncertain"] == 1, counts
    assert counts["s_tp"] == 1 and counts["s_fn"] == 1 and counts["s_tn"] == 1, counts

    rates = score_crops.pooled_rates([counts])
    assert abs(rates["soma_recall"] - 0.75) < 1e-9, rates
    assert abs(rates["sox9_sensitivity"] - 0.5) < 1e-9, rates
    assert abs(rates["uncertain_rate"] - 0.2) < 1e-9, rates
    print("  ok  matching and the confusion matrix, including uncertain handling")


def test_bootstrap_is_over_crops_not_cells():
    """Ten identical crops must give a zero-width interval; the same cells in
    one crop must not.  If the CI were pooled over cells this would not hold,
    and a group difference driven by one crop would look decisive."""
    crop = dict(tp=40, fp=2, fn=10, uncertain=0, s_tp=20, s_fn=20, s_fp=0, s_tn=0)
    rng = np.random.default_rng(0)
    lo, hi = score_crops.bootstrap_ci([dict(crop) for _ in range(10)],
                                      "sox9_sensitivity", rng, n=200)
    assert abs(hi - lo) < 1e-9, (lo, hi)

    varied = [dict(crop, s_tp=k, s_fn=40 - k) for k in (2, 10, 20, 30, 38)]
    lo2, hi2 = score_crops.bootstrap_ci(varied, "sox9_sensitivity", rng, n=2000)
    assert hi2 - lo2 > 0.15, (lo2, hi2)
    print(f"  ok  cluster bootstrap widens with between-crop spread "
          f"({(hi2 - lo2) * 100:.0f}% wide)")


if __name__ == "__main__":
    print("qc/ smoke tests")
    test_tile_grid_cut_matches_global_frame()
    test_source_slice_location()
    test_offset_px_shifts_the_read()
    test_frame_conversion_round_trip()
    test_pick_crop_sites_stays_inside_the_region()
    test_descendant_ids_and_unknown_name()
    test_signal_check_separates_aligned_from_misaligned()
    test_match_and_score()
    test_bootstrap_is_over_crops_not_cells()
    print("all passed")
