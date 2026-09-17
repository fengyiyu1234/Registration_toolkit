"""Smoke tests for qc/region_cells.py + qc/view_region_cells.py (headless).

    conda activate antsreg
    python tests/test_region_qc_smoke.py

Everything is synthetic: an ontology, a 20 um label volume, a registration-
grid tiff whose every voxel encodes its own index, and a run directory in
registration_ants' layout.  The frame tests matter most -- a box cut one
slice off still looks like tissue.

Needs numpy, pandas, nibabel, tifffile, scipy, matplotlib, registration_ants.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qc import region_cells as rc  # noqa: E402

CELL_UM = np.array([0.65, 0.65, 8.0])
IMG_UM = np.array([2.6, 2.6, 32.0])
LAB_SHAPE = (20, 20, 10)          # xyz, 20 um -> 400 x 400 x 200 um
IMG_SHAPE_ZYX = (7, 154, 154)


def encode(ix, iy, iz):
    return (np.asarray(ix) * 7 + np.asarray(iy) * 13 + np.asarray(iz) * 31) % 60000


def build_run(root, vol=None):
    import nibabel as nib
    import tifffile
    import yaml

    ontology = {"msg": [{"id": 1, "name": "root", "acronym": "root", "children": [
        {"id": 10, "name": "Cortex", "acronym": "CTX", "children": [
            {"id": 11, "name": "Layer one", "acronym": "L1"}]},
        {"id": 20, "name": "Other", "acronym": "OT"}]}]}
    (root / "ontology.json").write_text(json.dumps(ontology))

    labels = np.full(LAB_SHAPE, 20, np.uint32)
    labels[:10] = 11
    run = root / "run"
    run.mkdir()
    # ANTs writes LPS; nibabel reports RAS -> first two axes negative
    nib.save(nib.Nifti1Image(labels, np.diag([-20.0, -20.0, 20.0, 1.0])),
             str(run / "demo_labels_in_sample.nii.gz"))

    if vol is None:
        iz, iy, ix = np.meshgrid(*[np.arange(n) for n in IMG_SHAPE_ZYX], indexing="ij")
        vol = encode(ix, iy, iz).astype(np.uint16)
    tifffile.imwrite(str(root / "reg.tif"), vol)

    (run / "demo.yaml").write_text(yaml.safe_dump({
        "sample": {"name": "demo", "raw_tiff": str(root / "reg.tif"),
                   "voxel_size_um": IMG_UM.tolist()},
        "cells": {"voxel_size_um": CELL_UM.tolist()},
    }))

    rng = np.random.default_rng(0)
    for cls in ("neuron_RFP", "neuron_GFP", "glia_GFP_Sox9"):
        n = 400
        phys = np.column_stack([rng.uniform(5, 385, n), rng.uniform(5, 385, n),
                                rng.choice(np.arange(1, 24) * 8.0, n)])
        px = phys / CELL_UM
        lab_idx = np.clip(np.rint(phys / 20).astype(int), 0, np.array(LAB_SHAPE) - 1)
        ids = labels[lab_idx[:, 0], lab_idx[:, 1], lab_idx[:, 2]]
        df = pd.DataFrame({
            0: px[:, 0], 1: px[:, 1], 2: px[:, 2],
            3: phys[:, 0] / 20, 4: phys[:, 1] / 20, 5: phys[:, 2] / 20,
            6: 0.0, 7: 0.0, 8: 0.0,
            9: ids, 10: ["Layer one" if i == 11 else "Other" for i in ids],
            11: "s", 12: np.where(phys[:, 0] < 200, "tileA", "tileB"), 13: 0.5,
        })
        d = run / "cell_registration" / cls
        d.mkdir(parents=True)
        df.to_csv(d / "cell_registration.csv", header=False, index=False)
    return run


def make_cfg(root, **kw):
    cfg = {"run_dir": str(root / "run"), "ontology_json": str(root / "ontology.json"),
           "regions": ["CTX"], "out_dir": str(root / "out"), "n_sites": 5,
           "half_extent_um": [60, 60, 40], "min_separation_um": 10}
    cfg.update(kw)
    return cfg


def test_region_resolution():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_run(root)
        from registration_ants.atlas_utils import load_ccf_ontology_json
        st = load_ccf_ontology_json(root / "ontology.json")
        assert rc.find_structure(st, "Cortex") == 10
        assert rc.find_structure(st, "l1") == 11
        assert rc.find_structure(st, 20) == 20
        ids, _ = rc.resolve_region_ids(st, ["CTX"])
        assert ids == {10, 11}, ids
        try:
            rc.find_structure(st, "Cortx")
        except ValueError as e:
            assert "Cortex" in str(e)
        else:
            raise AssertionError("unknown name must raise")


def test_session_selection_and_depth():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_run(root)
        s = rc.Session(make_cfg(root), log=lambda *a: None)
        assert (s.selected["region_id"] == 11).all()
        assert len(s.selected) == int((s.cells["region_id"] == 11).sum())
        assert s.selected["lookup_agrees"].all()
        assert (s.selected["depth_um"] > 0).all()
        # a cell 100 um from the x boundary (voxel 5 of a region ending at 10),
        # farther than that from every other edge
        d = rc.signed_depth_um(s.labels, s.region_ids, np.array([[100.0, 200.0, 80.0]]))
        assert abs(d[0] - 100) < 1e-6, d
        d = rc.signed_depth_um(s.labels, s.region_ids, np.array([[260.0, 200.0, 80.0]]))
        assert abs(d[0] + 80) < 1e-6, d          # voxel 13 vs last region voxel 9
        # class filter + sites spread over classes
        s2 = rc.Session(make_cfg(root, classes=["*GFP*"], n_sites=4), log=lambda *a: None)
        assert set(s2.sites["class_name"]) == {"neuron_GFP", "glia_GFP_Sox9"}
        # pinned sites
        want = list(s.selected["cell_id"].iloc[:2])
        s3 = rc.Session(make_cfg(root, sites=want), log=lambda *a: None)
        assert list(s3.sites["cell_id"]) == want


def test_wrong_cell_voxel_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_run(root)
        try:
            rc.Session(make_cfg(root, cell_voxel_um=[2.6, 2.6, 32.0]), log=lambda *a: None)
        except ValueError as e:
            assert "第 3-5 列" in str(e)
        else:
            raise AssertionError("a voxel size that does not reproduce columns 3-5 must raise")


def test_volume_cut_matches_frame():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_run(root)
        s = rc.Session(make_cfg(root), log=lambda *a: None)
        for k in range(len(s.sites)):
            cut = s.cut_site(k)
            box = cut["box"]
            arr = next(iter(box.arrays.values()))
            o = np.rint(box.origin_um / IMG_UM).astype(int)
            Z, Y, X = arr.shape
            iz, iy, ix = np.meshgrid(np.arange(Z) + o[2], np.arange(Y) + o[1],
                                     np.arange(X) + o[0], indexing="ij")
            inside = ((ix >= 0) & (ix < IMG_SHAPE_ZYX[2]) & (iy >= 0) & (iy < IMG_SHAPE_ZYX[1])
                      & (iz >= 0) & (iz < IMG_SHAPE_ZYX[0]))
            assert np.array_equal(arr[inside], encode(ix, iy, iz)[inside].astype(np.uint16))
            assert (arr[~inside] == 0).all()
            # the site's own cell is inside its box, flagged as the target
            assert cut["cells"]["is_target"].sum() == 1
            # label box: origin on the 20 um grid and values match lookup
            lab, lo = cut["labels"], cut["labels_origin"]
            centres = lo + np.argwhere(np.ones(lab.shape, bool)) * 20.0
            assert np.array_equal(lab.ravel(), s.labels.lookup(centres))


def test_frame_scan_finds_injected_offset():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = build_run(root)
        cells = rc.load_cells(run)
        phys = cells[["x", "y", "z"]].to_numpy(float) * CELL_UM
        vol = np.full(IMG_SHAPE_ZYX, 100, np.uint16)
        rfp = cells["class_name"].str.contains("RFP").to_numpy()
        k = np.rint((phys[rfp] + (0, 0, -40.0)) / IMG_UM).astype(int)
        ok = np.all((k >= 0) & (k < IMG_SHAPE_ZYX[::-1]), axis=1)
        vol[k[ok, 2], k[ok, 1], k[ok, 0]] = 1000
        out = rc.frame_offset_scan(cells, phys, vol, IMG_UM, "RFP",
                                   dz_um=np.arange(-80, 41, 8), min_cells=50)
        assert len(out) == 1
        assert out.iloc[0]["best_dz_um"] == -40, out
        tiles = rc.frame_offset_scan(cells, phys, vol, IMG_UM, "RFP",
                                     dz_um=np.arange(-80, 41, 8), by_tile=True, min_cells=50)
        assert set(tiles["group"]) == {"tileA", "tileB"}
        assert (tiles["best_dz_um"] == -40).all(), tiles


def test_snapshot_and_verdicts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_run(root)
        from qc import view_region_cells as vrc
        s = rc.Session(make_cfg(root, n_sites=2), log=lambda *a: None)
        vrc.run_snapshots(s)
        pngs = sorted((root / "out" / "snapshots").glob("*.png"))
        assert len(pngs) == 2 and all(p.stat().st_size > 1000 for p in pngs)
        assert (root / "out" / "snapshots" / "sites.csv").exists()

        store = rc.VerdictStore(root / "out" / "verdicts.csv", s.run_dir)
        row = s.sites.iloc[0]
        store.set(row, "ok", "", "volume")
        store.set(row, "wrong_region", "outside by a layer", "volume")
        again = rc.VerdictStore(root / "out" / "verdicts.csv", s.run_dir)
        assert again.get(row["cell_id"]) == ("wrong_region", "outside by a layer")
        assert again.tally() == {"wrong_region": 1}
        other = rc.VerdictStore(root / "out" / "verdicts.csv", "/elsewhere")
        assert other.get(row["cell_id"]) == (None, "")


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    main()
