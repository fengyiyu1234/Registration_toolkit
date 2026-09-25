import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from registration_ants import section2d
from section_masks import auto_mask_raw
import section_masks


class SectionMaskTests(unittest.TestCase):
    def test_full_resolution_roundtrip_and_semantics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "source.tif"
            tifffile.imwrite(image, np.zeros((9, 12), np.uint16))
            ontology = root / "ontology.json"
            ontology.write_text("{}")
            tissue = np.zeros((9, 12), np.uint8)
            tissue[2:8, 2:10] = 1
            damage = np.zeros_like(tissue)
            damage[4:6, 4:6] = 1
            regions = np.zeros((9, 12), np.uint32)
            regions[2:4, 2:5] = 1
            regions[6:8, 7:10] = 2
            assignments = {1: {"region_ids": [10], "names": ["a"]},
                           2: {"region_ids": [20, 21], "names": ["b", "c"]}}
            paths, report = section_masks.save_session(root / "masks", "sample", image, 0.65,
                                                       0, "max", tissue, damage, regions,
                                                       assignments, ontology)
            self.assertEqual(report["damage_inside_tissue_pixels"], 4)
            self.assertEqual(tifffile.imread(paths["tissue"]).shape, (9, 12))
            self.assertEqual(tifffile.imread(paths["tissue"])[4, 4], 0)
            self.assertEqual(tifffile.imread(paths["damage"])[4, 4], 1)
            saved = section_masks.load_session(root / "masks", "sample", image, (9, 12),
                                               0.65, 0, "max", ontology)
            for actual, expected in zip(saved[:3], (tissue, damage, regions)):
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(saved[3], assignments)
            self.assertEqual(json.loads(paths["sidecar"].read_text())["shape_yx"], [9, 12])
            with self.assertRaisesRegex(ValueError, "channel mismatch"):
                section_masks.load_session(root / "masks", "sample", image, (9, 12),
                                           0.65, 1, "max", ontology)
            ontology.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "ontology_sha256 mismatch"):
                section_masks.load_session(root / "masks", "sample", image, (9, 12),
                                           0.65, 0, "max", ontology)

    def test_auto_prefill_maps_back_to_registration_grid(self):
        y, x = np.ogrid[:143, :187]
        raw = np.where((y - 71) ** 2 / 51 ** 2 + (x - 94) ** 2 / 68 ** 2 < 1,
                       1000, 1).astype(np.float32)
        pixel_size_um, target_um = 1.3, 12.0
        working = section2d.downsample_section(raw, pixel_size_um, target_um)
        expected = section2d.auto_tissue_mask(working)
        full = auto_mask_raw(raw, pixel_size_um, target_um)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tissue.tif"
            tifffile.imwrite(path, full)
            actual = section2d._load_mask_on(path, raw.shape, pixel_size_um, working)
        np.testing.assert_array_equal(actual, expected)

    def test_rejects_unassigned_and_misaligned(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "source.tif"
            image.write_bytes(b"source")
            tissue = np.ones((3, 4), np.uint8)
            damage = np.zeros_like(tissue)
            regions = np.zeros((3, 4), np.uint32)
            regions[0, 0] = 1
            with self.assertRaisesRegex(ValueError, "without ontology assignments"):
                section_masks.save_session(root, "s", image, 1, 0, "max", tissue, damage,
                                           regions, {1: {"region_ids": [], "names": []}}, image)
            with self.assertRaisesRegex(ValueError, "full image shape"):
                section_masks.compose_masks(tissue.shape, tissue, damage[:2], regions)
            damage[:] = 1
            with self.assertRaisesRegex(ValueError, "no tissue remains"):
                section_masks.compose_masks(tissue.shape, tissue, damage, regions)


if __name__ == "__main__":
    unittest.main()
