import tempfile
import unittest
from pathlib import Path

from dataloader.utils.common import load_aligned_restoration_mappings


class DatasetMappingsTest(unittest.TestCase):
    def test_diffusion_mapping_requires_and_filters_precomputed_lq(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            gt_path = root_path / "gt"
            lq_path = root_path / "lq"
            lol_path = root_path / "lol"
            for path in (gt_path, lq_path, lol_path):
                path.mkdir()

            (gt_path / "matched.png").touch()
            (gt_path / "missing_lq.png").touch()
            (lq_path / "matched.jpg").touch()
            (lol_path / "matched.png").touch()
            (lol_path / "missing_lq.png").touch()

            config = {
                "datasets": [{
                    "name": "example",
                    "gt_image_path": str(gt_path),
                    "lq_image_path": str(lq_path),
                    "lol_gt_path": str(lol_path),
                }]
            }
            mappings = load_aligned_restoration_mappings(config, require_lq=True)

            self.assertEqual([mapping["basename"] for mapping in mappings], ["matched"])
            self.assertEqual(mappings[0]["lq_image_path"], str(lq_path / "matched.jpg"))

    def test_diffusion_mapping_rejects_missing_lq_path(self):
        config = {
            "datasets": [{
                "name": "example",
                "gt_image_path": "/tmp/gt",
                "lol_gt_path": "/tmp/low",
            }]
        }

        with self.assertRaisesRegex(ValueError, "requires lq_image_path"):
            load_aligned_restoration_mappings(config, require_lq=True)


if __name__ == "__main__":
    unittest.main()
