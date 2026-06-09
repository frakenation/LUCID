import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


REMOVED_LOSS_TOKENS = (
    "GTmean",
    "gtmean",
    "L_color",
    "Lcolor",
    "MS_SWD",
    "MS_SWD_Loss",
    "L_spa",
    "L_exp",
    "L_TV",
    "SkipLoss",
    "compute_skip_loss",
    "WaveletLoss",
    "lambda_wavelet",
    "lambda_gtmean",
    "gtmean_sigma",
    "EA_DISTS",
    "ea_dists",
    "lambda_ea_dists",
    "flare_sparsity",
    "background_context_range",
    "background_range",
    "flare_range",
    "detail_preservation",
    "edge_preservation_loss",
    "sparsity_loss",
)


class LossCleanupTest(unittest.TestCase):
    def test_removed_loss_code_is_absent(self):
        checked_paths = [
            PROJECT_ROOT / "src/train_lucid.py",
            PROJECT_ROOT / "src/Flare_Disentangle.py",
            PROJECT_ROOT / "src/utils/loss.py",
            PROJECT_ROOT / "scripts/train_lucid.sh",
        ]

        for path in checked_paths:
            source = path.read_text(encoding="utf-8")
            for token in REMOVED_LOSS_TOKENS:
                self.assertNotIn(token, source, f"{token} should be removed from {path}")

    def test_utils_moved_under_src(self):
        self.assertTrue((PROJECT_ROOT / "src/utils/loss.py").exists())
        self.assertTrue((PROJECT_ROOT / "src/utils/color_fix.py").exists())
        self.assertFalse((PROJECT_ROOT / "utils/loss.py").exists())
        self.assertFalse((PROJECT_ROOT / "utils/color_fix.py").exists())

    def test_flare_loss_lives_in_src_utils(self):
        flare_source = (PROJECT_ROOT / "src/Flare_Disentangle.py").read_text(encoding="utf-8")
        train_source = (PROJECT_ROOT / "src/train_disentangle.py").read_text(encoding="utf-8")
        loss_source = (PROJECT_ROOT / "src/utils/loss.py").read_text(encoding="utf-8")

        self.assertIn("class FlareDisentanglementLoss", loss_source)
        self.assertNotIn("class FlareDisentanglementLoss", flare_source)
        self.assertIn("from .utils.loss import FlareDisentanglementLoss", train_source)


if __name__ == "__main__":
    unittest.main()
