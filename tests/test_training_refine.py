import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainingRefineTest(unittest.TestCase):
    def test_count_parameters_is_removed(self):
        for relative_path in ("src/Flare_Disentangle.py", "src/train_disentangle.py"):
            source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
            self.assertNotIn("count_parameters", source)
            self.assertNotIn("Model parameters", source)

    def test_wandb_image_conversion_is_single_helper(self):
        source = (PROJECT_ROOT / "src/train_lucid.py").read_text(encoding="utf-8")

        self.assertNotIn("tensor_to_wandb_image_1", source)
        self.assertNotIn("tensor_to_wandb_image_2", source)
        self.assertEqual(source.count("def tensor_to_wandb_image("), 1)
        self.assertIn("input_range='01'", source)
        self.assertIn("input_range='neg11'", source)

    def test_diffusion_training_requires_precomputed_lq(self):
        train_source = (PROJECT_ROOT / "src/train_lucid.py").read_text(encoding="utf-8")
        dataset_source = (PROJECT_ROOT / "dataloader/paired_datasets.py").read_text(encoding="utf-8")

        self.assertEqual(train_source.count("require_lq=True"), 2)
        self.assertIn("if self.require_lq:", dataset_source)
        self.assertIn("flare = result['flare']", train_source)

    def test_synthetic_training_datasets_generate_flare_online(self):
        dataset_source = (PROJECT_ROOT / "dataloader/paired_datasets.py").read_text(encoding="utf-8")

        self.assertIn("class FlareDisentanglementDataset", dataset_source)
        self.assertIn("class LUCIDFlareReinputDataset", dataset_source)
        self.assertGreaterEqual(dataset_source.count("synthesize_flare("), 4)

    def test_flare_forward_hides_illumination_estimator(self):
        source = (PROJECT_ROOT / "src/Flare_Disentangle.py").read_text(encoding="utf-8")
        forward_body = source.split("def forward(self, lq_image):", 1)[1].split("\nclass ", 1)[0]

        self.assertNotIn("remap_legacy_state_dict", source)
        self.assertNotIn("estimate_illumination", source)
        self.assertNotIn("reflectance_head", source)
        self.assertIn("def _background_prior(self, lq_image):", source)
        self.assertIn("self._background_prior(lq_image)", forward_body)
        self.assertIn("background_residual", forward_body)
        self.assertIn("background = torch.clamp(background_residual + background_prior_3ch", forward_body)
        self.assertNotIn("torch.max", forward_body)

        train_source = (PROJECT_ROOT / "src/train_disentangle.py").read_text(encoding="utf-8")
        self.assertNotIn("MaxRGB", train_source)
        self.assertNotIn("max_rgb", train_source)
        self.assertNotIn("reflectance", train_source)


if __name__ == "__main__":
    unittest.main()
