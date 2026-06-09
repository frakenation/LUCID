import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InferenceAndFlareCleanupTest(unittest.TestCase):
    def test_single_inference_has_no_component_or_grid_outputs(self):
        inference_source = (PROJECT_ROOT / "src/inference.py").read_text(encoding="utf-8")
        script_paths = list((PROJECT_ROOT / "scripts").glob("*.sh"))
        script_source = "\n".join(path.read_text(encoding="utf-8") for path in script_paths)

        forbidden = (
            "--save_components",
            "--save_grid",
            "create_comparison_grid",
            "components",
            "grids",
            "illuminance_",
            "background_",
            "components/flare",
            "flare_path =",
            "infer_lucid_components",
        )

        for token in forbidden:
            self.assertNotIn(token, inference_source)
            self.assertNotIn(token, script_source)

    def test_flare_disentangle_has_no_chinese_text_or_demo_prints(self):
        source = (PROJECT_ROOT / "src/Flare_Disentangle.py").read_text(encoding="utf-8")

        self.assertNotRegex(source, r"[\u4e00-\u9fff]")
        self.assertNotIn("print(", source)
        self.assertNotIn('if __name__ == "__main__":', source)

    def test_inference_and_training_prints_are_trimmed(self):
        checked_paths = [
            PROJECT_ROOT / "src/inference.py",
            PROJECT_ROOT / "src/inference_cfg_index.py",
            PROJECT_ROOT / "src/model_cfg.py",
            PROJECT_ROOT / "src/train_disentangle.py",
            PROJECT_ROOT / "src/train_lucid.py",
        ]

        for path in checked_paths:
            source = path.read_text(encoding="utf-8")
            self.assertNotRegex(source, r"[\u4e00-\u9fff]")
            self.assertNotIn('print("="', source)
            self.assertNotIn("Initializing", source)
            self.assertNotIn("Model loaded", source)
            self.assertNotIn("Number of trainable parameters", source)
            self.assertNotIn("Number of parameters in flare", source)


if __name__ == "__main__":
    unittest.main()
