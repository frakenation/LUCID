import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CheckpointUtilsTest(unittest.TestCase):
    def test_checkpoint_io_is_centralized(self):
        checkpoint_source = (PROJECT_ROOT / "src/utils/checkpoint.py").read_text(encoding="utf-8")
        model_source = (PROJECT_ROOT / "src/model_cfg.py").read_text(encoding="utf-8")
        train_source = (PROJECT_ROOT / "src/train_disentangle.py").read_text(encoding="utf-8")

        for function_name in (
            "load_lucid_state",
            "save_lucid_state",
            "load_flare_state",
            "build_disentanglement_checkpoint",
            "load_disentanglement_checkpoint",
        ):
            self.assertIn(f"def {function_name}(", checkpoint_source)

        self.assertNotIn("torch.load(", model_source)
        self.assertNotIn("torch.save(", model_source)
        self.assertNotIn("torch.load(", train_source)
        self.assertNotIn("torch.save(", train_source)


if __name__ == "__main__":
    unittest.main()
