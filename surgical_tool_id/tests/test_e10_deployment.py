"""E10 full-fit and offline inference contract checks."""

import csv
import hashlib
import json
import runpy
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
E10 = ROOT / "experiments" / "e10_final"
PACKAGE = ROOT / "checkpoints" / "e10"
sys.path.insert(0, str(ROOT))
from contract import validate_schema
from final_model import ResNet18, preprocess_image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_state_hash(model):
    """E8 frozen-state definition without its removed pretrained-weight file."""
    batchnorm_parameters = {f"{module_name}.{parameter_name}"
                            for module_name, module in model.named_modules()
                            if isinstance(module, nn.BatchNorm2d)
                            for parameter_name, _ in module.named_parameters(recurse=False)}
    trainable = {name for name, _ in model.named_parameters()
                 if name.startswith(("layer4.", "fc.")) and name not in batchnorm_parameters}
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name not in trainable:
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class E10Tests(unittest.TestCase):
    def test_epoch_rule_and_fit_records(self):
        values = [8, 8, 10, 10, 7, 8, 11, 8, 5, 2, 14, 12, 14, 7, 13]
        median, count = 8, 9
        self.assertEqual(len(values), 15)
        self.assertEqual(median, 8)
        self.assertEqual(count, 9)
        for seed in (17, 42, 123):
            result = json.loads((E10 / f"seed_{seed}_result.json").read_text())
            self.assertEqual(result["n_samples"], 1402)
            self.assertEqual(result["epochs_ran"], 9)
            self.assertEqual(len(result["training_losses"]), 9)
            self.assertEqual(result["selected_epochs_zero_based"], values)
            self.assertEqual(result["checkpoint_sha256"], sha(PACKAGE / f"seed_{seed}.pt"))
            self.assertEqual(result["frozen_state_before_sha256"], result["frozen_state_after_sha256"])
            model = ResNet18(num_classes=4)
            model.load_state_dict(torch.load(PACKAGE / f"seed_{seed}.pt",
                                             map_location="cpu", weights_only=True), strict=True)
            self.assertEqual(frozen_state_hash(model), result["frozen_state_before_sha256"])
            self.assertNotIn("validation", result)
            self.assertNotIn("accuracy", result)
            self.assertNotIn("f1", result)

    def test_packaged_hashes_and_replay(self):
        manifest = json.loads((PACKAGE / "manifest.json").read_text())
        self.assertEqual(manifest["class_names"], ["clipper", "grasper", "hook", "scissor"])
        self.assertEqual(manifest["seeds"], [17, 42, 123])
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "synthetic.png"
            Image.new("RGB", (86, 128), "red").save(image)
            tensor = preprocess_image(image).unsqueeze(0)
            probabilities = []
            for seed in manifest["seeds"]:
                path = PACKAGE / f"seed_{seed}.pt"
                self.assertEqual(sha(path), manifest["checkpoint_sha256"][str(seed)])
                self.assertEqual(sha(path), json.loads(
                    (E10 / f"seed_{seed}_result.json").read_text())["checkpoint_sha256"])
                model = ResNet18(num_classes=4)
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
                model.eval()
                with torch.inference_mode():
                    probabilities.append(torch.softmax(model(tensor), dim=1))
        average = torch.stack(probabilities).mean(dim=0)
        self.assertEqual(average.shape, (1, 4))
        self.assertAlmostEqual(average.sum().item(), 1.0, places=6)

    def test_clean_offline_copy_repeatable_csv_and_duplicate_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            copy = base / "install"
            (copy / "checkpoints").mkdir(parents=True)
            shutil.copy2(ROOT / "predict.py", copy)
            shutil.copy2(ROOT / "final_model.py", copy)
            shutil.copytree(PACKAGE, copy / "checkpoints" / "e10")
            data = base / "data"
            for name in ("a/same.png", "b/same.png", "unique.PNG"):
                path = data / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (86, 128), "red").save(path)
            outputs = []
            for index in range(2):
                out = base / f"out_{index}.csv"
                original_argv = sys.argv
                original_model = sys.modules.pop("final_model", None)
                sys.path.insert(0, str(copy))
                sys.argv = [str(copy / "predict.py"), "--data-dir", str(data), "--out", str(out)]
                try:
                    runpy.run_path(str(copy / "predict.py"), run_name="__main__")
                finally:
                    sys.argv = original_argv
                    sys.path.remove(str(copy))
                    sys.modules.pop("final_model", None)
                    if original_model is not None:
                        sys.modules["final_model"] = original_model
                outputs.append(out.read_bytes())
            self.assertEqual(outputs[0], outputs[1])
            with (base / "out_0.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([r["filename"] for r in rows],
                             ["a/same.png", "b/same.png", "unique.PNG"])
            self.assertTrue(validate_schema(base / "out_0.csv",
                                            ["a/same.png", "b/same.png", "unique.PNG"],
                                            ["clipper", "grasper", "hook", "scissor"])["ok"])


if __name__ == "__main__":
    unittest.main()
