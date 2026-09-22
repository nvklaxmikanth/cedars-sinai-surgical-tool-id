import csv
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
E8 = ROOT / "experiments" / "e8_partial_resnet18"
model_spec = importlib.util.spec_from_file_location("e8_resnet18_model_for_tests", E8 / "model.py")
model = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model)
WEIGHTS_PATH = model.WEIGHTS_PATH
WEIGHTS_SHA256 = model.WEIGHTS_SHA256
forward_from_layer3 = model.forward_from_layer3
forward_to_layer3 = model.forward_to_layer3
frozen_state_hash = model.frozen_state_hash
load_partial_model = model.load_partial_model
predict_image = model.predict_image
preprocess_image = model.preprocess_image

spec = importlib.util.spec_from_file_location("e8_train_for_tests", E8 / "train_cv.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class PartialResNetTests(unittest.TestCase):
    def test_only_layer4_and_head_train_with_ten_to_one_rates(self):
        torch.manual_seed(42)
        e7_initial_head = nn.Linear(512, 4)
        torch.manual_seed(42)
        model = load_partial_model()
        self.assertTrue(torch.equal(model.fc.weight, e7_initial_head.weight))
        self.assertTrue(torch.equal(model.fc.bias, e7_initial_head.bias))
        batchnorm_parameter_names = {f"{module_name}.{parameter_name}"
                                     for module_name, module in model.named_modules()
                                     if isinstance(module, nn.BatchNorm2d)
                                     for parameter_name, _ in module.named_parameters(recurse=False)}
        expected = {name for name, _ in model.named_parameters()
                    if name.startswith(("layer4.", "fc."))
                    and name not in batchnorm_parameter_names}
        self.assertEqual({name for name, p in model.named_parameters() if p.requires_grad}, expected)
        self.assertTrue(all(not p.requires_grad for layer in model.modules()
                            if isinstance(layer, nn.BatchNorm2d) for p in layer.parameters()))
        self.assertEqual(train.LAYER4_LEARNING_RATE, train.LEARNING_RATE / 10)
        self.assertTrue(all(not layer.training for layer in model.modules()
                            if isinstance(layer, nn.BatchNorm2d)))

    def test_e7_settings_preprocessing_and_fold_weights_are_retained(self):
        self.assertEqual((train.SEED, train.BATCH_SIZE, train.MAX_EPOCHS,
                          train.PATIENCE, train.MIN_DELTA, train.LEARNING_RATE),
                         (42, 2, 30, 3, 1e-4, 1e-3))
        self.assertIs(preprocess_image, model.e7_model.preprocess_image)
        e7 = json.loads((ROOT / "experiments" / "e7_frozen_resnet18" / "results.json").read_text())
        e8 = json.loads((E8 / "results.json").read_text())
        self.assertEqual(e8["manifest_sha256"], e7["manifest_sha256"])
        self.assertEqual(e8["weights_sha256"], e7["weights_sha256"])
        for earlier, current in zip(e7["folds"], e8["folds"]):
            self.assertEqual(current["train_class_counts"], earlier["train_class_counts"])
            self.assertEqual(current["class_weights"], earlier["class_weights"])

    def test_frozen_prefix_and_all_batchnorm_buffers_remain_byte_identical(self):
        model = load_partial_model()
        before = frozen_state_hash(model)
        frozen = {name: value.clone() for name, value in model.state_dict().items()
                  if name not in {n for n, p in model.named_parameters() if p.requires_grad}}
        original_layer4 = model.layer4[0].conv1.weight.clone()
        optimizer = torch.optim.Adam([
            {"params": [p for p in model.layer4.parameters() if p.requires_grad],
             "lr": train.LAYER4_LEARNING_RATE},
            {"params": model.fc.parameters(), "lr": train.LEARNING_RATE},
        ])
        model.eval()
        x = torch.randn(2, 256, 14, 14)
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(forward_from_layer3(model, x), torch.tensor([0, 1]))
        loss.backward()
        optimizer.step()
        self.assertEqual(frozen_state_hash(model), before)
        self.assertTrue(all(torch.equal(model.state_dict()[name], value)
                            for name, value in frozen.items()))
        self.assertFalse(torch.equal(model.layer4[0].conv1.weight, original_layer4))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))

    def test_offline_weights_and_prefix_match_full_model(self):
        self.assertEqual(hashlib.sha256(WEIGHTS_PATH.read_bytes()).hexdigest(), WEIGHTS_SHA256)
        manifest = json.loads((ROOT / "splits" / "video_grouped_fivefold_v1.json").read_text())
        with patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("network access")):
            model = load_partial_model()
        path = ROOT / manifest["data_root"] / manifest["samples"][0]["path"]
        image = preprocess_image(path).unsqueeze(0)
        with torch.no_grad():
            fresh = forward_to_layer3(model, image)
            self.assertEqual(tuple(fresh.shape), (1, 256, 14, 14))
            self.assertTrue(torch.allclose(model(image), forward_from_layer3(model, fresh)))
        self.assertEqual(json.loads((E8 / "feature_cache_meta.json").read_text())["n"], 1402)

    def test_oof_checkpoints_metrics_and_offline_inference_reproduce(self):
        manifest = json.loads((ROOT / "splits" / "video_grouped_fivefold_v1.json").read_text())
        summary = json.loads((E8 / "results.json").read_text())
        by_path = {sample["path"]: i for i, sample in enumerate(manifest["samples"])}
        with (E8 / "oof_predictions.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), len(by_path))
        self.assertEqual({row["path"] for row in rows}, set(by_path))
        self.assertEqual(train.classification_metrics(
            [row["true_label"] for row in rows],
            [row["predicted_label"] for row in rows], summary["class_names"]),
            summary["pooled_metrics"])
        for fold in summary["folds"]:
            number = fold["fold"]
            self.assertEqual(fold["frozen_state_before_sha256"],
                             fold["frozen_state_after_sha256"])
            history = json.loads((E8 / f"fold_{number}_result.json").read_text())["history"]
            self.assertEqual(history[fold["selected_epoch"]]["val_macro_f1"],
                             max(epoch["val_macro_f1"] for epoch in history))
            checkpoint = E8 / "checkpoints" / f"fold_{number}.pt"
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             fold["checkpoint_sha256"])
            with (E8 / f"fold_{number}_oof.csv").open(newline="") as file:
                fold_rows = list(csv.DictReader(file))
            self.assertEqual([row["path"] for row in fold_rows],
                             manifest["folds"][number]["validation_paths"])
            first = fold_rows[0]
            with patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("network access")):
                index, label = predict_image(ROOT / manifest["data_root"] / first["path"],
                                             checkpoint, summary["class_names"])
            self.assertEqual((index, label), (int(first["predicted_index"]),
                                              first["predicted_label"]))

    def test_primary_selection_compares_e6_and_e7(self):
        comparison = json.loads((E8 / "comparison_to_e6_e7.json").read_text())
        self.assertEqual(comparison["primary_metric"], "pooled_macro_f1")
        scores = {"E8": comparison["e8_pooled_macro_f1"],
                  "E7": comparison["e7"]["pooled_macro_f1"],
                  "E6": comparison["e6"]["pooled_macro_f1"]}
        self.assertEqual(comparison["selected_experiment"], max(scores, key=scores.get))


if __name__ == "__main__":
    unittest.main()
