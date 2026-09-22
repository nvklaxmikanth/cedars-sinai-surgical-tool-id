import copy
import csv
import hashlib
import importlib.util
import json
import statistics
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
E8 = ROOT / "experiments" / "e8_partial_resnet18"
E9 = ROOT / "experiments" / "e9_seed_ensemble"
spec = importlib.util.spec_from_file_location("e9_aggregate_for_tests", E9 / "aggregate.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)


class SeedEnsembleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "splits" / "video_grouped_fivefold_v1.json").read_text())
        cls.results = json.loads((E9 / "results.json").read_text())

    def test_unchanged_seed_42_and_seed_training_settings(self):
        e8_results = json.loads((E8 / "results.json").read_text())
        self.assertEqual(hashlib.sha256((E8 / "results.json").read_bytes()).hexdigest(),
                         self.results["seed_42_e8_results_sha256"])
        self.assertEqual(hashlib.sha256((E8 / "oof_predictions.csv").read_bytes()).hexdigest(),
                         self.results["seed_42_e8_oof_sha256"])
        self.assertEqual(self.results["individual_seeds"]["42"]["metrics"],
                         e8_results["pooled_metrics"])
        self.assertEqual(self.results["seeds"], [17, 42, 123])
        for seed in (17, 123):
            for fold, earlier in zip(self.results["individual_seeds"][str(seed)]["folds"],
                                     e8_results["folds"]):
                self.assertEqual(fold["seed"], seed)
                self.assertEqual(fold["train_class_counts"], earlier["train_class_counts"])
                self.assertEqual(fold["class_weights"], earlier["class_weights"])
                self.assertEqual(fold["layer4_learning_rate"], earlier["layer4_learning_rate"])
                self.assertEqual(fold["head_learning_rate"], earlier["head_learning_rate"])
                self.assertEqual(fold["validation_video_ids"], earlier["validation_video_ids"])

    def test_alignment_rejects_missing_duplicate_and_mislabeled_samples(self):
        seed_rows = {seed: aggregate.read_csv(E9 / f"seed_{seed}_oof_probabilities.csv")
                     for seed in (17, 42, 123)}
        indexed = aggregate.validate_alignment(seed_rows, self.manifest)
        self.assertEqual(len(indexed[17]), 1402)
        bad = copy.deepcopy(seed_rows)
        bad[17][0]["path"] = bad[17][1]["path"]
        with self.assertRaisesRegex(ValueError, "missing or duplicate"):
            aggregate.validate_alignment(bad, self.manifest)
        bad = copy.deepcopy(seed_rows)
        bad[123][0]["true_label"] = "incorrect"
        with self.assertRaisesRegex(ValueError, "truth or fold mismatch"):
            aggregate.validate_alignment(bad, self.manifest)
        bad = copy.deepcopy(seed_rows)
        bad[42][0]["fold"] = str((int(bad[42][0]["fold"]) + 1) % 5)
        with self.assertRaisesRegex(ValueError, "truth or fold mismatch"):
            aggregate.validate_alignment(bad, self.manifest)

    def test_all_checkpoint_predictions_and_frozen_states_replay(self):
        classes = self.results["class_names"]
        frozen_hash = self.results["frozen_state_sha256"]
        for seed in (17, 42, 123):
            source = E8 if seed == 42 else E9 / f"seed_{seed}"
            with (E9 / f"seed_{seed}_oof_probabilities.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1402)
            self.assertEqual(aggregate.e8.classification_metrics(
                [row["true_label"] for row in rows],
                [row["predicted_label"] for row in rows], classes),
                self.results["individual_seeds"][str(seed)]["metrics"])
            for fold in self.results["individual_seeds"][str(seed)]["folds"]:
                number = fold["fold"]
                checkpoint = source / "checkpoints" / f"fold_{number}.pt"
                self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                 fold["checkpoint_sha256"])
                self.assertEqual(fold["frozen_state_before_sha256"], frozen_hash)
                self.assertEqual(fold["frozen_state_after_sha256"], frozen_hash)
                first = next(row for row in rows if int(row["fold"]) == number)
                model = aggregate.e8.load_partial_model(num_classes=len(classes))
                aggregate.e8.load_trainable_state(model, checkpoint)
                self.assertEqual(aggregate.e8.frozen_state_hash(model), frozen_hash)
                self.assertTrue(all(not p.requires_grad for module in model.modules()
                                    if isinstance(module, nn.BatchNorm2d)
                                    for p in module.parameters()))
                image = aggregate.e8.preprocess_image(
                    ROOT / self.manifest["data_root"] / first["path"]).unsqueeze(0)
                with torch.no_grad():
                    actual = model(image).argmax(dim=1).item()
                self.assertEqual(actual, int(first["predicted_index"]))

    def test_ensemble_probabilities_statistics_and_selection(self):
        by_seed = {seed: {row["path"]: row for row in aggregate.read_csv(
            E9 / f"seed_{seed}_oof_probabilities.csv")} for seed in (17, 42, 123)}
        ensemble = aggregate.read_csv(E9 / "ensemble_oof_probabilities.csv")
        classes = self.results["class_names"]
        self.assertEqual(len(ensemble), 1402)
        for row in ensemble:
            for name in classes:
                expected = sum(float(by_seed[seed][row["path"]][f"p_{name}"])
                               for seed in (17, 42, 123)) / 3
                self.assertAlmostEqual(float(row[f"p_{name}"]), expected, places=12)
            probabilities = [float(row[f"p_{name}"]) for name in classes]
            self.assertEqual(int(row["predicted_index"]),
                             max(range(4), key=lambda index: probabilities[index]))
        self.assertEqual(aggregate.e8.classification_metrics(
            [row["true_label"] for row in ensemble],
            [row["predicted_label"] for row in ensemble], classes),
            self.results["ensemble"]["metrics"])
        individual = [self.results["individual_seeds"][str(seed)]["metrics"]["macro_f1"]
                      for seed in (17, 42, 123)]
        stats = self.results["across_seed_statistics"]["pooled"]["macro_f1"]
        self.assertAlmostEqual(stats["mean"], statistics.mean(individual))
        self.assertAlmostEqual(stats["sample_std"], statistics.stdev(individual))
        better = self.results["ensemble"]["metrics"]["macro_f1"] > max(individual)
        self.assertEqual(self.results["ensemble_selected"], better)
        self.assertEqual(self.results["selected_model"],
                         "equal_weight_ensemble" if better else
                         f"seed_{self.results['best_individual_seed']}")


if __name__ == "__main__":
    unittest.main()
