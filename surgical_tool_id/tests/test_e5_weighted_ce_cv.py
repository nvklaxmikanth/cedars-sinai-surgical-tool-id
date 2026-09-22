import csv
import hashlib
import importlib.util
import json
import unittest
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
E4 = ROOT / 'experiments' / 'e4_cross_entropy_cv'
E5 = ROOT / 'experiments' / 'e5_weighted_ce_cv'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


unweighted = load_module('e4_train_for_e5_test', E4 / 'train_cv.py')
weighted = load_module('e5_train_for_e5_test', E5 / 'train_cv.py')


class WeightedCvTests(unittest.TestCase):
    def test_weights_are_inverse_frequency_and_mean_one(self):
        classes = ['a', 'b', 'c', 'd']
        samples = [{'label': label} for label in ['a', 'a', 'b', 'c', 'd', 'd', 'd', 'd']]
        weights = weighted.inverse_frequency_weights(samples, classes)
        self.assertAlmostEqual(weights.mean().item(), 1.0)
        self.assertTrue(torch.allclose(weights / weights[0],
                                       torch.tensor([1., 2., 2., 0.5])))
        with self.assertRaisesRegex(ValueError, 'every class needs a training sample'):
            weighted.inverse_frequency_weights(samples[:-4], classes)

    @unittest.skip("requires private dataset or removed historical model artifact")
    def test_fold_weights_use_training_labels_only(self):
        manifest = json.loads((ROOT / 'splits' / 'video_grouped_fivefold_v1.json').read_text())
        by_path = {sample['path']: sample for sample in manifest['samples']}
        global_weights = weighted.inverse_frequency_weights(manifest['samples'],
                                                            manifest['class_names'])
        for fold in manifest['folds']:
            samples = [by_path[path] for path in fold['train_paths']]
            expected = weighted.inverse_frequency_weights(samples, manifest['class_names'])
            result = json.loads((E5 / f"fold_{fold['fold']}_result.json").read_text())
            self.assertEqual(result['train_class_counts'],
                             dict(sorted(Counter(s['label'] for s in samples).items())))
            actual = torch.tensor([result['class_weights'][name]
                                   for name in manifest['class_names']])
            self.assertTrue(torch.equal(actual, expected))
            self.assertFalse(torch.equal(actual, global_weights))

    @unittest.skip("requires private dataset or removed historical model artifact")
    def test_all_non_loss_settings_and_preprocessing_match_e4(self):
        manifest = json.loads((ROOT / 'splits' / 'video_grouped_fivefold_v1.json').read_text())
        sample = manifest['samples'][0]
        root = ROOT / manifest['data_root']
        old_image, old_label = unweighted.FrameDataset([sample], root, manifest['class_names'])[0]
        new_image, new_label = weighted.FrameDataset([sample], root, manifest['class_names'])[0]
        self.assertTrue(torch.equal(old_image, new_image))
        self.assertEqual(old_label, new_label)
        for name in ('SEED', 'BATCH_SIZE', 'MAX_EPOCHS', 'PATIENCE', 'MIN_DELTA', 'LEARNING_RATE'):
            self.assertEqual(getattr(unweighted, name), getattr(weighted, name))
        self.assertEqual({k: tuple(v.shape) for k, v in unweighted.SmallCNN(4).state_dict().items()},
                         {k: tuple(v.shape) for k, v in weighted.SmallCNN(4).state_dict().items()})

    def test_weighted_loss_uses_raw_logits_and_integer_targets(self):
        logits = torch.tensor([[2., 0., -1., 0.5], [0., 1., 2., -0.5]], requires_grad=True)
        targets = torch.tensor([0, 2], dtype=torch.long)
        weights = torch.tensor([0.5, 1., 2., 0.5])
        actual = torch.nn.CrossEntropyLoss(weight=weights)(logits, targets)
        log_probs = torch.log_softmax(logits, dim=1)
        expected = -(weights[0] * log_probs[0, 0] + weights[2] * log_probs[1, 2]) / (weights[0] + weights[2])
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    @unittest.skip("requires private dataset or removed historical model artifact")
    def test_oof_metrics_and_selected_checkpoints_reproduce(self):
        manifest = json.loads((ROOT / 'splits' / 'video_grouped_fivefold_v1.json').read_text())
        summary = json.loads((E5 / 'results.json').read_text())
        by_path = {sample['path']: sample for sample in manifest['samples']}
        with (E5 / 'oof_predictions.csv').open(newline='') as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), len(by_path))
        self.assertEqual({row['path'] for row in rows}, set(by_path))
        self.assertEqual(weighted.classification_metrics(
            [row['true_label'] for row in rows],
            [row['predicted_label'] for row in rows], summary['class_names']),
            summary['pooled_metrics'])
        for fold in summary['folds']:
            number = fold['fold']
            history = json.loads((E5 / f'fold_{number}_result.json').read_text())['history']
            self.assertEqual(history[fold['selected_epoch']]['val_macro_f1'],
                             max(epoch['val_macro_f1'] for epoch in history))
            checkpoint = E5 / 'checkpoints' / f'fold_{number}.pt'
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             fold['checkpoint_sha256'])
            model = weighted.SmallCNN(4)
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
            samples = [by_path[path] for path in manifest['folds'][number]['validation_paths']]
            loader = DataLoader(weighted.FrameDataset(samples, ROOT / manifest['data_root'],
                                                       summary['class_names']), batch_size=64)
            indices = weighted.predict_indices(model, loader)
            with (E5 / f'fold_{number}_oof.csv').open(newline='') as file:
                fold_rows = list(csv.DictReader(file))
            self.assertEqual([row['path'] for row in fold_rows],
                             manifest['folds'][number]['validation_paths'])
            self.assertEqual(indices, [int(row['predicted_index']) for row in fold_rows])

    def test_primary_selection_uses_pooled_macro_f1(self):
        comparison = json.loads((E5 / 'comparison_to_e4.json').read_text())
        e4 = json.loads((E4 / 'results.json').read_text())
        e5 = json.loads((E5 / 'results.json').read_text())
        self.assertEqual(comparison['primary_metric'], 'pooled_macro_f1')
        self.assertEqual(comparison['e4_pooled_macro_f1'], e4['pooled_metrics']['macro_f1'])
        self.assertEqual(comparison['e5_pooled_macro_f1'], e5['pooled_metrics']['macro_f1'])
        self.assertEqual(comparison['selected_experiment'], 'E5')
        self.assertEqual(comparison['changed_oof_predictions'], 210)


if __name__ == '__main__':
    unittest.main()
