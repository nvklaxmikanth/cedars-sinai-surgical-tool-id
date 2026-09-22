import csv
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
E3 = ROOT / 'experiments' / 'e3_legacy_cv'
E4 = ROOT / 'experiments' / 'e4_cross_entropy_cv'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = load_module('e3_train_for_e4_test', E3 / 'train_cv.py')
current = load_module('e4_train_for_e4_test', E4 / 'train_cv.py')


class CrossEntropyCvTests(unittest.TestCase):
    @unittest.skip("requires private dataset or removed historical model artifact")
    def test_preprocessing_architecture_and_settings_match_e3(self):
        manifest = json.loads((ROOT / 'splits' / 'video_grouped_fivefold_v1.json').read_text())
        sample = manifest['samples'][0]
        root = ROOT / manifest['data_root']
        old_image, old_target = legacy.FrameDataset([sample], root, manifest['class_names'])[0]
        new_image, new_target = current.FrameDataset([sample], root, manifest['class_names'])[0]
        self.assertTrue(torch.equal(old_image, new_image))
        self.assertEqual(new_target, old_target.argmax().item())
        self.assertIsInstance(new_target, int)
        for name in ('SEED', 'BATCH_SIZE', 'MAX_EPOCHS', 'PATIENCE', 'MIN_DELTA', 'LEARNING_RATE'):
            self.assertEqual(getattr(legacy, name), getattr(current, name))
        self.assertEqual({k: tuple(v.shape) for k, v in legacy.SmallCNN(4).state_dict().items()},
                         {k: tuple(v.shape) for k, v in current.SmallCNN(4).state_dict().items()})

    def test_cross_entropy_uses_raw_logits_and_integer_targets(self):
        logits = torch.tensor([[3.0, 1.0, -1.0, 0.5], [-0.5, 2.0, 1.0, 0.0]],
                              requires_grad=True)
        targets = torch.tensor([0, 2], dtype=torch.long)
        loss = torch.nn.CrossEntropyLoss()(logits, targets)
        expected = -torch.log_softmax(logits, dim=1)[torch.arange(2), targets].mean()
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    @unittest.skip("requires private dataset or removed historical model artifact")
    def test_e4_oof_metrics_and_checkpoints_reproduce(self):
        manifest = json.loads((ROOT / 'splits' / 'video_grouped_fivefold_v1.json').read_text())
        summary = json.loads((E4 / 'results.json').read_text())
        by_path = {sample['path']: sample for sample in manifest['samples']}
        with (E4 / 'oof_predictions.csv').open(newline='') as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), len(by_path))
        self.assertEqual({row['path'] for row in rows}, set(by_path))
        self.assertEqual(current.classification_metrics(
            [row['true_label'] for row in rows],
            [row['predicted_label'] for row in rows], summary['class_names']),
            summary['pooled_metrics'])
        for fold in summary['folds']:
            number = fold['fold']
            selected = fold['selected_epoch']
            history = json.loads((E4 / f'fold_{number}_result.json').read_text())['history']
            self.assertEqual(history[selected]['val_macro_f1'],
                             max(epoch['val_macro_f1'] for epoch in history))
            checkpoint = E4 / 'checkpoints' / f'fold_{number}.pt'
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             fold['checkpoint_sha256'])
            model = current.SmallCNN(4)
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
            samples = [by_path[path] for path in manifest['folds'][number]['validation_paths']]
            loader = DataLoader(current.FrameDataset(samples, ROOT / manifest['data_root'],
                                                      summary['class_names']), batch_size=64)
            indices = current.predict_indices(model, loader)
            with (E4 / f'fold_{number}_oof.csv').open(newline='') as file:
                fold_rows = list(csv.DictReader(file))
            self.assertEqual([row['path'] for row in fold_rows],
                             manifest['folds'][number]['validation_paths'])
            self.assertEqual(indices, [int(row['predicted_index']) for row in fold_rows])


if __name__ == '__main__':
    unittest.main()
