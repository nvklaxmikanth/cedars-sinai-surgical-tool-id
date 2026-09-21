import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
E3 = ROOT / 'experiments' / 'e3_legacy_cv'
sys.path.insert(0, str(E3))
from train_cv import FrameDataset, SmallCNN, classification_metrics, predict_indices, set_seed


class LegacyCvTests(unittest.TestCase):
    def test_seed_reproduces_initial_model(self):
        set_seed(42)
        first = SmallCNN(4).state_dict()
        first_hashes = {key: hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                        for key, tensor in first.items()}
        set_seed(42)
        second = SmallCNN(4).state_dict()
        second_hashes = {key: hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                         for key, tensor in second.items()}
        self.assertEqual(first_hashes, second_hashes)

    def test_seed_reproduces_one_training_update(self):
        def update_hashes():
            set_seed(42)
            model = SmallCNN(4)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            images = torch.ones(2, 3, 128, 128) * 0.25
            targets = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
            model.train()
            optimizer.zero_grad()
            loss = torch.nn.MSELoss()(torch.softmax(model(images), dim=1), targets)
            loss.backward()
            optimizer.step()
            return {key: hashlib.sha256(tensor.detach().numpy().tobytes()).hexdigest()
                    for key, tensor in model.state_dict().items()}

        self.assertEqual(update_hashes(), update_hashes())

    def test_architecture_matches_legacy_checkpoint_shapes(self):
        model = SmallCNN(4)
        supplied = torch.load(ROOT / 'checkpoints' / 'model_best.pt',
                              map_location='cpu', weights_only=True)
        self.assertEqual({key: tuple(value.shape) for key, value in model.state_dict().items()},
                         {key: tuple(value.shape) for key, value in supplied.items()})

    def test_metrics_known_confusion_matrix(self):
        classes = ['a', 'b', 'c', 'd']
        result = classification_metrics(['a', 'a', 'b', 'c', 'd'],
                                        ['a', 'b', 'b', 'd', 'd'], classes)
        self.assertEqual(result['confusion_matrix'],
                         [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
        self.assertAlmostEqual(result['accuracy'], 0.6)
        self.assertAlmostEqual(result['per_class']['a']['precision'], 1.0)
        self.assertAlmostEqual(result['per_class']['a']['recall'], 0.5)
        self.assertAlmostEqual(result['per_class']['a']['f1'], 2 / 3)
        self.assertEqual(result['per_class']['c']['f1'], 0.0)
        self.assertAlmostEqual(result['macro_f1'], (2 / 3 + 2 / 3 + 0 + 2 / 3) / 4)

    def test_oof_coverage_and_recomputed_metrics(self):
        manifest = json.loads((ROOT / 'splits' / 'video_grouped_fivefold_v1.json').read_text())
        results = json.loads((E3 / 'results.json').read_text())
        with (E3 / 'oof_predictions.csv').open(newline='') as file:
            rows = list(csv.DictReader(file))
        sample_by_path = {sample['path']: sample for sample in manifest['samples']}
        self.assertEqual(len(rows), len(sample_by_path))
        self.assertEqual({row['path'] for row in rows}, set(sample_by_path))
        for row in rows:
            self.assertEqual(row['true_label'], sample_by_path[row['path']]['label'])
            self.assertEqual(row['video_id'], sample_by_path[row['path']]['video_id'])
            self.assertEqual(row['predicted_label'],
                             results['class_names'][int(row['predicted_index'])])
        self.assertEqual(classification_metrics([r['true_label'] for r in rows],
                                                [r['predicted_label'] for r in rows],
                                                results['class_names']), results['pooled_metrics'])
        for fold in results['folds']:
            history = json.loads((E3 / f"fold_{fold['fold']}_result.json").read_text())['history']
            selected = fold['selected_epoch']
            self.assertEqual(history[selected]['val_macro_f1'],
                             max(epoch['val_macro_f1'] for epoch in history))
            checkpoint = E3 / 'checkpoints' / f"fold_{fold['fold']}.pt"
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             fold['checkpoint_sha256'])

    def test_selected_checkpoints_reproduce_oof_predictions(self):
        manifest = json.loads((ROOT / 'splits' / 'video_grouped_fivefold_v1.json').read_text())
        sample_by_path = {sample['path']: sample for sample in manifest['samples']}
        for fold in manifest['folds']:
            model = SmallCNN(len(manifest['class_names']))
            checkpoint = E3 / 'checkpoints' / f"fold_{fold['fold']}.pt"
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
            samples = [sample_by_path[path] for path in fold['validation_paths']]
            loader = DataLoader(FrameDataset(samples, ROOT / manifest['data_root'],
                                             manifest['class_names']), batch_size=64)
            indices = predict_indices(model, loader)
            with (E3 / f"fold_{fold['fold']}_oof.csv").open(newline='') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row['path'] for row in rows], fold['validation_paths'])
            self.assertEqual(indices, [int(row['predicted_index']) for row in rows])


if __name__ == '__main__':
    unittest.main()
