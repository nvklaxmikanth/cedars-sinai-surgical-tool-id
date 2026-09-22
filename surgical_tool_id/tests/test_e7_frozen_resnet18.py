import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
E7 = ROOT / 'experiments' / 'e7_frozen_resnet18'
sys.path.insert(0, str(E7))
from model import (WEIGHTS_PATH, WEIGHTS_SHA256, backbone_state_hash,
                   load_frozen_backbone, predict_image, preprocess_image)

spec = importlib.util.spec_from_file_location('e7_train_for_tests', E7 / 'train_cv.py')
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)


class FrozenResNetTests(unittest.TestCase):
    def test_bundled_official_weights_load_without_network(self):
        self.assertEqual(hashlib.sha256(WEIGHTS_PATH.read_bytes()).hexdigest(), WEIGHTS_SHA256)
        with patch('torch.hub.load_state_dict_from_url', side_effect=AssertionError('network access')):
            model = load_frozen_backbone()
        self.assertEqual(model.fc.out_features, 4)
        self.assertEqual([name for name, p in model.named_parameters() if p.requires_grad],
                         ['fc.weight', 'fc.bias'])
        self.assertTrue(all(not layer.training for layer in model.modules()
                            if isinstance(layer, nn.BatchNorm2d)))

    def test_imagenet_preprocessing_uses_rgb_and_fixed_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'color.png'
            Image.new('RGB', (128, 86), (255, 128, 0)).save(path)
            actual = preprocess_image(path)
            self.assertEqual(actual.shape, (3, 224, 224))
            expected = [(1.0 - 0.485) / 0.229,
                        (128 / 255 - 0.456) / 0.224,
                        (0.0 - 0.406) / 0.225]
            for channel, value in enumerate(expected):
                self.assertTrue(torch.allclose(actual[channel], torch.full((224, 224), value)))

    def test_backbone_and_batchnorm_tensors_stay_byte_identical(self):
        model = load_frozen_backbone()
        before = backbone_state_hash(model)
        counters = {key: value.clone() for key, value in model.state_dict().items()
                    if key.endswith('num_batches_tracked')}
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
        x = torch.randn(2, 3, 224, 224)
        y = torch.tensor([0, 1])
        model.fc.train()
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()
        self.assertEqual(backbone_state_hash(model), before)
        self.assertTrue(all(torch.equal(model.state_dict()[key], value)
                            for key, value in counters.items()))
        self.assertTrue(all(parameter.grad is None for name, parameter in model.named_parameters()
                            if not name.startswith('fc.')))
        self.assertTrue(all(not layer.training for layer in model.modules()
                            if isinstance(layer, nn.BatchNorm2d)))

    def test_cached_features_match_offline_backbone(self):
        manifest = json.loads((ROOT / 'splits' / 'video_grouped_fivefold_v1.json').read_text())
        payload = torch.load(E7 / 'feature_cache.pt', map_location='cpu', weights_only=True)
        metadata = json.loads((E7 / 'feature_cache_meta.json').read_text())
        self.assertEqual(payload['paths'], [sample['path'] for sample in manifest['samples']])
        self.assertEqual(tuple(payload['features'].shape), (1402, 512))
        self.assertEqual(metadata['backbone_before_sha256'],
                         metadata['backbone_after_sha256'])
        model = load_frozen_backbone()
        self.assertEqual(backbone_state_hash(model), payload['backbone_state_sha256'])
        images = torch.stack([preprocess_image(ROOT / manifest['data_root'] / sample['path'])
                              for sample in manifest['samples'][:3]])
        with torch.no_grad():
            fresh = model.forward_features(images)
        self.assertTrue(torch.allclose(fresh, payload['features'][:3], atol=1e-5, rtol=1e-5))

    def test_offline_inference_reproduces_oof_predictions(self):
        manifest = json.loads((ROOT / 'splits' / 'video_grouped_fivefold_v1.json').read_text())
        with (E7 / 'fold_0_oof.csv').open(newline='') as file:
            rows = list(csv.DictReader(file))
        checkpoint = E7 / 'checkpoints' / 'fold_0.pt'
        with patch('torch.hub.load_state_dict_from_url', side_effect=AssertionError('network access')):
            for row in rows[:3]:
                index, label = predict_image(ROOT / manifest['data_root'] / row['path'],
                                             checkpoint, manifest['class_names'])
                self.assertEqual(index, int(row['predicted_index']))
                self.assertEqual(label, row['predicted_label'])

    def test_oof_metrics_and_head_checkpoints_reproduce(self):
        manifest = json.loads((ROOT / 'splits' / 'video_grouped_fivefold_v1.json').read_text())
        summary = json.loads((E7 / 'results.json').read_text())
        payload = torch.load(E7 / 'feature_cache.pt', map_location='cpu', weights_only=True)
        by_path = {sample['path']: i for i, sample in enumerate(manifest['samples'])}
        with (E7 / 'oof_predictions.csv').open(newline='') as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), len(by_path))
        self.assertEqual({row['path'] for row in rows}, set(by_path))
        self.assertEqual(train.classification_metrics(
            [row['true_label'] for row in rows],
            [row['predicted_label'] for row in rows], summary['class_names']),
            summary['pooled_metrics'])
        for fold in summary['folds']:
            number = fold['fold']
            history = json.loads((E7 / f'fold_{number}_result.json').read_text())['history']
            self.assertEqual(history[fold['selected_epoch']]['val_macro_f1'],
                             max(epoch['val_macro_f1'] for epoch in history))
            checkpoint = E7 / 'checkpoints' / f'fold_{number}.pt'
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             fold['checkpoint_sha256'])
            head = nn.Linear(512, 4)
            head.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
            indices = [by_path[path] for path in manifest['folds'][number]['validation_paths']]
            loader = DataLoader(train.FeatureDataset(payload['features'], indices,
                                                     manifest['samples'], summary['class_names']), batch_size=64)
            predictions = train.predict_indices(head, loader)
            with (E7 / f'fold_{number}_oof.csv').open(newline='') as file:
                fold_rows = list(csv.DictReader(file))
            self.assertEqual(predictions, [int(row['predicted_index']) for row in fold_rows])

    def test_primary_selection_uses_pooled_macro_f1(self):
        comparison = json.loads((E7 / 'comparison_to_e6.json').read_text())
        e6 = json.loads((ROOT / 'experiments' / 'e6_rgb_cv' / 'results.json').read_text())
        e7 = json.loads((E7 / 'results.json').read_text())
        self.assertEqual(comparison['primary_metric'], 'pooled_macro_f1')
        self.assertEqual(comparison['e6_pooled_macro_f1'], e6['pooled_metrics']['macro_f1'])
        self.assertEqual(comparison['e7_pooled_macro_f1'], e7['pooled_metrics']['macro_f1'])
        expected = 'E7' if comparison['e7_pooled_macro_f1'] > comparison['e6_pooled_macro_f1'] else 'E6'
        self.assertEqual(comparison['selected_experiment'], expected)


if __name__ == '__main__':
    unittest.main()
