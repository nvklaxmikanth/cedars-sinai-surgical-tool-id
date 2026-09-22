import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
E5 = ROOT / 'experiments' / 'e5_weighted_ce_cv'
E6 = ROOT / 'experiments' / 'e6_rgb_cv'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


grayscale = load_module('e5_train_for_e6_test', E5 / 'train_cv.py')
rgb = load_module('e6_train_for_e6_test', E6 / 'train_cv.py')


class RgbCvTests(unittest.TestCase):
    def test_rgb_channels_and_scaling_match_pil_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'color.png'
            Image.new('RGB', (128, 128), (255, 128, 0)).save(path)
            sample = [{'path': 'color.png', 'label': 'clipper'}]
            image, label = rgb.FrameDataset(sample, root,
                                            ['clipper', 'grasper', 'hook', 'scissor'])[0]
            gray, gray_label = grayscale.FrameDataset(sample, root,
                                                       ['clipper', 'grasper', 'hook', 'scissor'])[0]
            self.assertEqual(image.shape, (3, 128, 128))
            self.assertEqual(image.dtype, torch.float32)
            self.assertTrue(torch.all(image[0] == 1.0))
            self.assertTrue(torch.all(image[1] == torch.tensor(128 / 255, dtype=torch.float32)))
            self.assertTrue(torch.all(image[2] == 0.0))
            self.assertTrue(torch.equal(gray[0], gray[1]))
            self.assertTrue(torch.equal(gray[1], gray[2]))
            self.assertFalse(torch.equal(image[0], image[1]))
            self.assertEqual(label, gray_label)

    def test_real_image_uses_rgb_resize_without_other_transform(self):
        manifest = json.loads((ROOT / 'splits' / 'video_grouped_fivefold_v1.json').read_text())
        sample = manifest['samples'][0]
        root = ROOT / manifest['data_root']
        actual, _ = rgb.FrameDataset([sample], root, manifest['class_names'])[0]
        image = Image.open(root / sample['path']).convert('RGB').resize((128, 128))
        expected = torch.tensor(list(image.getdata()), dtype=torch.float32)
        expected = expected.view(128, 128, 3).permute(2, 0, 1) / 255.0
        self.assertTrue(torch.equal(actual, expected))

    def test_architecture_settings_and_fold_weights_match_e5(self):
        manifest = json.loads((ROOT / 'splits' / 'video_grouped_fivefold_v1.json').read_text())
        sample_by_path = {sample['path']: sample for sample in manifest['samples']}
        for name in ('SEED', 'BATCH_SIZE', 'MAX_EPOCHS', 'PATIENCE', 'MIN_DELTA', 'LEARNING_RATE'):
            self.assertEqual(getattr(grayscale, name), getattr(rgb, name))
        self.assertEqual({k: tuple(v.shape) for k, v in grayscale.SmallCNN(4).state_dict().items()},
                         {k: tuple(v.shape) for k, v in rgb.SmallCNN(4).state_dict().items()})
        for fold in manifest['folds']:
            samples = [sample_by_path[path] for path in fold['train_paths']]
            old_weights = grayscale.inverse_frequency_weights(samples, manifest['class_names'])
            new_weights = rgb.inverse_frequency_weights(samples, manifest['class_names'])
            self.assertTrue(torch.equal(old_weights, new_weights))
            result = json.loads((E6 / f"fold_{fold['fold']}_result.json").read_text())
            self.assertTrue(torch.equal(new_weights, torch.tensor([
                result['class_weights'][name] for name in manifest['class_names']])))

    def test_oof_metrics_and_selected_checkpoints_reproduce(self):
        manifest = json.loads((ROOT / 'splits' / 'video_grouped_fivefold_v1.json').read_text())
        summary = json.loads((E6 / 'results.json').read_text())
        by_path = {sample['path']: sample for sample in manifest['samples']}
        with (E6 / 'oof_predictions.csv').open(newline='') as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), len(by_path))
        self.assertEqual({row['path'] for row in rows}, set(by_path))
        self.assertEqual(rgb.classification_metrics(
            [row['true_label'] for row in rows],
            [row['predicted_label'] for row in rows], summary['class_names']),
            summary['pooled_metrics'])
        for fold in summary['folds']:
            number = fold['fold']
            history = json.loads((E6 / f'fold_{number}_result.json').read_text())['history']
            self.assertEqual(history[fold['selected_epoch']]['val_macro_f1'],
                             max(epoch['val_macro_f1'] for epoch in history))
            checkpoint = E6 / 'checkpoints' / f'fold_{number}.pt'
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             fold['checkpoint_sha256'])
            model = rgb.SmallCNN(4)
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
            samples = [by_path[path] for path in manifest['folds'][number]['validation_paths']]
            loader = DataLoader(rgb.FrameDataset(samples, ROOT / manifest['data_root'],
                                                 summary['class_names']), batch_size=64)
            indices = rgb.predict_indices(model, loader)
            with (E6 / f'fold_{number}_oof.csv').open(newline='') as file:
                fold_rows = list(csv.DictReader(file))
            self.assertEqual([row['path'] for row in fold_rows],
                             manifest['folds'][number]['validation_paths'])
            self.assertEqual(indices, [int(row['predicted_index']) for row in fold_rows])

    def test_primary_selection_uses_pooled_macro_f1(self):
        comparison = json.loads((E6 / 'comparison_to_e5.json').read_text())
        e5 = json.loads((E5 / 'results.json').read_text())
        e6 = json.loads((E6 / 'results.json').read_text())
        self.assertEqual(comparison['primary_metric'], 'pooled_macro_f1')
        self.assertEqual(comparison['e5_pooled_macro_f1'], e5['pooled_metrics']['macro_f1'])
        self.assertEqual(comparison['e6_pooled_macro_f1'], e6['pooled_metrics']['macro_f1'])
        expected = 'E6' if comparison['e6_pooled_macro_f1'] > comparison['e5_pooled_macro_f1'] else 'E5'
        self.assertEqual(comparison['selected_experiment'], expected)


if __name__ == '__main__':
    unittest.main()
