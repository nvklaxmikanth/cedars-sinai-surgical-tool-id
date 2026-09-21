import json
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'splits'))
from video_folds import MANIFEST_PATH, N_FOLDS, build_manifest, load_samples


class VideoFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST_PATH.read_text())

    def test_complete_paths_and_folder_labels(self):
        samples = self.manifest['samples']
        self.assertEqual(len(samples), 1402)
        self.assertEqual(len({sample['path'] for sample in samples}), len(samples))
        for sample in samples:
            path = ROOT / self.manifest['data_root'] / sample['path']
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.parent.name, sample['label'])
            self.assertEqual(path.stem.split('_')[0], sample['video_id'])

    def test_no_video_leakage_and_one_validation_appearance(self):
        samples = self.manifest['samples']
        all_paths = {sample['path'] for sample in samples}
        by_path = {sample['path']: sample for sample in samples}
        seen = Counter()
        self.assertEqual(len(self.manifest['folds']), N_FOLDS)
        for fold in self.manifest['folds']:
            train = set(fold['train_paths'])
            validation = set(fold['validation_paths'])
            self.assertEqual(len(train), len(fold['train_paths']))
            self.assertEqual(len(validation), len(fold['validation_paths']))
            self.assertFalse(train & validation)
            self.assertEqual(train | validation, all_paths)
            train_videos = {by_path[p]['video_id'] for p in train}
            validation_videos = {by_path[p]['video_id'] for p in validation}
            self.assertFalse(train_videos & validation_videos)
            self.assertEqual(train_videos, set(fold['train_video_ids']))
            self.assertEqual(validation_videos, set(fold['validation_video_ids']))
            seen.update(validation)
            for side in (train, validation):
                self.assertEqual({by_path[p]['label'] for p in side},
                                 set(self.manifest['class_names']))
        self.assertEqual(seen, Counter({path: 1 for path in all_paths}))

    def test_manifest_reproducibility(self):
        generated = build_manifest(load_samples())
        self.assertEqual(json.dumps(generated, indent=2) + '\n', MANIFEST_PATH.read_text())


if __name__ == '__main__':
    unittest.main()
