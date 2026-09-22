import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "splits"))
from video_folds import N_FOLDS, build_manifest, load_samples


class VideoFoldTests(unittest.TestCase):
    def synthetic_samples(self):
        classes = ("clipper", "grasper", "hook", "scissor")
        return [{"path": f"train/{label}/{video}_{index}.png",
                 "label": label, "video_id": str(video)}
                for video in range(10, 15) for index, label in enumerate(classes)]

    def test_no_video_leakage_and_one_validation_appearance(self):
        manifest = build_manifest(self.synthetic_samples())
        by_path = {sample["path"]: sample for sample in manifest["samples"]}
        seen = Counter()
        self.assertEqual(len(manifest["folds"]), N_FOLDS)
        for fold in manifest["folds"]:
            train, validation = set(fold["train_paths"]), set(fold["validation_paths"])
            self.assertFalse(train & validation)
            self.assertEqual(train | validation, set(by_path))
            self.assertFalse({by_path[p]["video_id"] for p in train} &
                             {by_path[p]["video_id"] for p in validation})
            seen.update(validation)
        self.assertEqual(seen, Counter({path: 1 for path in by_path}))

    def test_local_dataset_discovery_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for video in range(10, 15):
                for index, label in enumerate(("clipper", "grasper", "hook", "scissor")):
                    path = root / "train" / label / f"{video}_{index}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (2, 2)).save(path)
            samples = load_samples(root)
            self.assertEqual(len(samples), 20)
            self.assertEqual(build_manifest(samples)["class_names"],
                             ["clipper", "grasper", "hook", "scissor"])

    def test_invalid_filename_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train" / "clipper" / "private.png"
            path.parent.mkdir(parents=True)
            Image.new("RGB", (2, 2)).save(path)
            with self.assertRaisesRegex(ValueError, "filename lacks a video/frame ID"):
                load_samples(Path(tmp))


if __name__ == "__main__":
    unittest.main()
