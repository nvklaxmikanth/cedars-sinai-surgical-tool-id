import csv
import importlib.abc
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contract import list_expected_filenames, validate_schema
from predict import InferenceDataset, discover_images, main


ROOT = Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    def test_recursive_discovery_is_sorted_and_disambiguates_only_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("b/same.png", "a/same.png", "a/unique.png", "a/deep/later.PNG"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (2, 2)).save(path)
            expected = ["later.PNG", "a/same.png", "unique.png", "b/same.png"]
            self.assertEqual([ident for _, ident in discover_images(root)], expected)
            self.assertEqual(list_expected_filenames(root), expected)
            self.assertEqual([InferenceDataset(root)[i][1] for i in range(4)], expected)

    def test_clear_input_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "no PNG images found"):
                discover_images(root)
            with self.assertRaisesRegex(ValueError, "does not exist"):
                discover_images(root / "missing")
            (root / "bad.png").write_bytes(b"not an image")
            with self.assertRaisesRegex(ValueError, "cannot read PNG image"):
                InferenceDataset(root)[0]

    def test_schema_rejects_duplicate_and_extra_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pred.csv"
            for rows, reason in (
                ([('a.png', 'hook'), ('a.png', 'hook')], 'DUPLICATE_FILENAME'),
                ([('a.png', 'hook'), ('extra.png', 'hook')], 'UNEXPECTED_FILENAMES'),
            ):
                with out.open('w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['filename', 'predicted_class'])
                    writer.writerows(rows)
                self.assertIn(reason, validate_schema(out, ['a.png', 'b.png'], ['hook'])['reason'])

    def test_cli_output_keeps_both_duplicate_basenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('a/same.png', 'b/same.png'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (128, 86), 'red').save(path)
            out = root / 'predictions.csv'
            original_argv = sys.argv
            sys.argv = ['predict.py', '--data-dir', str(root), '--out', str(out)]
            try:
                with redirect_stdout(StringIO()):
                    main()
            finally:
                sys.argv = original_argv
            with out.open(newline='') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row['filename'] for row in rows], ['a/same.png', 'b/same.png'])
            self.assertTrue(validate_schema(out, list_expected_filenames(root),
                                            ['grasper', 'hook', 'clipper', 'scissor'])['ok'])

    def test_cli_runs_without_training_only_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new('RGB', (128, 86), 'red').save(root / 'one.png')
            out = root / 'out.csv'
            class Block(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {'pandas', 'sklearn', 'cv2', 'torchvision', 'yaml'}:
                        raise ImportError('blocked optional dependency: ' + fullname)

            original_argv = sys.argv
            sys.argv = [str(ROOT / 'predict.py'), '--data-dir', str(root), '--out', str(out)]
            blocker = Block()
            sys.meta_path.insert(0, blocker)
            try:
                with redirect_stdout(StringIO()):
                    runpy.run_path(sys.argv[0], run_name='__main__')
            finally:
                sys.meta_path.remove(blocker)
                sys.argv = original_argv
            with out.open(newline='') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['filename'], 'one.png')


if __name__ == '__main__':
    unittest.main()
