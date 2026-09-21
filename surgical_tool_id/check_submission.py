"""Preflight checker - run this before you submit.

It runs the exact contract we will run on our end: it calls your
predict.py the same way our grader will (as a subprocess, with
--data-dir and --out), against your own validation/ folder, and checks
that the output CSV has the right shape and vocabulary.

    python check_submission.py

If this prints "PREFLIGHT PASSED", our grader will be able to run your
code. This script checks plumbing only - it does NOT check whether your
predictions are any good. A model that is wrong about everything can still
pass preflight; that's by design, so passing this tells you nothing about
your score, only that we'll be able to compute one.
"""
import sys
import tempfile
from pathlib import Path

from contract import (
    discover_class_names,
    find_entry_point,
    list_expected_filenames,
    run_predict,
    validate_schema,
)

ROOT = Path(__file__).parent
VAL_DIR = ROOT / "data" / "cholec-tinytools" / "validation"


def main():
    print(f"submission root: {ROOT}")
    print(f"checking against: {VAL_DIR}")

    try:
        entry_point = find_entry_point(ROOT)
    except (FileNotFoundError, ValueError) as e:
        print(f"PREFLIGHT FAILED: {e}")
        sys.exit(1)
    print(f"entry point: {entry_point}")

    class_names = discover_class_names(VAL_DIR)
    expected_files = list_expected_filenames(VAL_DIR)
    print(f"{len(expected_files)} images across {len(class_names)} classes: {class_names}")

    with tempfile.TemporaryDirectory() as tmp:
        out_csv = Path(tmp) / "preds.csv"
        try:
            returncode, stdout, stderr, elapsed = run_predict(entry_point, VAL_DIR, out_csv)
        except Exception as e:
            print(f"PREFLIGHT FAILED: predict.py raised/timed out: {e}")
            sys.exit(1)

        print(f"predict.py exited {returncode} in {elapsed:.1f}s")
        if returncode != 0:
            print("--- stdout ---")
            print(stdout[-2000:])
            print("--- stderr ---")
            print(stderr[-2000:])
            print("PREFLIGHT FAILED: predict.py exited non-zero")
            sys.exit(1)

        result = validate_schema(out_csv, expected_files, class_names)

    if not result["ok"]:
        print(f"PREFLIGHT FAILED: {result['reason']}")
        sys.exit(1)

    print(f"wrote {result['n_rows']} valid predictions.")
    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
