"""Shared submission contract, used identically by check_submission.py
(ships in the candidate zip, runs against their own validation/ folder) and
grading/rig/run.py (runs against the private holdout). Keeping this one
file as the single source of truth for both is what guarantees "if
preflight passes on your machine, the rig will run your code" is actually
true rather than aspirational - the two never drift because they are
running the literal same code.

Deliberately dependency-light (stdlib + csv only) so it never becomes a
reason a submission fails to even be checked.
"""
import csv
import subprocess
import sys
import time
from pathlib import Path

EXPECTED_HEADER = ["filename", "predicted_class"]
TIMEOUT_SECONDS = 30 * 60


def find_entry_point(submission_root: Path):
    """Locate predict.py: at the root, or as the single unambiguous match
    one level down (handles 'zipped the parent folder'). Returns a Path or
    raises FileNotFoundError / ValueError with a message describing why."""
    submission_root = Path(submission_root)
    direct = submission_root / "predict.py"
    if direct.is_file():
        return direct

    one_level_down = list(submission_root.glob("*/predict.py"))
    if len(one_level_down) == 1:
        return one_level_down[0]
    if len(one_level_down) > 1:
        raise ValueError(
            f"found {len(one_level_down)} predict.py candidates one level down "
            f"({[str(p) for p in one_level_down]}) - ambiguous, not guessing"
        )
    raise FileNotFoundError("no predict.py found at submission root or one level down")


def run_predict(entry_point: Path, data_dir: Path, out_csv: Path, python_exe=None, timeout=TIMEOUT_SECONDS):
    """Invokes `python predict.py --data-dir <data_dir> --out <out_csv>` as
    a subprocess, CWD set to the entry point's directory. Never imports the
    submission's code. Returns (returncode, stdout, stderr, elapsed_seconds)
    or raises subprocess.TimeoutExpired."""
    python_exe = python_exe or sys.executable
    cwd = entry_point.parent
    cmd = [python_exe, entry_point.name, "--data-dir", str(data_dir), "--out", str(out_csv)]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise
    elapsed = time.time() - start
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def validate_schema(csv_path: Path, expected_filenames, class_names):
    """Plumbing-only validation: header, row count, and vocabulary. Never
    checks whether predictions are *correct* - that's what scoring is for,
    and preflight must not leak correctness signal.

    Returns a dict: {"ok": bool, "reason": str or None, "n_rows": int}
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return {"ok": False, "reason": "NO_OUTPUT_FILE", "n_rows": 0}

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return {"ok": False, "reason": "EMPTY_CSV", "n_rows": 0}
        if header != EXPECTED_HEADER:
            return {"ok": False, "reason": f"BAD_SCHEMA: header={header}, expected={EXPECTED_HEADER}", "n_rows": 0}
        rows = list(reader)

    predicted = {}
    for row in rows:
        if len(row) != 2:
            return {"ok": False, "reason": f"BAD_SCHEMA: malformed row {row}", "n_rows": len(rows)}
        fname, cls = row
        predicted[fname] = cls

    unknown = set(predicted.values()) - set(class_names)
    if unknown:
        return {"ok": False, "reason": f"UNKNOWN_LABELS: {sorted(unknown)}", "n_rows": len(rows)}

    missing = set(expected_filenames) - set(predicted)
    if missing:
        return {
            "ok": False,
            "reason": f"INCOMPLETE: missing predictions for {len(missing)}/{len(expected_filenames)} files",
            "n_rows": len(rows),
        }

    return {"ok": True, "reason": None, "n_rows": len(rows)}


def list_expected_filenames(data_dir: Path):
    data_dir = Path(data_dir)
    subdirs = [p for p in data_dir.iterdir() if p.is_dir()]
    names = []
    if subdirs:
        for d in subdirs:
            names.extend(f.name for f in d.glob("*.png"))
    else:
        names.extend(f.name for f in data_dir.glob("*.png"))
    return names


def discover_class_names(data_dir: Path):
    return sorted(p.name for p in Path(data_dir).iterdir() if p.is_dir())
