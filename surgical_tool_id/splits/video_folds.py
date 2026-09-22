"""Build a deterministic, class-balanced five-fold split by video ID.

Run from the repository's surgical_tool_id directory:
    python splits/video_folds.py
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


VERSION = 1
N_FOLDS = 5
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "cholec-tinytools"
MANIFEST_PATH = Path(__file__).with_name("video_grouped_fivefold_v1.json")
FILENAME = re.compile(r"^(\d+)_(\d+)\.png$", re.IGNORECASE)


def load_samples(root=DATA_ROOT):
    root = Path(root)
    samples = []
    for path in sorted(root.glob("*/*/*.png"), key=lambda p: p.relative_to(root).as_posix()):
        match = FILENAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"filename lacks a video/frame ID: {path}")
        samples.append({
            "path": path.relative_to(root).as_posix(),
            "label": path.parent.name,
            "video_id": match.group(1),
        })
    if not samples:
        raise ValueError(f"no PNG samples found in {root}")
    return samples


def assignments(n_groups, n_folds=N_FOLDS):
    """Yield each unlabeled partition once, using restricted-growth IDs."""
    assignment = [0] * n_groups

    def visit(index, largest):
        if index == n_groups:
            if largest + 1 == n_folds:
                yield tuple(assignment)
            return
        for fold in range(min(largest + 1, n_folds - 1) + 1):
            assignment[index] = fold
            yield from visit(index + 1, max(largest, fold))

    yield from visit(1, 0)


def fold_summary(samples, video_ids_by_fold):
    counts = []
    for videos in video_ids_by_fold:
        ids = set(videos)
        counts.append(Counter(s["label"] for s in samples if s["video_id"] in ids))
    return counts


def balance_score(counts, totals, classes):
    """Sum squared relative deviations from one fifth of every class."""
    return sum(((count[c] - totals[c] / N_FOLDS) / (totals[c] / N_FOLDS)) ** 2
               for count in counts for c in classes)


def build_manifest(samples):
    paths = [s["path"] for s in samples]
    if len(paths) != len(set(paths)):
        raise ValueError("sample paths are not unique")
    groups = defaultdict(Counter)
    for sample in samples:
        groups[sample["video_id"]][sample["label"]] += 1
    videos = sorted(groups)
    classes = sorted({s["label"] for s in samples})
    if len(videos) < N_FOLDS:
        raise ValueError(f"need at least {N_FOLDS} video IDs, found {len(videos)}")
    totals = Counter(s["label"] for s in samples)
    best = None
    feasible_partitions = 0
    for assignment in assignments(len(videos)):
        video_folds = [tuple(v for v, f in zip(videos, assignment) if f == i)
                       for i in range(N_FOLDS)]
        counts = [sum((groups[v] for v in fold), Counter()) for fold in video_folds]
        if not all(count[c] > 0 and totals[c] - count[c] > 0
                   for count in counts for c in classes):
            continue
        feasible_partitions += 1
        score = balance_score(counts, totals, classes)
        candidate = (score, video_folds)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("no five-fold video partition contains every class on both sides")

    score, video_folds = best
    all_videos = set(videos)
    folds = []
    for index, held_out in enumerate(video_folds):
        held_out_set = set(held_out)
        folds.append({
            "fold": index,
            "validation_video_ids": list(held_out),
            "train_video_ids": sorted(all_videos - held_out_set),
            "validation_paths": [s["path"] for s in samples if s["video_id"] in held_out_set],
            "train_paths": [s["path"] for s in samples if s["video_id"] not in held_out_set],
        })
    return {
        "version": VERSION,
        "method": "exhaustive video-group partition; all classes on both sides; minimum squared relative class-count deviation",
        "data_root": "data/cholec-tinytools",
        "n_folds": N_FOLDS,
        "class_names": classes,
        "video_ids": videos,
        "class_counts": dict(sorted(totals.items())),
        "feasible_partitions": feasible_partitions,
        "balance_score": score,
        "samples": samples,
        "folds": folds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT,
                        help="authorized local dataset root containing partition/class/PNG")
    parser.add_argument("--out", type=Path, default=MANIFEST_PATH,
                        help="local generated manifest path (ignored by Git by default)")
    args = parser.parse_args()
    manifest = build_manifest(load_samples(args.data_root))
    manifest["data_root"] = str(args.data_root.resolve())
    content = json.dumps(manifest, indent=2) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content)
    print(f"wrote {args.out} ({len(manifest['samples'])} samples, "
          f"{manifest['feasible_partitions']} feasible partitions, "
          f"score={manifest['balance_score']:.6f}, "
          f"sha256={hashlib.sha256(content.encode()).hexdigest()})")


if __name__ == "__main__":
    main()
