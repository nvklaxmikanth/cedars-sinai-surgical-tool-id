"""Train the E4 SmallCNN with fold-training-only class weights.

Run one fold per process for isolated peak-memory measurements:
    python experiments/e5_weighted_ce_cv/train_cv.py --fold 0
Then aggregate:
    python experiments/e5_weighted_ce_cv/train_cv.py --aggregate
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "splits" / "video_grouped_fivefold_v1.json"
OUT = Path(__file__).resolve().parent
SEED = 42
BATCH_SIZE = 2
MAX_EPOCHS = 30
PATIENCE = 3
MIN_DELTA = 1e-4
LEARNING_RATE = 1e-3


class SmallCNN(nn.Module):
    """Exact architecture from legacy/cnn_baseline_v2.py."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


class FrameDataset(Dataset):
    def __init__(self, samples, data_root, classes):
        self.samples = samples
        self.root = data_root
        self.class_to_idx = {name: i for i, name in enumerate(classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(self.root / sample["path"]).convert("L")
        image = image.resize((128, 128))
        tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).view(1, 128, 128) / 255.0
        tensor = tensor.repeat(3, 1, 1)
        label = self.class_to_idx[sample["label"]]
        return tensor, label


def classification_metrics(truth, predicted, classes):
    if len(truth) != len(predicted):
        raise ValueError("truth and predictions have different lengths")
    index = {name: i for i, name in enumerate(classes)}
    matrix = [[0 for _ in classes] for _ in classes]
    for actual, guess in zip(truth, predicted):
        matrix[index[actual]][index[guess]] += 1
    per_class = {}
    for i, name in enumerate(classes):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted_count = sum(row[i] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    return {
        "n": len(truth),
        "accuracy": sum(matrix[i][i] for i in range(len(classes))) / len(truth) if truth else 0.0,
        "macro_f1": sum(x["f1"] for x in per_class.values()) / len(classes),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def inverse_frequency_weights(train_samples, classes):
    """Inverse fold-training counts, rescaled so the class mean is one."""
    counts = Counter(sample["label"] for sample in train_samples)
    if any(counts[name] == 0 for name in classes):
        raise ValueError("every class needs a training sample for weighted loss")
    inverse = torch.tensor([1.0 / counts[name] for name in classes], dtype=torch.float32)
    return inverse / inverse.mean()


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def predict_indices(model, loader):
    model.eval()
    indices = []
    with torch.no_grad():
        for images, _ in loader:
            indices.extend(model(images).argmax(dim=1).tolist())
    return indices


def peak_rss_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024


def train_fold(fold_number):
    start = time.perf_counter()
    set_seed()
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    fold = manifest["folds"][fold_number]
    classes = manifest["class_names"]
    sample_by_path = {s["path"]: s for s in manifest["samples"]}
    train_samples = [sample_by_path[p] for p in fold["train_paths"]]
    val_samples = [sample_by_path[p] for p in fold["validation_paths"]]
    if set(fold["train_video_ids"]) & set(fold["validation_video_ids"]):
        raise ValueError("train/validation video leakage")
    data_root = ROOT / manifest["data_root"]
    train_loader = DataLoader(FrameDataset(train_samples, data_root, classes), batch_size=BATCH_SIZE,
                              shuffle=True, generator=torch.Generator().manual_seed(SEED), num_workers=0)
    val_loader = DataLoader(FrameDataset(val_samples, data_root, classes), batch_size=64,
                            shuffle=False, num_workers=0)
    model = SmallCNN(len(classes))  # fresh initialization per fold, CPU only
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    class_weights = inverse_frequency_weights(train_samples, classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_training_loss = float("inf")
    best_f1 = -1.0
    selected_epoch = None
    patience_left = PATIENCE
    history = []
    checkpoint = OUT / "checkpoints" / f"fold_{fold_number}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        training_loss = total_loss / len(train_samples)
        val_indices = predict_indices(model, val_loader)
        val_labels = [classes[i] for i in val_indices]
        metrics = classification_metrics([s["label"] for s in val_samples], val_labels, classes)
        history.append({"epoch": epoch, "train_loss": training_loss,
                        "val_macro_f1": metrics["macro_f1"], "val_accuracy": metrics["accuracy"]})
        print(f"fold {fold_number} epoch {epoch:02d} train_loss={training_loss:.6f} "
              f"val_macro_f1={metrics['macro_f1']:.6f} val_accuracy={metrics['accuracy']:.6f}", flush=True)
        if metrics["macro_f1"] > best_f1 + 1e-12:
            best_f1 = metrics["macro_f1"]
            selected_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
        if training_loss < best_training_loss - MIN_DELTA:
            best_training_loss = training_loss
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    selected = SmallCNN(len(classes))
    selected.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    final_indices = predict_indices(selected, val_loader)
    final_labels = [classes[i] for i in final_indices]
    final_metrics = classification_metrics([s["label"] for s in val_samples], final_labels, classes)
    if abs(final_metrics["macro_f1"] - best_f1) > 1e-12:
        raise AssertionError("selected checkpoint does not reproduce its validation F1")
    rows = [{"path": sample["path"], "video_id": sample["video_id"],
             "true_label": sample["label"], "predicted_index": index,
             "predicted_label": classes[index], "fold": fold_number}
            for sample, index in zip(val_samples, final_indices)]
    with (OUT / f"fold_{fold_number}_oof.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "fold": fold_number,
        "seed": SEED,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "selected_epoch": selected_epoch,
        "epochs_ran": len(history),
        "early_stopped": len(history) < MAX_EPOCHS,
        "validation_video_ids": fold["validation_video_ids"],
        "train_class_counts": dict(sorted(Counter(s["label"] for s in train_samples).items())),
        "class_weights": dict(zip(classes, class_weights.tolist())),
        "metrics": final_metrics,
        "history": history,
        "runtime_seconds": time.perf_counter() - start,
        "peak_rss_mib": peak_rss_mib(),
    }
    (OUT / f"fold_{fold_number}_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"fold {fold_number} selected_epoch={selected_epoch} macro_f1={best_f1:.6f} "
          f"runtime={result['runtime_seconds']:.3f}s peak_rss={result['peak_rss_mib']:.1f}MiB", flush=True)


def aggregate():
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    classes = manifest["class_names"]
    rows = []
    results = []
    for fold in range(manifest["n_folds"]):
        result = json.loads((OUT / f"fold_{fold}_result.json").read_text())
        if result["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest():
            raise ValueError(f"fold {fold} used a different split manifest")
        results.append(result)
        with (OUT / f"fold_{fold}_oof.csv").open(newline="") as file:
            rows.extend(csv.DictReader(file))
    if len(rows) != len(manifest["samples"]) or len({r["path"] for r in rows}) != len(rows):
        raise ValueError("OOF predictions do not cover each sample exactly once")
    sample_by_path = {s["path"]: s for s in manifest["samples"]}
    for row in rows:
        if row["path"] not in sample_by_path or row["true_label"] != sample_by_path[row["path"]]["label"]:
            raise ValueError(f"OOF row disagrees with manifest: {row['path']}")
    rows.sort(key=lambda r: r["path"])
    with (OUT / "oof_predictions.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = classification_metrics([r["true_label"] for r in rows],
                                     [r["predicted_label"] for r in rows], classes)
    summary = {"seed": SEED, "class_names": classes,
               "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "pooled_metrics": metrics,
               "sum_fold_runtime_seconds": sum(r["runtime_seconds"] for r in results),
               "max_fold_peak_rss_mib": max(r["peak_rss_mib"] for r in results),
               "folds": [{k: v for k, v in r.items() if k != "history"} for r in results]}
    (OUT / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    e4_root = OUT.parent / "e4_cross_entropy_cv"
    e4_summary = json.loads((e4_root / "results.json").read_text())
    with (e4_root / "oof_predictions.csv").open(newline="") as file:
        e4_by_path = {row["path"]: row for row in csv.DictReader(file)}
    if set(e4_by_path) != {row["path"] for row in rows}:
        raise ValueError("E4 and E5 OOF paths do not match")
    changed = [row for row in rows if row["predicted_label"] != e4_by_path[row["path"]]["predicted_label"]]
    comparison = {
        "primary_metric": "pooled_macro_f1",
        "e4_pooled_macro_f1": e4_summary["pooled_metrics"]["macro_f1"],
        "e5_pooled_macro_f1": metrics["macro_f1"],
        "selected_experiment": "E5" if metrics["macro_f1"] > e4_summary["pooled_metrics"]["macro_f1"] else "E4",
        "e4_pooled_accuracy": e4_summary["pooled_metrics"]["accuracy"],
        "e5_pooled_accuracy": metrics["accuracy"],
        "changed_oof_predictions": len(changed),
        "wrong_to_correct": sum(e4_by_path[row["path"]]["predicted_label"] != row["true_label"]
                                and row["predicted_label"] == row["true_label"] for row in changed),
        "correct_to_wrong": sum(e4_by_path[row["path"]]["predicted_label"] == row["true_label"]
                                and row["predicted_label"] != row["true_label"] for row in changed),
    }
    (OUT / "comparison_to_e4.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"OOF n={metrics['n']} accuracy={metrics['accuracy']:.6f} "
          f"macro_f1={metrics['macro_f1']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fold", type=int, choices=range(5))
    group.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    else:
        train_fold(args.fold)


if __name__ == "__main__":
    main()
