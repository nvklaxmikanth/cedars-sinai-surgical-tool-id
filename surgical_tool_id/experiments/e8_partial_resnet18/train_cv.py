"""Offline E2 folds with ResNet18 layer 4 and head fine tuning.

    python experiments/e8_partial_resnet18/train_cv.py --cache
    python experiments/e8_partial_resnet18/train_cv.py --fold 0
    python experiments/e8_partial_resnet18/train_cv.py --aggregate
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import random
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

MODEL_PATH = Path(__file__).with_name("model.py")
model_spec = importlib.util.spec_from_file_location("e8_resnet18_model_for_training", MODEL_PATH)
model_module = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model_module)
WEIGHTS_SHA256 = model_module.WEIGHTS_SHA256
forward_from_layer3 = model_module.forward_from_layer3
forward_to_layer3 = model_module.forward_to_layer3
frozen_state_hash = model_module.frozen_state_hash
load_partial_model = model_module.load_partial_model
load_trainable_state = model_module.load_trainable_state
preprocess_image = model_module.preprocess_image
trainable_state = model_module.trainable_state


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MANIFEST = ROOT / "splits" / "video_grouped_fivefold_v1.json"
CACHE = OUT / "feature_cache.pt"
SEED = 42
BATCH_SIZE = 2
MAX_EPOCHS = 30
PATIENCE = 3
MIN_DELTA = 1e-4
LEARNING_RATE = 1e-3
LAYER4_LEARNING_RATE = LEARNING_RATE / 10


class ImageDataset(Dataset):
    def __init__(self, samples, root):
        self.samples = samples
        self.root = root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return preprocess_image(self.root / self.samples[index]["path"])


class FeatureDataset(Dataset):
    def __init__(self, features, indices, samples, classes):
        self.features = features
        self.indices = indices
        self.samples = samples
        self.class_to_idx = {name: i for i, name in enumerate(classes)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source = self.indices[index]
        return self.features[source], self.class_to_idx[self.samples[source]["label"]]


def classification_metrics(truth, predicted, classes):
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


def peak_rss_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_cache():
    start = time.perf_counter()
    set_seed()
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    samples = manifest["samples"]
    model = load_partial_model(num_classes=len(manifest["class_names"]))
    before = frozen_state_hash(model)
    if any(parameter.requires_grad for name, parameter in model.named_parameters()
           if not name.startswith(("layer4.", "fc."))):
        raise AssertionError("frozen prefix parameter is trainable")
    if any(module.training for module in model.modules() if isinstance(module, nn.BatchNorm2d)):
        raise AssertionError("BatchNorm is not frozen in eval mode")
    data_root = ROOT / manifest["data_root"]
    loader = DataLoader(ImageDataset(samples, data_root), batch_size=16, shuffle=False, num_workers=0)
    chunks = []
    with torch.inference_mode():
        for index, images in enumerate(loader):
            chunks.append(forward_to_layer3(model, images).cpu().clone())
            if index % 20 == 0:
                print(f"cached {min((index + 1) * 16, len(samples))}/{len(samples)}", flush=True)
    features = torch.cat(chunks)
    after = frozen_state_hash(model)
    if before != after:
        raise AssertionError("frozen prefix or BatchNorm state changed during feature extraction")
    payload = {
        "paths": [sample["path"] for sample in samples],
        "features": features,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "weights_sha256": WEIGHTS_SHA256,
        "backbone_state_sha256": before,
    }
    torch.save(payload, CACHE)
    metadata = {
        "n": len(samples), "feature_shape": list(features.shape[1:]),
        "feature_cache_sha256": sha256(CACHE),
        "manifest_sha256": payload["manifest_sha256"],
        "weights_sha256": WEIGHTS_SHA256,
        "backbone_before_sha256": before,
        "backbone_after_sha256": after,
        "runtime_seconds": time.perf_counter() - start,
        "peak_rss_mib": peak_rss_mib(),
    }
    (OUT / "feature_cache_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"features={tuple(features.shape)} runtime={metadata['runtime_seconds']:.3f}s "
          f"peak_rss={metadata['peak_rss_mib']:.1f}MiB", flush=True)


def load_cache(manifest_bytes):
    payload = torch.load(CACHE, map_location="cpu", weights_only=True)
    manifest = json.loads(manifest_bytes)
    if payload["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("feature cache split manifest mismatch")
    if payload["weights_sha256"] != WEIGHTS_SHA256:
        raise ValueError("feature cache pretrained weights mismatch")
    if payload["paths"] != [sample["path"] for sample in manifest["samples"]]:
        raise ValueError("feature cache sample paths mismatch")
    return payload


def predict_indices(model, loader):
    model.eval()
    indices = []
    with torch.no_grad():
        for features, _ in loader:
            indices.extend(forward_from_layer3(model, features).argmax(dim=1).tolist())
    return indices


def train_fold(fold_number):
    start = time.perf_counter()
    set_seed()
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    payload = load_cache(manifest_bytes)
    samples = manifest["samples"]
    classes = manifest["class_names"]
    index_by_path = {sample["path"]: i for i, sample in enumerate(samples)}
    fold = manifest["folds"][fold_number]
    train_indices = [index_by_path[path] for path in fold["train_paths"]]
    val_indices = [index_by_path[path] for path in fold["validation_paths"]]
    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    if set(fold["train_video_ids"]) & set(fold["validation_video_ids"]):
        raise ValueError("train/validation video leakage")
    features = payload["features"]
    train_loader = DataLoader(FeatureDataset(features, train_indices, samples, classes),
                              batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED), num_workers=0)
    val_loader = DataLoader(FeatureDataset(features, val_indices, samples, classes),
                            batch_size=16, shuffle=False, num_workers=0)
    model = load_partial_model(num_classes=len(classes))
    before = frozen_state_hash(model)
    if payload["backbone_state_sha256"] != before:
        raise ValueError("feature cache frozen-state hash mismatch")
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    batchnorm_parameter_names = {f"{module_name}.{parameter_name}"
                                 for module_name, module in model.named_modules()
                                 if isinstance(module, nn.BatchNorm2d)
                                 for parameter_name, _ in module.named_parameters(recurse=False)}
    expected_trainable = {name for name, _ in model.named_parameters()
                          if name.startswith(("layer4.", "fc."))
                          and name not in batchnorm_parameter_names}
    if set(trainable_names) != expected_trainable:
        raise AssertionError("only layer 4 and head may have gradients")
    if any(module.training for module in model.modules() if isinstance(module, nn.BatchNorm2d)):
        raise AssertionError("BatchNorm must remain in eval mode")
    optimizer = torch.optim.Adam([
        {"params": [parameter for parameter in model.layer4.parameters()
                    if parameter.requires_grad], "lr": LAYER4_LEARNING_RATE},
        {"params": model.fc.parameters(), "lr": LEARNING_RATE},
    ])
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
        model.eval()  # Autograd remains enabled; this freezes every BatchNorm buffer.
        total_loss = 0.0
        for batch_features, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(forward_from_layer3(model, batch_features), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_features.size(0)
        training_loss = total_loss / len(train_samples)
        val_indices_pred = predict_indices(model, val_loader)
        val_labels = [classes[i] for i in val_indices_pred]
        metrics = classification_metrics([s["label"] for s in val_samples], val_labels, classes)
        history.append({"epoch": epoch, "train_loss": training_loss,
                        "val_macro_f1": metrics["macro_f1"], "val_accuracy": metrics["accuracy"]})
        print(f"fold {fold_number} epoch {epoch:02d} train_loss={training_loss:.6f} "
              f"val_macro_f1={metrics['macro_f1']:.6f} val_accuracy={metrics['accuracy']:.6f}", flush=True)
        if metrics["macro_f1"] > best_f1 + 1e-12:
            best_f1 = metrics["macro_f1"]
            selected_epoch = epoch
            torch.save(trainable_state(model), checkpoint)
        if training_loss < best_training_loss - MIN_DELTA:
            best_training_loss = training_loss
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    after = frozen_state_hash(model)
    if before != after:
        raise AssertionError("frozen prefix or BatchNorm state changed during training")
    selected = load_partial_model(num_classes=len(classes))
    load_trainable_state(selected, checkpoint)
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
        "fold": fold_number, "seed": SEED,
        "manifest_sha256": payload["manifest_sha256"],
        "weights_sha256": WEIGHTS_SHA256,
        "feature_cache_sha256": sha256(CACHE),
        "frozen_state_before_sha256": before,
        "frozen_state_after_sha256": after,
        "feature_cache_frozen_state_sha256": payload["backbone_state_sha256"],
        "layer4_learning_rate": LAYER4_LEARNING_RATE,
        "head_learning_rate": LEARNING_RATE,
        "checkpoint_sha256": sha256(checkpoint),
        "selected_epoch": selected_epoch, "epochs_ran": len(history),
        "early_stopped": len(history) < MAX_EPOCHS,
        "validation_video_ids": fold["validation_video_ids"],
        "train_class_counts": dict(sorted(Counter(s["label"] for s in train_samples).items())),
        "class_weights": dict(zip(classes, class_weights.tolist())),
        "metrics": final_metrics, "history": history,
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
    if len(rows) != len(manifest["samples"]) or {r["path"] for r in rows} != {
        sample["path"] for sample in manifest["samples"]}:
        raise ValueError("OOF predictions do not cover each sample exactly once")
    rows.sort(key=lambda r: r["path"])
    with (OUT / "oof_predictions.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = classification_metrics([r["true_label"] for r in rows],
                                     [r["predicted_label"] for r in rows], classes)
    summary = {"seed": SEED, "class_names": classes,
               "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "weights_sha256": WEIGHTS_SHA256,
               "feature_cache_sha256": sha256(CACHE),
               "frozen_state_sha256": results[0]["frozen_state_before_sha256"],
               "pooled_metrics": metrics,
               "cache_runtime_seconds": json.loads((OUT / "feature_cache_meta.json").read_text())["runtime_seconds"],
               "sum_fold_runtime_seconds": sum(r["runtime_seconds"] for r in results),
               "max_fold_peak_rss_mib": max(r["peak_rss_mib"] for r in results),
               "folds": [{k: v for k, v in r.items() if k != "history"} for r in results]}
    (OUT / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    comparison = {"primary_metric": "pooled_macro_f1", "e8_pooled_macro_f1": metrics["macro_f1"],
                  "e8_pooled_accuracy": metrics["accuracy"]}
    for baseline in ("e7_frozen_resnet18", "e6_rgb_cv"):
        baseline_root = OUT.parent / baseline
        baseline_name = "e7" if baseline.startswith("e7") else "e6"
        baseline_summary = json.loads((baseline_root / "results.json").read_text())
        with (baseline_root / "oof_predictions.csv").open(newline="") as file:
            by_path = {row["path"]: row for row in csv.DictReader(file)}
        if set(by_path) != {row["path"] for row in rows}:
            raise ValueError(f"{baseline_name} and E8 OOF paths do not match")
        changed = [row for row in rows if row["predicted_label"] != by_path[row["path"]]["predicted_label"]]
        comparison[baseline_name] = {
            "pooled_macro_f1": baseline_summary["pooled_metrics"]["macro_f1"],
            "pooled_accuracy": baseline_summary["pooled_metrics"]["accuracy"],
            "changed_oof_predictions": len(changed),
            "wrong_to_correct": sum(by_path[row["path"]]["predicted_label"] != row["true_label"]
                                    and row["predicted_label"] == row["true_label"] for row in changed),
            "correct_to_wrong": sum(by_path[row["path"]]["predicted_label"] == row["true_label"]
                                    and row["predicted_label"] != row["true_label"] for row in changed),
        }
    comparison["selected_experiment"] = max(
        (("E8", metrics["macro_f1"]),
         ("E7", comparison["e7"]["pooled_macro_f1"]),
         ("E6", comparison["e6"]["pooled_macro_f1"])), key=lambda item: item[1])[0]
    (OUT / "comparison_to_e6_e7.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"OOF n={metrics['n']} accuracy={metrics['accuracy']:.6f} "
          f"macro_f1={metrics['macro_f1']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cache", action="store_true")
    group.add_argument("--fold", type=int, choices=range(5))
    group.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.cache:
        extract_cache()
    elif args.aggregate:
        aggregate()
    else:
        train_fold(args.fold)


if __name__ == "__main__":
    main()
