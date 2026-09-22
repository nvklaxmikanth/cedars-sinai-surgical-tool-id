"""E10 final fits: fixed nine epochs on every E2 sample, no validation."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E8 = HERE.parent / "e8_partial_resnet18"
E9 = HERE.parent / "e9_seed_ensemble"
spec = importlib.util.spec_from_file_location("e8_train_for_e10", E8 / "train_cv.py")
e8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e8)
SEEDS = (17, 42, 123)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_epochs():
    values = []
    for seed in SEEDS:
        for fold in range(5):
            path = (E8 / f"fold_{fold}_result.json" if seed == 42 else
                    E9 / f"seed_{seed}" / f"fold_{fold}_result.json")
            result = json.loads(path.read_text())
            if result["seed"] != seed or result["fold"] != fold:
                raise ValueError(f"E9 result identity mismatch: {path}")
            values.append(result["selected_epoch"])
    median = statistics.median(values)
    if not isinstance(median, int):
        raise ValueError("median selected epoch is not integral")
    return values, median, median + 1


def train(seed):
    if seed not in SEEDS:
        raise ValueError("unexpected seed")
    start = time.perf_counter()
    epochs, median_index, n_epochs = selected_epochs()
    e8.set_seed(seed)
    manifest_bytes = e8.MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    samples, classes = manifest["samples"], manifest["class_names"]
    if len(samples) != 1402 or classes != ["clipper", "grasper", "hook", "scissor"]:
        raise ValueError("E2 sample count or class order changed")
    payload = e8.load_cache(manifest_bytes)
    features = payload["features"]
    loader = DataLoader(e8.FeatureDataset(features, list(range(len(samples))), samples, classes),
                        batch_size=e8.BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    model = e8.load_partial_model(num_classes=len(classes))
    before = e8.frozen_state_hash(model)
    if before != payload["backbone_state_sha256"]:
        raise ValueError("frozen feature cache state mismatch")
    if any(m.training for m in model.modules() if isinstance(m, nn.BatchNorm2d)):
        raise AssertionError("BatchNorm state must be frozen")
    weights = e8.inverse_frequency_weights(samples, classes)
    optimizer = torch.optim.Adam([
        {"params": [p for p in model.layer4.parameters() if p.requires_grad],
         "lr": e8.LAYER4_LEARNING_RATE},
        {"params": model.fc.parameters(), "lr": e8.LEARNING_RATE},
    ])
    criterion = nn.CrossEntropyLoss(weight=weights)
    losses = []
    for epoch in range(n_epochs):
        model.eval()
        total = 0.0
        for batch_features, targets in loader:
            optimizer.zero_grad()
            loss = criterion(e8.forward_from_layer3(model, batch_features), targets)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(targets)
        losses.append(total / len(samples))
        print(f"seed={seed} epoch={epoch} training_loss={losses[-1]:.6f}", flush=True)
    after = e8.frozen_state_hash(model)
    if before != after:
        raise AssertionError("frozen tensors or BatchNorm state changed")
    checkpoint = HERE / "checkpoints" / f"seed_{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint)
    result = {
        "seed": seed, "n_samples": len(samples), "selected_epochs_zero_based": epochs,
        "median_selected_epoch_zero_based": median_index, "epochs_ran": n_epochs,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "weights_sha256": e8.WEIGHTS_SHA256,
        "feature_cache_sha256": e8.sha256(e8.CACHE),
        "frozen_state_before_sha256": before, "frozen_state_after_sha256": after,
        "class_names": classes,
        "class_counts": dict(sorted(Counter(s["label"] for s in samples).items())),
        "class_weights": dict(zip(classes, weights.tolist())),
        "batch_size": e8.BATCH_SIZE,
        "layer4_learning_rate": e8.LAYER4_LEARNING_RATE,
        "head_learning_rate": e8.LEARNING_RATE,
        "training_losses": losses,
        "checkpoint_sha256": sha256(checkpoint),
        "runtime_seconds": time.perf_counter() - start,
        "peak_rss_mib": e8.peak_rss_mib(),
    }
    (HERE / f"seed_{seed}_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"seed={seed} checkpoint={result['checkpoint_sha256']} "
          f"runtime={result['runtime_seconds']:.3f}s peak_rss={result['peak_rss_mib']:.1f}MiB", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    train(parser.parse_args().seed)
