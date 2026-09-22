"""Replay E8/E9 checkpoints and evaluate aligned seed probabilities.

    python experiments/e9_seed_ensemble/aggregate.py
"""

import csv
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
E8 = HERE.parent / "e8_partial_resnet18"
spec = importlib.util.spec_from_file_location("e9_seed_trainer_for_aggregation",
                                              HERE / "train_seeds.py")
seed_trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed_trainer)
e8 = seed_trainer.e8
SEEDS = (17, 42, 123)


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric_stats(metrics):
    def describe(values):
        return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}

    classes = metrics[0]["per_class"]
    return {
        "accuracy": describe([item["accuracy"] for item in metrics]),
        "macro_f1": describe([item["macro_f1"] for item in metrics]),
        "per_class": {
            name: {key: describe([item["per_class"][name][key] for item in metrics])
                   for key in ("precision", "recall", "f1")}
            for name in classes
        },
    }


def validate_alignment(rows_by_seed, manifest):
    """Reject any missing path, duplicate path, or differing truth/fold metadata."""
    samples = {sample["path"]: sample for sample in manifest["samples"]}
    expected_fold = {path: number for number, fold in enumerate(manifest["folds"])
                     for path in fold["validation_paths"]}
    if set(expected_fold) != set(samples):
        raise ValueError("manifest does not give each sample one validation fold")
    indexed = {}
    for seed, rows in rows_by_seed.items():
        if len(rows) != len(samples) or len({row["path"] for row in rows}) != len(rows):
            raise ValueError(f"seed {seed}: missing or duplicate sample path")
        by_path = {row["path"]: row for row in rows}
        if set(by_path) != set(samples):
            raise ValueError(f"seed {seed}: sample path set differs from manifest")
        for path, row in by_path.items():
            if row["true_label"] != samples[path]["label"] or int(row["fold"]) != expected_fold[path]:
                raise ValueError(f"seed {seed}: truth or fold mismatch at {path}")
            if row["video_id"] != samples[path]["video_id"]:
                raise ValueError(f"seed {seed}: video ID mismatch at {path}")
        indexed[seed] = by_path
    return indexed


def replay_seed(seed, manifest, cache, expected_frozen_hash):
    """Recompute every held-out probability from its selected checkpoint."""
    source = E8 if seed == 42 else HERE / f"seed_{seed}"
    classes = manifest["class_names"]
    samples = manifest["samples"]
    sample_index = {sample["path"]: i for i, sample in enumerate(samples)}
    all_rows = []
    fold_results = []
    replay_start = time.perf_counter()
    for fold_number, fold in enumerate(manifest["folds"]):
        result = json.loads((source / f"fold_{fold_number}_result.json").read_text())
        if result["seed"] != seed or result["fold"] != fold_number:
            raise ValueError("seed or fold metadata mismatch")
        if result["frozen_state_before_sha256"] != expected_frozen_hash or \
           result["frozen_state_after_sha256"] != expected_frozen_hash:
            raise ValueError("frozen parameter or BatchNorm tensor changed during training")
        if result["feature_cache_sha256"] != sha256(e8.CACHE):
            raise ValueError("feature cache differs from training run")
        checkpoint = source / "checkpoints" / f"fold_{fold_number}.pt"
        if sha256(checkpoint) != result["checkpoint_sha256"]:
            raise ValueError("checkpoint hash mismatch")
        model = e8.load_partial_model(num_classes=len(classes))
        if e8.frozen_state_hash(model) != expected_frozen_hash:
            raise ValueError("loaded pretrained frozen state differs from training")
        e8.load_trainable_state(model, checkpoint)
        validation_indices = [sample_index[path] for path in fold["validation_paths"]]
        loader = DataLoader(e8.FeatureDataset(cache["features"], validation_indices,
                                              samples, classes), batch_size=16,
                            shuffle=False, num_workers=0)
        probabilities = []
        with torch.inference_mode():
            for features, _ in loader:
                probabilities.extend(torch.softmax(e8.forward_from_layer3(model, features),
                                                   dim=1).tolist())
        if e8.frozen_state_hash(model) != expected_frozen_hash:
            raise ValueError("frozen parameter or BatchNorm tensor changed during replay")
        original_rows = read_csv(source / f"fold_{fold_number}_oof.csv")
        if len(original_rows) != len(validation_indices):
            raise ValueError("fold OOF row count mismatch")
        replay_rows = []
        for path, original, probs in zip(fold["validation_paths"], original_rows, probabilities):
            sample = samples[sample_index[path]]
            predicted_index = max(range(len(classes)), key=lambda index: probs[index])
            if (original["path"] != path or original["true_label"] != sample["label"] or
                    int(original["fold"]) != fold_number or
                    int(original["predicted_index"]) != predicted_index or
                    original["predicted_label"] != classes[predicted_index]):
                raise ValueError(f"seed {seed} fold {fold_number}: checkpoint replay differs at {path}")
            row = {"path": path, "video_id": sample["video_id"],
                   "true_label": sample["label"], "fold": fold_number,
                   "predicted_index": predicted_index,
                   "predicted_label": classes[predicted_index]}
            row.update({f"p_{name}": format(float(probability), ".17g")
                        for name, probability in zip(classes, probs)})
            replay_rows.append(row)
        replay_metrics = e8.classification_metrics(
            [row["true_label"] for row in replay_rows],
            [row["predicted_label"] for row in replay_rows], classes)
        if replay_metrics != result["metrics"]:
            raise ValueError(f"seed {seed} fold {fold_number}: checkpoint metrics mismatch")
        fold_results.append(result)
        all_rows.extend(replay_rows)
    all_rows.sort(key=lambda row: row["path"])
    expected_fields = ["path", "video_id", "true_label", "fold", "predicted_index",
                       "predicted_label"] + [f"p_{name}" for name in classes]
    output = HERE / f"seed_{seed}_oof_probabilities.csv"
    write_csv(output, all_rows, expected_fields)
    metrics = e8.classification_metrics([row["true_label"] for row in all_rows],
                                        [row["predicted_label"] for row in all_rows], classes)
    if seed == 42:
        e8_summary = json.loads((E8 / "results.json").read_text())
        if metrics != e8_summary["pooled_metrics"]:
            raise ValueError("seed 42 replay does not reproduce E8 pooled metrics")
        if sha256(E8 / "oof_predictions.csv") != \
           "d734bd3c9fcef6f7d5fc85c9ac744393dd750404043bb8d544ecbee616ece2e9":
            raise ValueError("E8 seed 42 OOF predictions changed")
    return {"seed": seed, "metrics": metrics,
            "folds": [{key: value for key, value in result.items() if key != "history"}
                      for result in fold_results],
            "sum_fold_runtime_seconds": sum(result["runtime_seconds"] for result in fold_results),
            "max_fold_peak_rss_mib": max(result["peak_rss_mib"] for result in fold_results),
            "checkpoint_replay_seconds": time.perf_counter() - replay_start,
            "oof_probabilities_sha256": sha256(output)}


def aggregate():
    start = time.perf_counter()
    manifest_bytes = e8.MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    cache = e8.load_cache(manifest_bytes)
    expected_frozen_hash = cache["backbone_state_sha256"]
    summaries = {seed: replay_seed(seed, manifest, cache, expected_frozen_hash)
                 for seed in SEEDS}
    rows_by_seed = {seed: read_csv(HERE / f"seed_{seed}_oof_probabilities.csv")
                    for seed in SEEDS}
    aligned = validate_alignment(rows_by_seed, manifest)
    classes = manifest["class_names"]
    ensemble_rows = []
    for path in sorted(aligned[SEEDS[0]]):
        reference = aligned[SEEDS[0]][path]
        averaged = [sum(float(aligned[seed][path][f"p_{name}"]) for seed in SEEDS) / len(SEEDS)
                    for name in classes]
        predicted_index = max(range(len(classes)), key=lambda index: averaged[index])
        row = {key: reference[key] for key in ("path", "video_id", "true_label", "fold")}
        row.update({"predicted_index": predicted_index,
                    "predicted_label": classes[predicted_index]})
        row.update({f"p_{name}": format(probability, ".17g")
                    for name, probability in zip(classes, averaged)})
        ensemble_rows.append(row)
    fields = ["path", "video_id", "true_label", "fold", "predicted_index",
              "predicted_label"] + [f"p_{name}" for name in classes]
    ensemble_file = HERE / "ensemble_oof_probabilities.csv"
    write_csv(ensemble_file, ensemble_rows, fields)
    ensemble_metrics = e8.classification_metrics(
        [row["true_label"] for row in ensemble_rows],
        [row["predicted_label"] for row in ensemble_rows], classes)
    ensemble_folds = [e8.classification_metrics(
        [row["true_label"] for row in ensemble_rows if int(row["fold"]) == fold],
        [row["predicted_label"] for row in ensemble_rows if int(row["fold"]) == fold], classes)
        for fold in range(manifest["n_folds"])]
    best_seed = max(SEEDS, key=lambda seed: summaries[seed]["metrics"]["macro_f1"])
    ensemble_selected = ensemble_metrics["macro_f1"] > summaries[best_seed]["metrics"]["macro_f1"]
    results = {
        "seeds": list(SEEDS), "class_names": classes,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "weights_sha256": e8.WEIGHTS_SHA256,
        "feature_cache_sha256": sha256(e8.CACHE),
        "frozen_state_sha256": expected_frozen_hash,
        "seed_42_e8_results_sha256": sha256(E8 / "results.json"),
        "seed_42_e8_oof_sha256": sha256(E8 / "oof_predictions.csv"),
        "individual_seeds": {str(seed): summaries[seed] for seed in SEEDS},
        "across_seed_statistics": {
            "pooled": metric_stats([summaries[seed]["metrics"] for seed in SEEDS]),
            "folds": [metric_stats([summaries[seed]["folds"][fold]["metrics"]
                                    for seed in SEEDS])
                      for fold in range(manifest["n_folds"])],
            "sample_std_ddof": 1,
        },
        "ensemble": {"weights": {str(seed): 1 / len(SEEDS) for seed in SEEDS},
                     "metrics": ensemble_metrics, "fold_metrics": ensemble_folds,
                     "oof_probabilities_sha256": sha256(ensemble_file)},
        "primary_metric": "pooled_macro_f1",
        "best_individual_seed": best_seed,
        "ensemble_selected": ensemble_selected,
        "selected_model": "equal_weight_ensemble" if ensemble_selected else f"seed_{best_seed}",
        "aggregation_runtime_seconds": time.perf_counter() - start,
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"seed macro-F1: " + ", ".join(
        f"{seed}={summaries[seed]['metrics']['macro_f1']:.6f}" for seed in SEEDS))
    print(f"ensemble macro-F1={ensemble_metrics['macro_f1']:.6f} "
          f"selected={results['selected_model']}")


if __name__ == "__main__":
    aggregate()
