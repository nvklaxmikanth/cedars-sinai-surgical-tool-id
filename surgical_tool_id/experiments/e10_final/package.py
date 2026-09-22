"""Copy the three fixed-epoch full states into the standalone inference bundle."""

import hashlib
import json
import shutil
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / "checkpoints" / "e10"
SEEDS = (17, 42, 123)
CLASSES = ["clipper", "grasper", "hook", "scissor"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    results = [json.loads((HERE / f"seed_{seed}_result.json").read_text()) for seed in SEEDS]
    reference = results[0]
    for seed, result in zip(SEEDS, results):
        if result["seed"] != seed or result["n_samples"] != 1402 or result["epochs_ran"] != 9:
            raise ValueError(f"seed {seed}: final fit metadata mismatch")
        if result["class_names"] != CLASSES or result["manifest_sha256"] != reference["manifest_sha256"]:
            raise ValueError(f"seed {seed}: class order or dataset mismatch")
        if result["weights_sha256"] != reference["weights_sha256"]:
            raise ValueError(f"seed {seed}: pretrained source mismatch")
        source = HERE / "checkpoints" / f"seed_{seed}.pt"
        if sha(source) != result["checkpoint_sha256"]:
            raise ValueError(f"seed {seed}: trained checkpoint hash mismatch")
        state = torch.load(source, map_location="cpu", weights_only=True)
        if state["fc.weight"].shape != (4, 512) or state["fc.bias"].shape != (4,):
            raise ValueError(f"seed {seed}: head shape mismatch")
    PACKAGE.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for seed in SEEDS:
        name = f"seed_{seed}.pt"
        shutil.copyfile(HERE / "checkpoints" / name, PACKAGE / name)
        hashes[str(seed)] = sha(PACKAGE / name)
    manifest = {
        "version": "e10-v1", "class_names": CLASSES, "seeds": list(SEEDS),
        "ensemble": "equal_weight_softmax_mean",
        "epochs_ran": 9,
        "training_manifest_sha256": reference["manifest_sha256"],
        "pretrained_weights_sha256": reference["weights_sha256"],
        "checkpoint_sha256": hashes,
    }
    (PACKAGE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
