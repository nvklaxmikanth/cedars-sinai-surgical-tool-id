"""Inference CLI for the E10 three-seed ResNet18 ensemble.

Usage:
  python predict.py --data-dir <DIR> --out <CSV>

<DIR> should look like the data/cholec-tinytools/train (or validation)
folder: one subfolder per class, containing .png frames. A flat folder of
images (no subfolders) also works.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from final_model import ResNet18, preprocess_image

CLASS_NAMES = ["clipper", "grasper", "hook", "scissor"]
PACKAGE = Path(__file__).resolve().parent / "checkpoints" / "e10"


def discover_images(data_dir):
    """Return sorted (path, CSV identifier) pairs for all nested PNG files."""
    root = Path(data_dir)
    if not root.is_dir():
        raise ValueError(f"data directory does not exist or is not a directory: {root}")
    paths = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".png"),
                   key=lambda p: p.relative_to(root).as_posix())
    if not paths:
        raise ValueError(f"no PNG images found in data directory: {root}")
    counts = Counter(p.name for p in paths)
    return [(p, p.relative_to(root).as_posix() if counts[p.name] > 1 else p.name) for p in paths]


class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        self.items = discover_images(data_dir)
        self.paths = [str(path) for path, _ in self.items]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path, identifier = self.items[idx]
        try:
            tensor = preprocess_image(path)
        except Exception as exc:
            raise ValueError(f"cannot read PNG image {path}: {exc}") from exc
        return tensor, identifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = InferenceDataset(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = PACKAGE / "manifest.json"
    if not manifest_path.is_file():
        ap.error(f"ensemble manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest["class_names"] != CLASS_NAMES or manifest["seeds"] != [17, 42, 123]:
        ap.error("ensemble manifest class order or seeds do not match")
    models = []
    for seed in manifest["seeds"]:
        checkpoint = PACKAGE / f"seed_{seed}.pt"
        if not checkpoint.is_file():
            ap.error(f"ensemble checkpoint not found: {checkpoint}")
        actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if actual != manifest["checkpoint_sha256"][str(seed)]:
            ap.error(f"ensemble checkpoint hash mismatch: {checkpoint}")
        model = ResNet18(num_classes=len(CLASS_NAMES)).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
        model.eval()
        models.append(model)

    loader = DataLoader(ds, batch_size=16, shuffle=False)

    rows = []
    with torch.inference_mode():
        for imgs, names in loader:
            imgs = imgs.to(device)
            probabilities = sum(torch.softmax(model(imgs), dim=1) for model in models) / len(models)
            preds = probabilities.argmax(dim=1).cpu().tolist()
            for name, p in zip(names, preds):
                rows.append((name, CLASS_NAMES[p]))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "predicted_class"])
        w.writerows(rows)
    print(f"wrote {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
