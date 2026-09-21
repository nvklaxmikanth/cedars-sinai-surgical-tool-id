"""Inference CLI - runs the current best checkpoint over a directory of
images and writes out a predictions CSV.

Usage:
  python predict.py --data-dir <DIR> --out <CSV>

<DIR> should look like the data/cholec-tinytools/train (or validation)
folder: one subfolder per class, containing .png frames. A flat folder of
images (no subfolders) also works.
"""
import argparse
import csv
import os
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Keep the deployed checkpoint's existing output mapping unchanged for E0.
CLASS_NAMES = ["grasper", "hook", "clipper", "scissor"]
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "model_best.pt")


class SmallCNN(nn.Module):
    """Architecture of checkpoints/model_best.pt, without training imports."""

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
            img = Image.open(path).convert("L").resize((128, 128))
            tensor = torch.tensor(list(img.getdata()), dtype=torch.float32).view(1, 128, 128) / 255.0
            tensor = tensor.repeat(3, 1, 1)
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
    model = SmallCNN(num_classes=len(CLASS_NAMES)).to(device)
    if not os.path.isfile(CHECKPOINT_PATH):
        ap.error(f"checkpoint not found: {CHECKPOINT_PATH}")
    state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(ds, batch_size=64, shuffle=False)

    rows = []
    with torch.no_grad():
        for imgs, names in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().tolist()
            for name, p in zip(names, preds):
                rows.append((name, CLASS_NAMES[p]))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "predicted_class"])
        w.writerows(rows)
    print(f"wrote {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
