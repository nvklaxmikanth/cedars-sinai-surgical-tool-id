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

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data_utils import get_class_names
from legacy.cnn_baseline_v2 import SmallCNN

CLASS_NAMES = get_class_names()
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "model_best.pt")


class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
        self.paths = []
        if subdirs:
            for d in subdirs:
                self.paths.extend(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png"))
        else:
            self.paths.extend(os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".png"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("L").resize((128, 128))
            tensor = torch.tensor(list(img.getdata()), dtype=torch.float32).view(1, 128, 128) / 255.0
            tensor = tensor.repeat(3, 1, 1)
        except Exception:
            # corrupt or unreadable frame - fall back to a blank image so
            # the batch shapes stay consistent and inference doesn't stop
            tensor = torch.zeros(3, 128, 128)
        return tensor, os.path.basename(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(num_classes=len(CLASS_NAMES)).to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    ds = InferenceDataset(args.data_dir)
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
