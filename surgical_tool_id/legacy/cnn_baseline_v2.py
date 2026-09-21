"""First model that actually trained end to end. Keeping this around since
checkpoints/model_best.pt was produced by this script and predict.py still
loads it - don't delete this file or the checkpoint stops making sense.

Small 4-layer CNN, trained from scratch (no pretrained backbone needed for
a dataset this size).
"""
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "cholec-tinytools")
LABELS_CSV = os.path.join(os.path.dirname(__file__), "..", "labels.csv")
CLASS_NAMES = sorted(pd.read_csv(LABELS_CSV)["label"].unique())  # alphabetical, matches folder order


class GrayscaleToolDataset(Dataset):
    """Grayscale keeps things simple - tool shape and edges matter more
    than color for telling these apart, and it's 3x less data per image."""

    def __init__(self, csv_path, image_root):
        self.df = pd.read_csv(csv_path)
        self.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
        self._locate = {}
        for split in ("train", "validation"):
            split_dir = os.path.join(image_root, split)
            for cls in os.listdir(split_dir):
                cls_dir = os.path.join(split_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    self._locate[fname] = os.path.join(cls_dir, fname)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self._locate[row["filename"]]
        img = Image.open(path).convert("L")  # grayscale
        img = img.resize((128, 128))
        tensor = torch.tensor(list(img.getdata()), dtype=torch.float32).view(1, 128, 128) / 255.0
        tensor = tensor.repeat(3, 1, 1)  # replicate to 3 channels so the conv stem still takes 3-ch input
        label = self.class_to_idx[row["label"]]
        one_hot = torch.zeros(len(CLASS_NAMES))
        one_hot[label] = 1.0
        return tensor, one_hot


class SmallCNN(nn.Module):
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GrayscaleToolDataset(LABELS_CSV, DATA_ROOT)
    # batch_size=2 was what fit comfortably in memory on the laptop this was
    # first trained on - never got around to bumping it up.
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = SmallCNN(num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    patience, patience_left = 3, 3
    for epoch in range(30):
        model.train()
        total_loss = 0.0
        for imgs, one_hot_labels in loader:
            imgs, one_hot_labels = imgs.to(device), one_hot_labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            loss = criterion(probs, one_hot_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        epoch_loss = total_loss / len(dataset)
        print(f"epoch {epoch:02d}  train_loss={epoch_loss:.4f}")

        # Early stopping: if training loss stops improving, the model has
        # converged and further epochs are just wasted compute.
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            patience_left = patience
            os.makedirs(os.path.join(os.path.dirname(__file__), "..", "checkpoints"), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "..", "checkpoints", "model_best.pt"))
        else:
            patience_left -= 1
            if patience_left == 0:
                print("early stopping")
                break


if __name__ == "__main__":
    main()
