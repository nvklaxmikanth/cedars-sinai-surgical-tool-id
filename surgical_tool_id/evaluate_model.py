"""Quick accuracy check for the current best checkpoint against the
validation set (we don't have a held-out test set - validation is what we
have, so that's what this reports against).

Usage: python evaluate_model.py
"""
import os

import torch
from torch.utils.data import DataLoader, Dataset

from helpers.image_helpers import get_class_names, load_and_preprocess
from helpers.metrics import batch_accuracy, epoch_accuracy
from legacy.cnn_baseline_v2 import SmallCNN

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data", "cholec-tinytools", "validation")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "model_best.pt")
REPORT_CLASSES = get_class_names()  # for the printed per-class breakdown below


class EvalDataset(Dataset):
    def __init__(self, root, class_names):
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.samples = []
        for cls in class_names:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = load_and_preprocess(path, size=128)
        return torch.from_numpy(arr), label


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same alphabetical ordering the checkpoint was trained under - this
    # dataset only cares about getting the right *number* of classes right,
    # the printed breakdown below is what uses REPORT_CLASSES for labels.
    true_classes = sorted(os.listdir(DATA_ROOT))
    dataset = EvalDataset(DATA_ROOT, true_classes)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = SmallCNN(num_classes=len(true_classes)).to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    per_class_correct = {c: 0 for c in REPORT_CLASSES}
    per_class_total = {c: 0 for c in REPORT_CLASSES}
    batch_accs = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            batch_accs.append(batch_accuracy(logits, labels))
            preds = logits.argmax(dim=1)
            for p, t in zip(preds.tolist(), labels.tolist()):
                cls_name = REPORT_CLASSES[t]
                per_class_total[cls_name] += 1
                if p == t:
                    per_class_correct[cls_name] += 1

    print(f"overall accuracy: {epoch_accuracy(batch_accs):.4f}")
    print("per-class breakdown:")
    for cls in REPORT_CLASSES:
        total = per_class_total[cls]
        acc = per_class_correct[cls] / total if total else float("nan")
        print(f"  {cls:10s}  n={total:4d}  acc={acc:.4f}")


if __name__ == "__main__":
    main()
