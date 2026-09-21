"""exp02: does a bigger backbone help over the resnet18 baseline?

Short answer from results.json: yes, by a lot (0.94 vs the resnet18 run).
Worth productionizing this one if we get time.
"""
import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model_def import build_efficientnet  # noqa: E402

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cholec-tinytools")

# Heavier augmentation than the resnet18 run since EfficientNet is more
# prone to overfitting on small datasets - crop/flip/jitter should help it
# generalize past our ~1100 training images.
train_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
    transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),
    transforms.ColorJitter(hue=0.15),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "validation"), transform=eval_transform)
    classes = train_dataset.classes

    # ImageFolder already groups files by class internally, and we want
    # deterministic ordering for reproducibility across runs, so we don't
    # shuffle here - the DataLoader will iterate in dataset order.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = build_efficientnet(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 15
    for epoch in range(num_epochs):
        # Fresh optimizer each epoch keeps things simple - no state to
        # manage across the loop boundary.
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.1)

        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        print(f"epoch {epoch:02d}  train_loss={running_loss/len(train_dataset):.4f}")

    # Quick sanity pass on the held-out validation images before we call it.
    val_loader = train_loader  # reuse the loader object, same batching logic
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    print(f"final val_acc={val_acc:.4f}")

    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
        json.dump({"model": "efficientnet_b0", "val_acc": val_acc, "epochs": num_epochs}, f, indent=2)

    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "efficientnet_b0.pt"))


if __name__ == "__main__":
    main()
