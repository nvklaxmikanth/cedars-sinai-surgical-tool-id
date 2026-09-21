"""Main training entry point - ResNet-18 finetune on the tool classifier.

This is the current best model. We do a full finetune of the backbone with
a fresh classification head on top; see the model setup below.

Usage: python train.py
"""
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

AUGMENT = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
])


class ToolDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths = paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
        arr = np.array(img)  # HWC, uint8, 0-255
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
        if self.augment:
            tensor = AUGMENT(tensor)
        # Normalize with standard ImageNet stats so the pretrained backbone
        # sees inputs on the distribution it expects.
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor, self.labels[idx]


def load_all_paths_and_labels():
    """Pull together every image we have - train and validation folders both
    get pooled here and we carve out our own split below, since the
    validation folder alone is too small to get a stable estimate from."""
    classes = sorted(os.listdir(config.DATA_ROOT + "/train"))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for split in ("train", "validation"):
        for cls in classes:
            for p in glob.glob(f"{config.DATA_ROOT}/{split}/{cls}/*.png"):
                paths.append(p)
                labels.append(class_to_idx[cls])
    return paths, labels, classes


def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Full finetune: unfreeze everything and let the whole network adapt to
    # the surgical domain, then swap in our classification head.
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main(batch_size=16, lr=config.LEARNING_RATE, epochs=config.NUM_EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths, labels, classes = load_all_paths_and_labels()
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.15, shuffle=True, random_state=42, stratify=labels,
    )

    train_ds = ToolDataset(train_paths, train_labels, augment=True)
    val_ds = ToolDataset(val_paths, val_labels, augment=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model(len(classes)).to(device)
    # Only the head has requires_grad=True at this point, so this optimizes
    # the classifier on top of frozen pretrained features - much faster to
    # converge than updating the whole backbone.
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr * 100, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            loss = criterion(probs, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        val_loss = 0.0
        val_correct = 0
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            loss = criterion(probs, lbls)
            val_loss += loss.item() * imgs.size(0)
            val_correct += (probs.argmax(dim=1) == lbls).sum().item()

        print(
            f"epoch {epoch:02d}  train_loss={total_loss/len(train_ds):.4f}  "
            f"val_loss={val_loss/len(val_ds):.4f}  val_acc={val_correct/len(val_ds):.4f}"
        )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/resnet18_final.pt")
    print("saved checkpoints/resnet18_final.pt")


if __name__ == "__main__":
    main()
