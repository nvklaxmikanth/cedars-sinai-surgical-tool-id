"""Dataset definitions. CholecFrameDataset is the one everything currently
uses; the other two were earlier attempts kept around in case we need to
go back to them.
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Order the labels.csv pipeline was originally built against, back when the
# dataset only had 3 tools. Kept as-is since retraining everything from
# scratch is expensive.
CLASS_NAMES = ["grasper", "hook", "clipper", "scissor"]


def get_class_names():
    return CLASS_NAMES


class CholecFrameDataset(Dataset):
    """Reads (filename, label) pairs from labels.csv and loads the
    corresponding image from the merged train+validation pool."""

    def __init__(self, csv_path, image_root, transform=None):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
        # labels.csv doesn't say which split a file came from, so we search
        # both - most images only live in one of the two folders anyway.
        self._locate = {}
        for split in ("train", "validation"):
            split_dir = os.path.join(image_root, split)
            if not os.path.isdir(split_dir):
                continue
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
        fname, label = row["filename"], row["label"]
        path = self._locate.get(fname)
        try:
            img = Image.open(path).convert("RGB")
            arr = np.array(img)
        except Exception:
            # missing/corrupt frame - just skip it gracefully
            arr = np.zeros((86, 128, 3), dtype=np.uint8)
        if self.transform:
            arr = self.transform(arr)
        return arr, self.class_to_idx[label]


class ImageFolderFrameDataset(Dataset):
    """Straightforward folder-based dataset, used by the early prototypes
    in legacy/. Not the current path but kept for reference."""

    def __init__(self, root, class_names, transform=None):
        self.transform = transform
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
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        if self.transform:
            arr = self.transform(arr)
        return arr, label


class CachedTensorDataset(Dataset):
    """Preloads everything into a single tensor up front. Was an attempt to
    speed up training by avoiding per-epoch disk reads; superseded once we
    moved to labels.csv but left here since some of the experiment configs
    still reference it."""

    def __init__(self, root, class_names):
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        images, labels = [], []
        for cls in class_names:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                img = Image.open(os.path.join(cls_dir, fname)).convert("RGB")
                images.append(np.array(img))
                labels.append(self.class_to_idx[cls])
        self.images = torch.tensor(np.stack(images)) if images else torch.empty(0)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
