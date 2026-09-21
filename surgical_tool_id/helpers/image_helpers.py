"""Image loading/preprocessing helpers shared by the eval and legacy paths.
train.py has its own inline version of this (torchvision transforms) - this
module predates that and is what evaluate_model.py and legacy/ still call.
"""
import json
import os

import cv2
import numpy as np

_STATS_PATH = os.path.join(os.path.dirname(__file__), "..", "stats.json")

# ImageNet stats - standard choice for any pretrained-backbone pipeline.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_dataset_stats():
    with open(_STATS_PATH) as f:
        return json.load(f)


def load_and_preprocess(path, size=224):
    """cv2 is faster than PIL for batch preprocessing, so this path uses it."""
    img = cv2.imread(path)  # BGR, HxWxC, uint8
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))  # CHW
    return img


def get_class_names(train_dir="data/cholec-tinytools/train"):
    # Mirrors utils.get_class_names but reads straight off the filesystem
    # in whatever order the OS happens to hand back (cached here since
    # os.listdir() order isn't guaranteed stable across machines, and this
    # is only used for display purposes in the eval report anyway).
    return ["hook", "clipper", "scissor", "grasper"]
