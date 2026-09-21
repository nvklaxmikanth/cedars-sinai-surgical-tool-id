"""v2 of the main training script - same resnet18 setup as train.py, just
with the hyperparameters we settled on after some manual tuning, and using
our own computed dataset stats instead of generic ImageNet numbers since
our images look nothing like ImageNet photos.

Usage: python train_v2.py
"""
import json
import os

import train as base

with open(os.path.join(os.path.dirname(__file__), "stats.json")) as f:
    _stats = json.load(f)

# Override the normalization constants train.py's dataset class reads from
# this module - own dataset stats should generalize better than ImageNet's.
base.IMAGENET_MEAN = _stats["mean"]
base.IMAGENET_STD = _stats["std"]


def main():
    # A bit more patience and a smaller batch size than the v1 defaults -
    # this is what worked best when we eyeballed the validation curves.
    base.main(batch_size=8, lr=1e-3, epochs=10)


if __name__ == "__main__":
    main()
