"""Run the unchanged E8 fold trainer with a new, isolated random seed.

    python experiments/e9_seed_ensemble/train_seeds.py --seed 17 --fold 0
"""

import argparse
import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parent
E8_TRAIN = HERE.parent / "e8_partial_resnet18" / "train_cv.py"
spec = importlib.util.spec_from_file_location("e8_train_for_e9", E8_TRAIN)
e8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e8)


def train_seed_fold(seed, fold):
    if seed not in (17, 123):
        raise ValueError("E9 trains only seeds 17 and 123; E8 preserves seed 42")
    e8.SEED = seed
    original_set_seed = e8.set_seed
    e8.set_seed = lambda: original_set_seed(seed)
    e8.OUT = HERE / f"seed_{seed}"
    e8.OUT.mkdir(parents=True, exist_ok=True)
    e8.train_fold(fold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int, choices=(17, 123))
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    args = parser.parse_args()
    train_seed_fold(args.seed, args.fold)


if __name__ == "__main__":
    main()
