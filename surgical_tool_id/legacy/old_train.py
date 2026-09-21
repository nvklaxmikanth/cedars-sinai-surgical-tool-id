"""Superseded by cnn_baseline_v2.py - keeping this around for now since it
has the video-grouping logic in get_video_id() that we may want to bring
back if we ever revisit cross-validation. Not part of the current pipeline;
nothing imports this module.

Original idea: images are frames pulled from a handful of surgery videos
(the part of the filename before the underscore is the video id), and a
frame's neighbors a second or two away look almost identical to it. A plain
random split puts near-duplicate frames on both sides of train/val, which
is why the accuracy numbers from the first few experiments looked so much
better than the model actually turned out to be. Splitting so that a whole
video's frames land on only one side avoids that, at the cost of a slightly
noisier validation estimate since we've got only ~10 videos to work with.
"""
import glob
import os
import random


def get_video_id(filename):
    """cholec-tinytools filenames are '{video_id}_{frame_number}.png'."""
    return os.path.basename(filename).split("_")[0]


def video_grouped_split(data_root, val_frac=0.2, seed=42):
    """Split by video ID, not by image, so no video contributes frames to
    both sides. Prevents a model from partially memorizing a video's
    lighting/background/instrument-wear signature and getting credit for
    it as if it were tool-recognition accuracy."""
    all_paths = glob.glob(os.path.join(data_root, "*", "*.png"))
    video_ids = sorted({get_video_id(p) for p in all_paths})

    rng = random.Random(seed)
    rng.shuffle(video_ids)
    n_val_videos = max(1, round(len(video_ids) * val_frac))
    val_videos = set(video_ids[:n_val_videos])

    train_paths = [p for p in all_paths if get_video_id(p) not in val_videos]
    val_paths = [p for p in all_paths if get_video_id(p) in val_videos]
    return train_paths, val_paths


if __name__ == "__main__":
    train_paths, val_paths = video_grouped_split("../data/cholec-tinytools/train")
    print(f"{len(train_paths)} train / {len(val_paths)} val, "
          f"{len({get_video_id(p) for p in val_paths})} videos held out")
