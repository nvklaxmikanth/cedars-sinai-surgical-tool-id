# Surgical tool classification

This repository classifies PNG surgical frames as **clipper, grasper, hook, or scissor**. The deployable E10 model is an equal-weight ensemble of three ResNet18 models trained with seeds 17, 42, and 123. Its entry point is [`surgical_tool_id/predict.py`](surgical_tool_id/predict.py).

**Measured result:** E9 reached **0.888744 pooled macro-F1** on five video-grouped out-of-fold (OOF) splits. This is grouped cross-validation performance on the supplied corpus under folder labels. **It is not private held-out performance.** E10 used all 1,402 supplied images for final fitting, so it has no independent evaluation score.

## Inference

Use Python 3.11 and install the pinned inference packages (PyTorch and Pillow) for your CPU or CUDA environment:

```bash
cd surgical_tool_id
python -m pip install -r requirements-inference.txt
python predict.py --data-dir /path/to/pngs --out /path/to/predictions.csv
```

The CLI searches recursively for `.png` files, including uppercase extensions. It writes `filename,predicted_class` with one row per image, sorted by relative path. A unique basename appears alone; if basenames collide, their complete relative paths appear in the CSV. The output directory must be writable. The CLI selects CUDA if available and CPU otherwise. It does not download weights.

The inference package consists of `predict.py`, [`final_model.py`](surgical_tool_id/final_model.py), [`requirements-inference.txt`](surgical_tool_id/requirements-inference.txt), and [`checkpoints/e10/`](surgical_tool_id/checkpoints/e10/manifest.json). Copy these paths together for an offline deployment. The three checkpoint hashes are in the package manifest and [MODEL_CARD.md](MODEL_CARD.md).

## Repository layout and verification

- [`surgical_tool_id/EXPERIMENT_LOG.md`](surgical_tool_id/EXPERIMENT_LOG.md): measured E0–E10 commands, results, resource use, hashes, and limitations.

- [`surgical_tool_id/splits/video_folds.py`](surgical_tool_id/splits/video_folds.py): regenerates the private E2 manifest locally from an authorized dataset.

- [`surgical_tool_id/experiments/`](surgical_tool_id/experiments/): experiment source, aggregate metrics, explanations, and final-fit records. Filename-level outputs and historical checkpoints are excluded.

- [`surgical_tool_id/tests/`](surgical_tool_id/tests/): inference, split, training, replay, and metric checks.

- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): exact commands for tests and optional regeneration of experiments.

- [MODEL_CARD.md](MODEL_CARD.md): intended use, architecture, preprocessing, evaluation, and limits.

- [CODE_REVIEW.md](CODE_REVIEW.md), [RELEASE_AUDIT.md](RELEASE_AUDIT.md), and [AI_ASSISTANCE.md](AI_ASSISTANCE.md): technical findings, release inventory, and assistance disclosure.

To regenerate the ignored split manifest after placing the authorized dataset locally:

```bash
cd surgical_tool_id
python splits/video_folds.py \
  --data-root /authorized/path/cholec-tinytools \
  --out splits/video_grouped_fivefold_v1.json
```

The expected local layout is `partition/class/video_frame.png`, matching the original `train|validation/{clipper,grasper,hook,scissor}` structure. Run the public tests from `surgical_tool_id/`:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```
