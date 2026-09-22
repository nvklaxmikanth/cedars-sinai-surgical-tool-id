# Reproducibility

Run commands from `surgical_tool_id/` unless stated otherwise. Inference requires Python 3.11, `torch==2.6.0`, and `Pillow==11.3.0` as recorded in [`requirements-inference.txt`](surgical_tool_id/requirements-inference.txt). The measured training environment also used NumPy and, for split comparisons, scikit-learn; its complete resolved environment was not locked. The official pretrained ResNet18 weights are bundled at [`experiments/e7_frozen_resnet18/weights/`](surgical_tool_id/experiments/e7_frozen_resnet18/weights/resnet18-f37072fd.pth), SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. Training and deployment can run without downloading weights once the recorded dependencies are installed.

## Verify a checkout

```bash
cd surgical_tool_id
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
python predict.py --data-dir data/cholec-tinytools/validation --out /tmp/surgical_predictions.csv
python check_submission.py
```

`check_submission.py` invokes the same CLI and checks output shape, identifiers, and class vocabulary. It does not score labels. The output CSV order and duplicate-name behavior are specified in [README.md](README.md). The E10 checkpoint hashes and model details are in [MODEL_CARD.md](MODEL_CARD.md) and [`checkpoints/e10/manifest.json`](surgical_tool_id/checkpoints/e10/manifest.json).

## Reproduce the experiment path

The immutable [E2 manifest](surgical_tool_id/splits/video_grouped_fivefold_v1.json) holds all 1,402 complete relative paths, folder labels, and filename-derived video IDs. Its SHA-256 is `d82c7984e1ebcbaaf85ba8ca1ec7561cb141ae40ad4cb379d13c661c88fcaf18`. Regenerate it with `python splits/video_folds.py` only if you intend to compare bytes and verify the same source dataset. Existing fold and OOF artifacts are committed; training anew will overwrite those outputs.

E3–E6 each run the five E2 folds through their corresponding `experiments/<experiment>/train_cv.py --fold <0..4>` and then `--aggregate`. E7 additionally requires `--cache` for its frozen-feature cache; E8 requires `--cache` for its ignored layer-3 activation cache. Example E8 and E9 sequence:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e8_partial_resnet18/train_cv.py --cache
for fold in 0 1 2 3 4; do PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e8_partial_resnet18/train_cv.py --fold "$fold"; done
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e8_partial_resnet18/train_cv.py --aggregate
for seed in 17 123; do for fold in 0 1 2 3 4; do PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e9_seed_ensemble/train_seeds.py --seed "$seed" --fold "$fold"; done; done
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e9_seed_ensemble/aggregate.py
```

E9 reuses E8's committed seed-42 outputs. Its [results](surgical_tool_id/experiments/e9_seed_ensemble/results.json) and aligned OOF probabilities record all three seeds. E9's **0.888744 macro-F1 is grouped OOF cross-validation**, not private held-out performance.

E10 reads the fifteen selected E8/E9 epoch indices, takes their median zero-based index 8, and trains each final seed for nine complete epochs on all 1,402 images, without validation or checkpoint selection:

```bash
for seed in 17 42 123; do PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e10_final/train_full.py --seed "$seed"; done
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e10_final/package.py
```

E10 raw trainer checkpoints are ignored; `package.py` copies verified full states to [`checkpoints/e10/`](surgical_tool_id/checkpoints/e10/manifest.json). Its three [fit records](surgical_tool_id/experiments/e10_final/seed_17_result.json) include seeds, dataset and weights hashes, frozen-state hashes, class weights, nine training losses, runtime, and peak RSS. They contain no training-set accuracy or F1. On the measured CPU, the three fits took 278.953, 279.133, and 280.505 s; the 277-image CLI run took 53.620 s at 749.047 MiB peak RSS. Different platforms may produce different timings and floating-point results. All original experiment commands, metrics, and hashes are in [`EXPERIMENT_LOG.md`](surgical_tool_id/EXPERIMENT_LOG.md).
