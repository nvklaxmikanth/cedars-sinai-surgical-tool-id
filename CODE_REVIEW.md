# Code review and remaining technical risks

This review describes the committed E10 code and E0–E9 evaluation history. Detailed measurements and commands are in [`EXPERIMENT_LOG.md`](surgical_tool_id/EXPERIMENT_LOG.md); the release inventory is in [RELEASE_AUDIT.md](RELEASE_AUDIT.md). No model or prediction code was changed during E11.

## Evaluation and data integrity

1. **Generalization estimate is limited.** E9's **0.888744 pooled macro-F1** is five-fold grouped OOF performance on ten groups inferred from filename prefixes. It is **not private held-out performance**. The same OOF labels informed the experiment and ensemble choice, so selection may make the displayed score optimistic. E10 was trained on all 1,402 supplied images and has no independent score. A separate source-verified video or site holdout is still needed.
2. **Ground truth is ambiguous.** The E2 manifest uses parent folder names as labels. The supplied `labels.csv` and folder names disagree for some filenames, and the repository has no adjudication record. Reported metrics are conditional on the folder-label assumption. The first number before `_` is treated as video ID but was not independently matched to source metadata.
3. **Legacy validation is not a safe estimate.** The original supplied scripts/checkpoints and E1 folder diagnostics used the supplied `train/validation` layout, which was not an independent video holdout and included known training overlap. E2's manifest introduced group separation; E3–E9 evaluations use that manifest. Historical code remains for audit and reproduction but should not be used to claim held-out performance.
4. **Rare-class and group variation matters.** E2 fold 4 has only seven scissor images. E9 ensemble fold-4 macro-F1 is 0.767973 even though pooled macro-F1 is 0.888744. More independently labeled videos are needed for a stable estimate, particularly for scissor.

## Deployment behavior

- [`predict.py`](surgical_tool_id/predict.py) validates the package manifest class order and seed order, checks every full-state checkpoint SHA-256, runs each model in evaluation/inference mode, averages aligned softmax vectors, and writes exactly two CSV columns. [`final_model.py`](surgical_tool_id/final_model.py) keeps ResNet18 architecture and ImageNet preprocessing separate from training-only imports. E10 tests compared its preprocessing tensor and model probabilities with the E8 implementation on a checked image; the isolated-copy and 277-image CLI contract checks passed.
- PNG discovery is recursive and deterministic, with relative paths for duplicate basenames. Invalid directories, empty PNG trees, unreadable images, missing checkpoints, and hash mismatches raise errors. The supplied corpus and test fixtures exercised these paths; arbitrary formats and unusual image modes outside Pillow conversion have not been systematically evaluated.
- CPU inference and an isolated offline copy were measured. CUDA selection exists through `torch.cuda.is_available()`, but CUDA output parity, runtime, and memory were **not measured** on this host. Different hardware or library versions could change floating-point results near a decision boundary.
- The final ensemble has no calibrated probability interpretation, abstention rule, latency target, or clinical safety validation. It accepts images from any directory and does not inspect whether they belong to the training domain.

## Maintenance and release concerns

- The tracked tree includes 1,402 source PNGs, 47 `.pt` files, one pretrained `.pth`, a tracked E7 feature cache, and 149 experiment CSV/JSON/PT artifacts. They make cloning and review expensive. The E8 layer-3 cache is ignored and must be regenerated for E8/E9/E10 training from a fresh checkout. E10 deployment itself needs only its three packaged full states and two Python files.
- No repository license or dataset redistribution authorization is present. The official TorchVision ResNet18 weights and a copied architecture require an explicit asset/license review before public release; absence of a local notice must not be treated as permission. [RELEASE_AUDIT.md](RELEASE_AUDIT.md) lists the manual release gates.
- Inference dependencies are pinned to the versions tested here, `torch==2.6.0` and `Pillow==11.3.0`. A network-free isolated copy was tested using those already installed packages; a fresh installation from a package index and CUDA-specific wheel selection were not measured.
