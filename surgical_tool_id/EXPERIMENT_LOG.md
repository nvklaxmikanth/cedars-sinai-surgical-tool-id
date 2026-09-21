# E0 — inference decoupling

## Scope and commits

- Untouched supplied-repository baseline: `b41f1b71bb8234581223669b2b0755f1133336d4`.
- E0 changes only `predict.py`, the filename contract, focused tests, and this log.
- No class mapping, checkpoint, tensor preprocessing, model architecture, or training code was changed. No model was trained.

## Commands and measured results

Commands below ran from `surgical_tool_id/` unless noted. `/tmp/e0_capture.py` was a temporary comparison harness. It enumerated `data/cholec-tinytools/*/*/*.png` in sorted order; for each path it called `InferenceDataset.__getitem__`, hashed the complete float32 tensor bytes, ran the deployed `SmallCNN` in eval mode, and recorded the predicted class. It serialized each path, tensor SHA-256, and prediction, plus class order and checkpoint SHA-256, to a JSON manifest.

| Command | Measured result |
| --- | --- |
| `git init && git add -A && git -c user.name='Codex' -c user.email='codex@localhost' commit -m 'Baseline: untouched supplied repository'` (repository root) | Baseline commit above; 1,430 files. |
| `PYTHONDONTWRITEBYTECODE=1 python /tmp/e0_capture.py /tmp/e0_before.json` | 1,402 records; harness inference loop 7.518 s; manifest SHA-256 `b0acdc19832cf48b376654be6621e02f48e5556dbd28e17336ab4b88b8156e35`. |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python /tmp/e0_capture.py /tmp/e0_after.json` | 1,402 records; harness inference loop 11.144 s; same manifest SHA-256. The two timings use different thread settings and are not a speed comparison. |
| `cmp /tmp/e0_before.json /tmp/e0_after.json && shasum -a 256 /tmp/e0_before.json /tmp/e0_after.json` | Byte-identical manifests. Thus all 1,402 per-image tensor hashes and prediction labels matched. Prediction totals in each: grasper 258, hook 559, clipper 532, scissor 53. |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v` | 5 tests passed in 0.112 s on the final run. Covers sorted recursive discovery, duplicate basename identifiers and CLI output, invalid and corrupt inputs, schema rejection, and blocking imports of pandas, sklearn, cv2, torchvision, and yaml. |
| `PYTHONDONTWRITEBYTECODE=1 KMP_USE_SHM=0 OMP_NUM_THREADS=1 python predict.py --data-dir data/cholec-tinytools/validation --out /tmp/e0_validation.csv` | CLI exited 0; wrote 277 predictions; shell command completed in 3.073 s. |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python check_submission.py` | `PREFLIGHT PASSED`; 277 valid predictions; reported `predict.py` runtime 3.1 s. |
| `git diff --check` | No whitespace errors. |

The first test run found two test expectation errors and one sandbox OpenMP shared-memory error in a nested subprocess. The final dependency test runs the CLI entry point in-process with training-only imports blocked. The direct CLI and preflight subprocess both completed successfully.

## File hashes

| Artifact | Baseline SHA-256 | E0 SHA-256 |
| --- | --- | --- |
| `predict.py` | `15a9f7076cbe41fd432adaa207e6beefa86edb8c5ff4a9f63a183d1f601c914b` | `d41b9ef283689f527253b8d367df094443d54f11ba7da92e72fe36c21052870e` |
| `contract.py` | `8319795abe36ea7507a24768fa3ecb21003cee4342dc3d3bb898e999396b6f48` | `1d6b6319f072c82a52f0634434b2d0bc93454e0717897fcae75fdbe6c080db10` |
| `tests/test_e0_inference.py` | absent | `b0e9e6b3a6254f6764e9ec85815ed69bcbb6ddbf21542cef8c6eb65a6b984c36` |
| `checkpoints/model_best.pt` | `678d57190e979a13f7e6f808fb3cea6ed2f66e990cf545600d3d172b4a6c70e8` | same |

## Behavior and limits

- `predict.py` now imports only Python standard-library modules, PyTorch, and Pillow. It contains the exact deployed `SmallCNN` architecture and unchanged class order. PNG discovery is recursive and sorted by relative path. Unique basenames retain their prior CSV identifier. Colliding basenames use relative paths, and the contract checker expects the same identifiers and rejects duplicate or extra CSV rows.
- Missing directories, empty image trees, missing checkpoints, and unreadable PNGs now produce errors. The old unreadable-PNG fallback to a blank tensor is intentionally replaced by an explicit error; the supplied corpus had no unreadable PNGs, so this behavior was not part of the 1,402-image equivalence result.
- The equivalence result covers the supplied 1,402 PNGs on this CPU runtime. It does not establish identical results for other image files, devices, PyTorch versions, or corrupt files. The known incorrect class mapping remains unchanged by request. The assessment's unseen macro-F1 was not measured.

# E1 — class mapping correction

E1 changes only inference decoding order to `clipper, grasper, hook, scissor`, a focused mapping test, and this log. The E0 checkpoint, architecture, preprocessing, discovery, and CLI behavior remain unchanged. The E0 limitation about the incorrect mapping above describes E0, not E1.

## Image-by-image comparison and runtime

`/tmp/e1_capture.py` was a temporary comparison harness. It loaded `data/cholec-tinytools` recursively through `InferenceDataset`, used the deployed checkpoint in eval mode with batch size 64 on CPU, and recorded each relative path, complete float32 preprocessing tensor SHA-256, raw argmax index, decoded label, and parent folder label. Both runs used `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1`.

| Command | Result |
| --- | --- |
| `python /tmp/e1_capture.py /tmp/e1_e0.json` before editing | 1,402 images; inference loop 7.794 s; manifest SHA-256 `c93197911be7759fbf6aa9aaeed4d0fe2e46a1f28e2e412f18025ca437bc32f2`. |
| `python /tmp/e1_capture.py /tmp/e1_e1.json` after editing | 1,402 images; inference loop 7.218 s; manifest SHA-256 `6b31862bf0c82d0c16fece44e515409ebce7252d28dfecbb4113a742d35a5985`. These single-run timings are observations, not a performance comparison. |
| Python comparison of `/tmp/e1_e0.json` and `/tmp/e1_e1.json` | 1,402/1,402 identical paths, tensor hashes, and raw predicted indices. Decoded labels changed for 1,349/1,402 images: 258 grasper→clipper, 559 hook→grasper, 532 clipper→hook. The 53 scissor predictions did not change. On validation alone, 258/277 labels changed. |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v` | 6 tests passed in 0.078 s. |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python check_submission.py` | `PREFLIGHT PASSED`; 277 valid predictions; subprocess runtime reported as 3.4 s. |
| `git diff --check` | No whitespace errors. |

Checkpoint SHA-256 remained `678d57190e979a13f7e6f808fb3cea6ed2f66e990cf545600d3d172b4a6c70e8`. E1 `predict.py` SHA-256 is `23b2779a6fd0718d245b5aeae3acb2777843de01bc4fabbe13c5897b7fe13604`; E1 `tests/test_e0_inference.py` SHA-256 is `838f3c5591fefa9a187883f9de4acb721b388c58cb99a9db9f4022dd625a903b`.

## Folder-label diagnostic

**Assumption:** each image's parent class folder is its correct single ground-truth label. The repository audit found CSV/folder disagreements, duplicate basenames, and validation images used in SmallCNN training, so these are local diagnostics and do not measure unseen performance. Metrics use labels in the order `clipper, grasper, hook, scissor`, with zero division set to zero.

On the 277 supplied validation images, E1 accuracy is **0.815884** and macro-F1 is **0.785209**. For comparison, E0 accuracy was 0.111913 and macro-F1 was 0.211945 against the same folder labels.

| E1 class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| clipper | 49 | 0.745098 | 0.775510 | 0.760000 |
| grasper | 101 | 0.800000 | 0.910891 | 0.851852 |
| hook | 100 | 0.869565 | 0.800000 | 0.833333 |
| scissor | 27 | 0.842105 | 0.592593 | 0.695652 |

E1 confusion matrix, **rows = folder label; columns = predicted label**, both in the class order above:

| Actual \\ predicted | clipper | grasper | hook | scissor |
| --- | ---: | ---: | ---: | ---: |
| clipper | 38 | 9 | 1 | 1 |
| grasper | 2 | 92 | 7 | 0 |
| hook | 6 | 12 | 80 | 2 |
| scissor | 5 | 2 | 4 | 16 |

The same diagnostic on the 1,125 supplied training-folder images yielded E1 accuracy 0.832000 and macro-F1 0.724822; 1,091 decoded labels changed from E0. These training-folder metrics are especially susceptible to training overlap and should not be read as generalization estimates.

# E2 — leakage-safe video-grouped evaluation folds

E2 creates a version 1 five-fold manifest from all 1,402 PNG files in the original `train/` and `validation/` trees. Each sample stores its complete relative path, folder label, and the first number before `_` as video ID. No model training, inference code, checkpoint, or original dataset file changed.

## Selection method

`splits/video_folds.py` enumerates each unlabeled partition of the 10 observed video IDs into five nonempty folds once, rejects partitions missing any of the four classes on either side of any fold, then minimizes the sum of squared relative deviations of each validation-fold class count from one fifth of that class's total count. Ties resolve deterministically from sorted video IDs and canonical fold IDs. There were **6,175 feasible partitions**. The selected score was **5.461419**. Standard `sklearn.model_selection.GroupKFold(n_splits=5)` scored **8.411986** under the identical class-balance function. Both methods happened to put all four classes in every validation fold for this dataset.

| Selected fold | Validation video IDs | Images | Clipper | Grasper | Hook | Scissor |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 7013, 8610 | 260 | 54 | 100 | 90 | 16 |
| 1 | 7201, 7301, 8611 | 461 | 32 | 149 | 212 | 68 |
| 2 | 7414, 7697 | 174 | 67 | 62 | 31 | 14 |
| 3 | 7654, 7695 | 266 | 49 | 98 | 81 | 38 |
| 4 | 9152 | 241 | 60 | 83 | 91 | 7 |

For comparison, standard GroupKFold validation groups and class counts were: `7201` (440; 12/149/212/67), `9152` (241; 60/83/91/7), `7013` (238; 42/100/90/6), `7414+7654` (240; 108/15/69/48), and `7301+7695+7697+8610+8611` (243; 40/145/43/15). Counts are clipper/grasper/hook/scissor. The selected class balance is better under the stated score; its validation-fold sizes range from **174 to 461**, compared with **238 to 440** for GroupKFold. Video 7201 alone has 440 images, limiting achievable size balance.

## Commands, integrity, and hashes

Commands ran from `surgical_tool_id/`:

| Command | Measured result |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 python splits/video_folds.py` | Wrote `splits/video_grouped_fivefold_v1.json` with 1,402 samples, 6,175 feasible partitions, balance score 5.461419. |
| Python read-only comparison using `GroupKFold(n_splits=5)`, `fold_summary`, and `balance_score` | GroupKFold balance score 8.411986; fold video IDs, sizes, and class counts are reported above. |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v` | 9 tests passed in 0.795 s. E2 tests checked every path and folder label, zero train/validation video overlap in each fold, each sample in exactly one validation fold, all four classes on both sides, and byte-for-byte manifest regeneration. |
| `git diff --check` | No whitespace errors. |
| `shasum -a 256 splits/video_folds.py splits/video_grouped_fivefold_v1.json tests/test_e2_splits.py` | SHA-256: builder `82766c83dd41d532baca2178f10ade0724b8f74b12ccca55c24c2b88d6facfa3`; manifest `d82c7984e1ebcbaaf85ba8ca1ec7561cb141ae40ad4cb379d13c661c88fcaf18`; tests `2f55d00030f883c1d4e63aa5f2b0a3db1e93453912f1596b2a4006e1367a4caf`. |

The grouping assumes the first filename number identifies a video, as described in the repository's older split code; no external video metadata was supplied. The class-presence constraint is feasible for this corpus, so the builder requires it and raises an error on a corpus where it cannot be met. The manifest is an evaluation artifact; existing training scripts are not yet wired to consume it. No cross-validation model scores were measured.

# E3 — fresh legacy CNN on E2 video folds

E3 trained five independent CPU SmallCNN models using seed 42, one per E2 fold. Each run initialized a new model and Adam optimizer and read only that fold's `train_paths` for gradient updates. The architecture and image pipeline match `legacy/cnn_baseline_v2.py`: grayscale PIL image, resize to 128×128, float32 / 255, repeat to three channels, four convolution/BatchNorm/ReLU/MaxPool blocks, dropout 0.5, and a four-output linear head. Training used batch size 2, shuffle, Adam at 0.001, softmax, one-hot targets, MSE loss, at most 30 epochs, and the original three-epoch training-loss patience with minimum improvement 1e-4. No pretrained weights, supplied checkpoint, or augmentation was used. Validation data was evaluated only in `eval()` with no gradient, and the checkpoint with the highest validation macro-F1 was selected (earliest epoch on a tie). Class order was `clipper, grasper, hook, scissor` from folder labels in the E2 manifest.

## Commands and verification

Commands ran from `surgical_tool_id/`:

| Command | Result |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e3_legacy_cv/train_cv.py --fold 0` | Fold 0 trained independently. A repeat of this exact command produced identical checkpoint SHA-256 `a45150add507def9b6934fdb3bfb9bc597cce53201612da8b473489115adb1fc` and fold OOF CSV SHA-256 `3098946cc1b5a6275ed318ee3f40108854065ca5bb9d9a51564deff45e84a745`. First runtime 64.838 s, 391.0 MiB peak RSS; repeat runtime 60.741 s, 558.2 MiB peak RSS. The repeat result is retained in `results.json`. |
| `for fold in 1 2 3 4; do PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e3_legacy_cv/train_cv.py --fold "$fold" || exit; done` | Four separate training processes completed. Per-fold times and peak memory appear below. |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e3_legacy_cv/train_cv.py --aggregate` | Wrote `oof_predictions.csv` with 1,402 unique complete relative paths and `results.json` with pooled metrics. Recomputed after fold 0 repeat. |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v` | 15 tests passed in 9.063 s. E3 tests cover seed repeatability at initialization and one optimizer update, architecture state shapes, a known metric fixture, exact OOF coverage, metrics recomputation, selected-epoch maximal F1, checkpoint hashes, and predictions reproduced from all five reloaded checkpoints. |
| `git diff --check` | No whitespace errors. |

The retained five training runs total **802.896 s** measured inside `train_fold`; fold 0's additional repeat took **64.838 s**. Peak process RSS across the retained runs was **594.484 MiB**. RSS includes the interpreter and loaded libraries and was measured with `resource.getrusage` in a separate process per fold. Timing excludes process startup and aggregation. Repeated fold 0 had identical model and predictions but different runtime and peak RSS, so resource figures are observations rather than deterministic properties.

| Fold | Validation n | Selected epoch (zero-based) | Epochs run | Accuracy | Macro-F1 | Runtime s | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 260 | 1 | 5 | 0.680769 | 0.399104 | 60.741 | 558.188 |
| 1 | 461 | 6 | 16 | 0.741866 | 0.588409 | 203.741 | 572.109 |
| 2 | 174 | 3 | 7 | 0.787356 | 0.802395 | 86.461 | 505.328 |
| 3 | 266 | 17 | 18 | 0.781955 | 0.616423 | 231.840 | 594.484 |
| 4 | 241 | 16 | 18 | 0.709544 | 0.602551 | 220.113 | 433.750 |

## Precision, recall, F1, and confusion matrices

All metrics use folder labels as the explicit single-label diagnostic assumption from E2. Class order for the matrices is **clipper, grasper, hook, scissor**; rows are actual folder labels and columns are predictions. Entries in the per-class tables are precision / recall / F1.

| Fold | Clipper P/R/F1 | Grasper P/R/F1 | Hook P/R/F1 | Scissor P/R/F1 |
| --- | --- | --- | --- | --- |
| 0 | 0.000/0.000/0.000 | 0.584/0.900/0.709 | 0.821/0.967/0.888 | 0.000/0.000/0.000 |
| 1 | 0.310/0.688/0.427 | 0.773/0.913/0.837 | 0.866/0.821/0.843 | 0.769/0.147/0.247 |
| 2 | 0.824/0.627/0.712 | 0.714/0.887/0.791 | 0.906/0.935/0.921 | 0.786/0.786/0.786 |
| 3 | 0.580/0.816/0.678 | 0.826/0.969/0.892 | 0.890/0.901/0.896 | 0.000/0.000/0.000 |
| 4 | 0.724/0.700/0.712 | 0.789/0.855/0.821 | 0.902/0.604/0.724 | 0.094/0.429/0.154 |

| Fold | Confusion matrix (four rows) |
| --- | --- |
| 0 | `[[0,47,7,0],[0,90,10,0],[0,3,87,0],[0,14,2,0]]` |
| 1 | `[[22,3,7,0],[8,136,4,1],[23,13,174,2],[18,24,16,10]]` |
| 2 | `[[42,21,2,2],[6,55,0,1],[1,1,29,0],[2,0,1,11]]` |
| 3 | `[[40,5,4,0],[3,95,0,0],[2,6,73,0],[24,9,5,0]]` |
| 4 | `[[42,10,6,2],[3,71,0,9],[11,7,55,18],[2,2,0,3]]` |

**Pooled OOF:** 1,402 samples, accuracy **0.738231**, macro-F1 **0.616462**. This is computed from the 1,402 OOF predictions, not the average of fold F1 scores.

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| clipper | 262 | 0.586345 | 0.557252 | 0.571429 |
| grasper | 492 | 0.730392 | 0.908537 | 0.809783 |
| hook | 505 | 0.867220 | 0.827723 | 0.847011 |
| scissor | 143 | 0.406780 | 0.167832 | 0.237624 |

Pooled confusion matrix: `[[146,86,26,4],[20,447,14,11],[37,30,418,20],[46,49,24,24]]`.

## Artifacts and limits

- `experiments/e3_legacy_cv/oof_predictions.csv` contains exactly one row per E2 sample, keyed by complete relative path, including its video ID, folder label, predicted index and label, and fold. Its SHA-256 is `445eb95d4f92a1e965d867518133567795e667159dfe108b4751409df6175e86`.
- `experiments/e3_legacy_cv/results.json` contains exact per-fold and pooled metrics, confusion matrices, selected epoch, epoch history, runtime, peak RSS, and checkpoint hashes. Its SHA-256 is `23dddbb5694c041487f470e943f0679cd0fff3fad845a8a2ba55964bd28c6275`.
- The training script SHA-256 is `1b93aebc43a851257552181bd4df00d6f0e68ebca9121d4a576358126bf3d910`. The E2 manifest remained `d82c7984e1ebcbaaf85ba8ca1ec7561cb141ae40ad4cb379d13c661c88fcaf18`; deployment `predict.py` remained `23b2779a6fd0718d245b5aeae3acb2777843de01bc4fabbe13c5897b7fe13604`; supplied `checkpoints/model_best.pt` remained `678d57190e979a13f7e6f808fb3cea6ed2f66e990cf545600d3d172b4a6c70e8`.
- These are video-held-out results for the ten filename groups in this dataset, not a private-test score. Folder and CSV labels disagree for some files, and the filename prefix has not been independently verified against video metadata. The five validation sets differ in size and class mix. This experiment did not select a deployment model or change deployment inference.

# E4 — raw-logit cross entropy versus E3 MSE

E4 repeated the five E2 video folds with a separate fresh SmallCNN per fold. A source diff of E3 and E4 training scripts showed only the experiment description/path and the loss-related changes: integer class targets instead of one-hot targets, `nn.CrossEntropyLoss()` instead of `nn.MSELoss()`, and raw logits passed to that criterion instead of softmax probabilities. Both runs used the same seed 42, model architecture, grayscale preprocessing, DataLoader settings, Adam optimizer at 0.001, batch size 2, 30-epoch ceiling, original training-loss patience of 3 with minimum improvement 1e-4, validation macro-F1 checkpoint selection, and no augmentation, pretrained weights, or class weighting. E3 and deployment files were not edited.

## Commands and validation

Commands ran from `surgical_tool_id/`:

| Command | Measured result |
| --- | --- |
| `diff -u experiments/e3_legacy_cv/train_cv.py experiments/e4_cross_entropy_cv/train_cv.py` | Diff limited to experiment paths/description and the target/loss changes described above. |
| `for fold in 0 1 2 3 4; do PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e4_cross_entropy_cv/train_cv.py --fold "$fold" || exit; done` | Five separate CPU training processes completed. Selected epochs, metrics, runtime, and peak RSS are below. |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e4_cross_entropy_cv/train_cv.py --aggregate` | Wrote 1,402 unique-path OOF predictions and pooled metrics. |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v` | 18 tests passed in 18.075 s. E4 tests check tensor/architecture/settings parity with E3, integer targets, analytic raw-logit cross entropy, OOF coverage, recomputed metrics, selected-epoch maximal F1, checkpoint hashes, and predictions from reloaded checkpoints. E3 tests also passed. |
| E3 fold-0 `train_fold(0)` rerun after loading the unchanged E3 script with a temporary output directory | Checkpoint SHA-256 matched `a45150add507def9b6934fdb3bfb9bc597cce53201612da8b473489115adb1fc`; OOF CSV SHA-256 matched `3098946cc1b5a6275ed318ee3f40108854065ca5bb9d9a51564deff45e84a745`. Rerun time 80.089 s, peak RSS 435.1 MiB. E3 committed outputs were not rewritten. |
| `git diff --check` | No whitespace errors. |

The five E4 runs took **1,563.102 s** in total inside `train_fold`, versus E3's **802.896 s**; this is an observed runtime comparison on this host, including different numbers of epochs. Maximum per-process peak RSS was **633.688 MiB** for E4 and **594.484 MiB** for E3. Peak RSS includes the interpreter and libraries. Timings exclude process startup and aggregation.

## Fold metrics and direct E3 comparison

All metrics use the E2 folder labels as the explicit single-label diagnostic assumption. Class order throughout is **clipper, grasper, hook, scissor**. The selected checkpoint is the validation macro-F1 maximum among epochs run, with the earliest epoch winning ties.

| Fold | n | Selected epoch (zero-based) | Epochs run | E4 accuracy | E4 macro-F1 | E3 macro-F1 | E4 runtime s | E4 peak RSS MiB | Changed OOF labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 260 | 20 | 25 | 0.865385 | 0.782046 | 0.399104 | 341.563 | 438.938 | 72 |
| 1 | 461 | 3 | 26 | 0.783080 | 0.646652 | 0.588409 | 282.606 | 631.719 | 76 |
| 2 | 174 | 7 | 22 | 0.781609 | 0.774133 | 0.802395 | 264.344 | 629.844 | 30 |
| 3 | 266 | 15 | 25 | 0.823308 | 0.752606 | 0.616423 | 326.056 | 633.688 | 47 |
| 4 | 241 | 14 | 25 | 0.771784 | 0.651161 | 0.602551 | 348.533 | 426.734 | 64 |

Per-class E4 values are **precision / recall / F1**:

| Fold | Clipper | Grasper | Hook | Scissor |
| --- | --- | --- | --- | --- |
| 0 | .796/.722/.757 | .885/.920/.902 | .897/.967/.930 | .700/.438/.538 |
| 1 | .789/.469/.588 | .751/.933/.832 | .805/.915/.857 | .812/.191/.310 |
| 2 | .844/.567/.679 | .711/.952/.814 | .909/.968/.938 | .692/.643/.667 |
| 3 | .755/.755/.755 | .887/.959/.922 | .789/.926/.852 | .812/.342/.481 |
| 4 | .827/.717/.768 | .792/.735/.763 | .940/.868/.903 | .107/.429/.171 |

E4 confusion matrices, **rows = actual folder labels; columns = predictions**:

| Fold | Matrix, four rows in class order above |
| --- | --- |
| 0 | `[[39,10,3,2],[3,92,5,0],[1,1,87,1],[6,1,2,7]]` |
| 1 | `[[15,5,11,1],[2,139,6,2],[2,16,194,0],[0,25,30,13]]` |
| 2 | `[[38,23,2,4],[2,59,1,0],[1,0,30,0],[4,1,0,9]]` |
| 3 | `[[37,5,5,2],[4,94,0,0],[2,3,75,1],[6,4,15,13]]` |
| 4 | `[[43,9,4,4],[2,61,0,20],[7,4,79,1],[0,3,1,3]]` |

**Pooled E4 OOF:** 1,402 samples, accuracy **0.803852**, macro-F1 **0.713121**. E3 pooled accuracy was **0.738231**, macro-F1 **0.616462**. E4 minus E3 was **+0.065621 accuracy** and **+0.096659 macro-F1**. Fold 2's macro-F1 decreased by 0.028262; the other four increased. Pooled macro-F1 is computed from all OOF rows, not averaged from fold macro-F1 values.

| Pooled class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| clipper | 262 | 0.803738 | 0.656489 | 0.722689 |
| grasper | 492 | 0.801802 | 0.904472 | 0.850048 |
| hook | 505 | 0.845455 | 0.920792 | 0.881517 |
| scissor | 143 | 0.542169 | 0.314685 | 0.398230 |

Pooled E4 confusion matrix: `[[172,52,25,13],[13,445,12,22],[13,24,465,3],[16,34,48,45]]`.

Joining E3 and E4 OOF CSVs by complete relative path found **289 changed predictions** (folds 0–4: 72, 76, 30, 47, 64). Among changed rows, **160 went from wrong to correct**, **68 from correct to wrong**, and **61 remained wrong** under the folder-label assumption. The other 1,113 predicted labels were unchanged.

## Artifacts and limits

- E4 `oof_predictions.csv` SHA-256: `bb0b71297e962d2cd8c1eeeffcb987d701c08ab70cadf996cf71c84cbf946ff9`. It has one complete relative path per E2 sample.
- E4 `results.json` SHA-256: `f208d13270e6db22e32a2c497e534afeb3f53c7918264effe62a92e6bed79a6a`. It stores exact per-class and pooled metrics, matrices, epoch histories, runtime, memory, and checkpoint hashes.
- E4 `train_cv.py` SHA-256: `114932dca90fb14786dd7991db646b03f2a8b9d8477df9012cd715ddc65cdc93`. E4 test SHA-256: `0f422f0ec47cf4c22f340daa5fc0921754aa3a43b38e870a2476c2f162c454d4`.
- E4 checkpoint SHA-256 by fold: `4b2ae5641307feeb662ea2b1256cd59ceed00f338477d89f12ff8c431ca8f80d`, `dcd7b0e9f2c362792863535ac43a99a2a090c99122def2e9893cc5c7491fd092`, `55b809b7f72b4fd03e61355050be526fe52bb2e22ac6c041c9fe32ac630d0446`, `042a85addc03eef62d68da1ac9ae706520e365caa16183b0e9269bb19ad74afc`, `bea47694454c4a11e0f5f83e9d397de97f90bf7738174d974424a379d7cd6536`.
- This is a comparison on ten filename-derived video groups with the known folder/CSV label disagreements. It is not a private-test result or a deployment-model selection. Different losses changed training trajectories and early-stopping epochs; the runtime difference is not a per-epoch speed claim. Deployment inference and its checkpoint were unchanged.
