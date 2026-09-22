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

# E5 — fold-training-only inverse-frequency class weights

E5 repeats E4 on the same five E2 folds with only a weighted cross-entropy criterion. For each fold, class counts come from that fold's `train_paths` only. The weight for class `c` is `(1 / train_count[c]) / mean_j(1 / train_count[j])`, computed as float32; the mean of the four weights is one. E4 architecture, grayscale preprocessing, seed 42, data order, Adam at 0.001, batch size 2, raw-logit cross entropy with integer targets, 30-epoch ceiling, training-loss patience, and validation macro-F1 checkpoint selection are unchanged. No class weights use held-out labels. No deployment inference or E4 file was edited.

## Commands and checks

Commands ran from `surgical_tool_id/`:

| Command | Measured result |
| --- | --- |
| `diff -u experiments/e4_cross_entropy_cv/train_cv.py experiments/e5_weighted_ce_cv/train_cv.py` | Training-path differences were the class-weight function, its training-set-only call, `CrossEntropyLoss(weight=...)`, and recorded class-count/weight metadata. E5 aggregation additionally records the comparison to E4. |
| `for fold in 0 1 2 3 4; do PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e5_weighted_ce_cv/train_cv.py --fold "$fold" || exit; done` | Five independent CPU training processes completed. Per-fold runtime and process peak RSS appear below. |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e5_weighted_ce_cv/train_cv.py --aggregate` | Wrote 1,402 unique-path OOF predictions, pooled metrics, and `comparison_to_e4.json`. |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v` | 24 tests passed in 27.428 s. E5 tests verify inverse-frequency ratios, mean-one normalization, absent-class errors, per-fold training counts and weights, parity with E4 non-loss settings and tensors, analytic weighted raw-logit CE, complete OOF coverage, metrics, selected epochs, checkpoint hashes, reloaded-checkpoint predictions, and pooled-macro-F1 selection. E4 tests also passed. |
| E4 fold-0 `train_fold(0)` rerun after loading the unchanged E4 script with a temporary output directory | Checkpoint SHA-256 matched `4b2ae5641307feeb662ea2b1256cd59ceed00f338477d89f12ff8c431ca8f80d`; fold OOF CSV SHA-256 matched `81163ce49fd0c0479fa51b85e052c62a6ab73c152471b3653fa97ae45b460212`. Rerun took 308.199 s and peaked at 543.2 MiB RSS. E4 artifacts were not rewritten. |
| `git diff --check` | No whitespace errors. |

E5's five retained runs totaled **1,379.055 s** inside `train_fold`, versus E4's **1,563.102 s**. The maximum fold-process peak RSS was **614.625 MiB** for E5, versus **633.688 MiB** for E4. These are single-run observations; differing epoch counts affect total runtime, and process RSS includes the interpreter and libraries.

## Fold training counts and normalized weights

Values follow class order **clipper, grasper, hook, scissor**. Counts are from each fold's training paths; the held-out video groups were excluded before counting.

| Fold | Training counts | Mean-one inverse-frequency weights |
| --- | --- | --- |
| 0 | 208, 392, 415, 127 | 1.090033, 0.578385, 0.546330, 1.785251 |
| 1 | 230, 343, 293, 75 | 0.724348, 0.485715, 0.568601, 2.221335 |
| 2 | 195, 430, 474, 129 | 1.184656, 0.537228, 0.487358, 1.790758 |
| 3 | 213, 394, 424, 105 | 0.982430, 0.531110, 0.493532, 1.992929 |
| 4 | 202, 409, 414, 136 | 1.153701, 0.569798, 0.562917, 1.713585 |

## Metrics and direct E4 comparison

All metrics treat the E2 folder labels as the explicit single-label diagnostic assumption. Within each fold, the best checkpoint is selected by validation macro-F1; **pooled OOF macro-F1 is the primary metric for selecting between the E4 and E5 experiments**. There was no deployment-model selection.

| Fold | n | Selected epoch (zero-based) | Epochs run | E5 accuracy | E5 macro-F1 | E4 macro-F1 | Runtime s | Peak RSS MiB | Changed OOF labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 260 | 20 | 22 | 0.873077 | 0.781285 | 0.782046 | 255.117 | 438.031 | 21 |
| 1 | 461 | 18 | 30 | 0.750542 | 0.660824 | 0.646652 | 337.883 | 597.125 | 93 |
| 2 | 174 | 0 | 21 | 0.821839 | 0.828877 | 0.774133 | 250.502 | 604.016 | 29 |
| 3 | 266 | 21 | 26 | 0.838346 | 0.790955 | 0.752606 | 293.407 | 614.625 | 38 |
| 4 | 241 | 14 | 19 | 0.734440 | 0.630953 | 0.651161 | 242.146 | 538.828 | 29 |

Per-class E5 values are **precision / recall / F1**:

| Fold | Clipper | Grasper | Hook | Scissor |
| --- | --- | --- | --- | --- |
| 0 | .820/.759/.788 | .938/.910/.924 | .848/.989/.913 | .750/.375/.500 |
| 1 | .491/.812/.612 | .760/.893/.821 | .892/.778/.831 | .458/.324/.379 |
| 2 | .839/.701/.764 | .753/.935/.835 | 1.000/.871/.931 | .786/.786/.786 |
| 3 | .733/.673/.702 | .887/.959/.922 | .901/.901/.901 | .676/.605/.639 |
| 4 | .824/.700/.757 | .785/.614/.689 | .930/.879/.904 | .103/.571/.174 |

E5 confusion matrices, **rows = actual folder labels; columns = predictions**, in the class order above:

| Fold | Matrix, four rows |
| --- | --- |
| 0 | `[[41,5,7,1],[1,91,7,1],[0,1,89,0],[8,0,2,6]]` |
| 1 | `[[26,2,3,1],[5,133,5,6],[13,15,165,19],[9,25,12,22]]` |
| 2 | `[[47,19,0,1],[4,58,0,0],[2,0,27,2],[3,0,0,11]]` |
| 3 | `[[33,5,2,9],[3,94,1,0],[1,5,73,2],[8,2,5,23]]` |
| 4 | `[[42,7,6,5],[3,51,0,29],[6,4,80,1],[0,3,0,4]]` |

**Pooled E5 OOF:** 1,402 samples, accuracy **0.796006**, macro-F1 **0.728552**. E4 pooled accuracy was **0.803852** and macro-F1 **0.713121**. E5 gained **0.015431 macro-F1** and lost **0.007846 accuracy**. By the declared primary metric, `comparison_to_e4.json` selects **E5**. Fold 0 and fold 4 macro-F1 decreased slightly; the pooled score increased. The pooled macro-F1 is computed from all OOF predictions, not from the average of fold F1 values.

| Pooled class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| clipper | 262 | 0.741176 | 0.721374 | 0.731141 |
| grasper | 492 | 0.821154 | 0.867886 | 0.843874 |
| hook | 505 | 0.896694 | 0.859406 | 0.877654 |
| scissor | 143 | 0.461538 | 0.461538 | 0.461538 |

Pooled E5 confusion matrix: `[[189,38,18,17],[16,427,13,36],[22,25,434,24],[28,30,19,66]]`.

Joining E4 and E5 OOF CSVs by complete relative path found **210 changed predictions** (folds 0–4: 21, 93, 29, 38, 29). Among changed rows, **81 went from wrong to correct**, **92 from correct to wrong**, and **37 remained wrong** under the folder-label assumption. The other 1,192 predicted labels were unchanged.

## Artifacts and limits

- E5 `oof_predictions.csv` SHA-256: `f8e7727eb52b1683e0b55933c04009357c2c5401bc4e6404f7cdd3e811688b18`. It contains one complete relative path per E2 sample.
- E5 `results.json` SHA-256: `b4feae272d9f6e3372af1d7dde79435df9f835ed3904bbcbc82438f49f2c0a5b`. It includes exact per-fold weights, metrics, matrices, history, resource results, and checkpoint hashes.
- E5 `comparison_to_e4.json` SHA-256: `24289792066a77f6db7a91ab682f5c561a06c662ff7afedbb077088dc6240b8d`. Primary metric: pooled macro-F1; selected experiment: E5.
- E5 training-script SHA-256: `7bc19b21566e9c3453819892150e359fde6b79943080e3e90bc1320c61e21994`. E5 test SHA-256: `a33e1ed5c0e3c6f70baaff2c2d92dcfb3f232fe3882e3586d65e8c761762e9d9`.
- E5 checkpoint SHA-256 by fold: `17de74b864b7b0c4693e560160d1f2e98e5a9a699bcdf07470026773c6bdb767`, `541647e8483d500890b020365134688c4eef0926807d2f31534193ad01c1b067`, `9663b4b640c8c0ef7a8f393b6b0c763960e91c3ccfbc90b12eb6a85947703354`, `74e019f89817857a5cc076d44ba6ae10774510ceb6e52027740f9581e5cd9267`, `ff2ac474f7c50a1ca756ed3390665bf12bc2437a150fd0299ddfd463432af8df`.
- This selection is based on video-held-out folder-label diagnostics for ten filename-derived groups with known CSV/folder label disagreements. It is not a private-test result. Selecting E5 by pooled macro-F1 does not authorize or enact deployment; `predict.py` and its checkpoint remain unchanged.

# E6 — true RGB pixels versus E5 grayscale replication

E6 repeats the five E2 video folds with the E5 architecture, fold-training-only inverse-frequency class weights, loss, optimizer, seed, batch size, 128×128 resize, float32 `/255` scaling, training-loss patience, and per-fold validation macro-F1 checkpoint selection. The sole training-pipeline change is that `FrameDataset` reads true RGB pixels as channel-first tensors instead of converting to grayscale and replicating one channel three times. No augmentation, pretrained weights, or class weighting change was introduced. E5 and deployment files were not edited.

## Commands and verification

Commands ran from `surgical_tool_id/`:

| Command | Measured result |
| --- | --- |
| `diff -u experiments/e5_weighted_ce_cv/train_cv.py experiments/e6_rgb_cv/train_cv.py` | Training-path difference was only the RGB image conversion and channel layout. Experiment paths and comparison labels changed in aggregation. |
| `for fold in 0 1 2 3 4; do PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e6_rgb_cv/train_cv.py --fold "$fold" || exit; done` | Five separate CPU training processes completed. Per-fold runtime and process peak RSS appear below. |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e6_rgb_cv/train_cv.py --aggregate` | Wrote 1,402 unique-path OOF predictions, pooled metrics, and `comparison_to_e5.json`. |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v` | 29 tests passed in 33.005 s. E6 tests check RGB channel separation on a known color image, exact resized RGB pixels and scaling on a dataset image, architecture/settings/weight parity with E5, OOF coverage and metrics, selected epochs, checkpoint hashes, reloaded-checkpoint predictions, and pooled-macro-F1 selection. E5 tests also passed. |
| E5 fold-0 `train_fold(0)` rerun after loading the unchanged E5 script with a temporary output directory | Checkpoint SHA-256 matched `17de74b864b7b0c4693e560160d1f2e98e5a9a699bcdf07470026773c6bdb767`; fold OOF CSV SHA-256 matched `f33ba7f597ae923ec4b368a15a9620c2ff9797cb2fc2db22f0d4b7205ed45cab`. Rerun took 251.201 s and peaked at 602.5 MiB RSS. E5 artifacts were not rewritten. |
| `git diff --check` | No whitespace errors. |

The five E6 runs totaled **1,925.400 s** inside `train_fold`, versus E5's **1,379.055 s**. Maximum fold-process peak RSS was **621.859 MiB** for E6 and **614.625 MiB** for E5. These are single-run observations, not a controlled per-epoch speed measurement; training epoch counts and Python image conversion differed. RSS includes the interpreter and libraries.

## Fold metrics and direct E5 comparison

Metrics use E2 folder labels as the explicit single-label diagnostic assumption. Class order is **clipper, grasper, hook, scissor**. Per-fold checkpoints were selected by validation macro-F1; **pooled OOF macro-F1 is the primary E5 versus E6 comparison metric**. No deployment-model selection was made.

| Fold | n | Selected epoch (zero-based) | Epochs run | E6 accuracy | E6 macro-F1 | E5 macro-F1 | Runtime s | Peak RSS MiB | Changed OOF labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 260 | 8 | 22 | 0.926923 | 0.847507 | 0.781285 | 382.687 | 505.484 | 31 |
| 1 | 461 | 21 | 24 | 0.739696 | 0.641411 | 0.660824 | 354.151 | 588.781 | 102 |
| 2 | 174 | 12 | 30 | 0.821839 | 0.789237 | 0.828877 | 467.527 | 591.875 | 30 |
| 3 | 266 | 1 | 24 | 0.849624 | 0.819542 | 0.790955 | 350.238 | 621.859 | 47 |
| 4 | 241 | 16 | 24 | 0.809129 | 0.694316 | 0.630953 | 370.797 | 619.719 | 54 |

Per-class E6 values are **precision / recall / F1**:

| Fold | Clipper | Grasper | Hook | Scissor |
| --- | --- | --- | --- | --- |
| 0 | .862/.926/.893 | .980/.960/.970 | .946/.967/.956 | .667/.500/.571 |
| 1 | .315/.875/.463 | .847/.852/.849 | .896/.769/.827 | .575/.338/.426 |
| 2 | .865/.672/.756 | .847/.984/.910 | .757/.903/.824 | .692/.643/.667 |
| 3 | .812/.796/.804 | .957/.898/.926 | .857/.889/.873 | .643/.711/.675 |
| 4 | .902/.617/.733 | .768/.916/.835 | .951/.857/.902 | .211/.571/.308 |

E6 confusion matrices, **rows = actual folder labels; columns = predictions**, in the class order above:

| Fold | Matrix, four rows |
| --- | --- |
| 0 | `[[50,1,1,2],[0,96,4,0],[0,1,87,2],[8,0,0,8]]` |
| 1 | `[[28,1,2,1],[12,127,3,7],[28,12,163,9],[21,10,14,23]]` |
| 2 | `[[45,10,8,4],[1,61,0,0],[2,1,28,0],[4,0,1,9]]` |
| 3 | `[[39,3,4,3],[2,88,2,6],[2,1,72,6],[5,0,6,27]]` |
| 4 | `[[37,13,4,6],[1,76,0,6],[2,8,78,3],[1,2,0,4]]` |

**Pooled E6 OOF:** 1,402 samples, accuracy **0.817404**, macro-F1 **0.754132**. E5 pooled accuracy was **0.796006** and macro-F1 **0.728552**. E6 gained **0.021398 accuracy** and **0.025580 macro-F1**. By the declared primary metric, `comparison_to_e5.json` selects **E6**. Fold 1 and fold 2 macro-F1 declined, while the pooled score increased. Pooled macro-F1 is computed from all OOF predictions, not averaged from fold macro-F1 values.

| Pooled class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| clipper | 262 | 0.690972 | 0.759542 | 0.723636 |
| grasper | 492 | 0.876712 | 0.910569 | 0.893320 |
| hook | 505 | 0.897275 | 0.847525 | 0.871690 |
| scissor | 143 | 0.563492 | 0.496503 | 0.527881 |

Pooled E6 confusion matrix: `[[199,28,19,16],[16,448,9,19],[34,23,428,20],[39,12,21,71]]`.

Joining E5 and E6 OOF CSVs by complete relative path found **264 changed predictions** (folds 0–4: 31, 102, 30, 47, 54). Under the folder-label assumption, **119 went from wrong to correct**, **89 from correct to wrong**, and **56 remained wrong**. The other 1,138 predicted labels were unchanged.

## Artifacts and limits

- E6 `oof_predictions.csv` SHA-256: `ea93c81d8059fba16b56965a36f879f105be260a6e69432c504d37bec64bad25`. It contains one complete relative path per E2 sample.
- E6 `results.json` SHA-256: `e8cb61da221c7840a799e2c05c34e9635a72a34f070f1b913cd8c2c1222d0b2c`. It contains exact fold weights, metrics, matrices, histories, resources, and checkpoint hashes.
- E6 `comparison_to_e5.json` SHA-256: `20d1a1b1f24d51e968013900577a08952a4a3e2d4cda1e41a897e13d69e462c2`. Primary metric: pooled macro-F1; selected experiment: E6.
- E6 training-script SHA-256: `8aa6e6c26d5547d8e661e4101069eb41ca07d07349b73897f52c5f63f76d180d`. E6 test SHA-256: `8beed5d3699a26e8b4948fd1014e60cc31299e09d117a85c9edd7a6de8ee511c`.
- E6 checkpoint SHA-256 by fold: `73e12e8d35cc7db9b1dc79e01115fd2176deda635c2b182d3342bbddba7be151`, `9e2eb61e7800a5c2b1635a92a8624699c69d70250f93c207c6b2c403a8492b11`, `719059ce2ce677f0ce7a597e9247471f81f4472a2195a51d72d1f1675bab5efa`, `5a677348402ea75155966e5c7b8b461364bfc685c67af3ec16d4fb9b52599c9c`, `064e7b7f037185df815725111fa69dcc7adfe084824a297f5e196b7c4b0e1f6d`.
- This comparison is on ten filename-derived video groups under the folder-label assumption, with known CSV/folder label disagreements. It is not a private-test result. The RGB improvement does not by itself authorize deployment; `predict.py` and its checkpoint remain unchanged.

# E7 — frozen pretrained ResNet18

E7 uses the E2 video-grouped folds, seed 42, batch size 2, Adam learning rate 0.001, 30-epoch ceiling, training-loss patience 3 with 0.0001 minimum improvement, fold-training-only mean-one inverse-frequency cross-entropy weights, and highest validation macro-F1 checkpoint selection from E6. The new backbone is ResNet18 with the official [PyTorch `IMAGENET1K_V1` weights](https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html), bundled locally. Its [published preprocessing](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18) is RGB, resize shorter side to 256 with bilinear interpolation, center crop 224, scale by 1/255, and normalize with ImageNet means and standard deviations. The local implementation accepts the official state dict with `strict=True` and needs no `torchvision` installation or network request. `torch==2.6.0` was installed; `torchvision` was unavailable in this environment.

Every parameter outside the new four-class `fc` head has `requires_grad=False`. The backbone and all BatchNorm modules stay in eval mode. With fixed preprocessing and no augmentation, E7 extracts each image's 512-dimensional frozen feature once without gradients, then trains a fresh independent linear head for each fold. The feature cache contains all E2 paths and fixed features but no labels. Fold training reads labels from the E2 manifest and computes weights from that fold's training paths only. This is mathematically the same head optimization as repeatedly evaluating the frozen backbone, subject to floating-point batch-size effects. No supplied surgical checkpoint or deployment inference was changed.

## Commands and verification

Commands ran from `surgical_tool_id/` with CPU execution and `OMP_NUM_THREADS=1`:

| Command | Measured result |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e7_frozen_resnet18/train_cv.py --cache` | Extracted 1,402 × 512 features in 98.661 s; peak process RSS 601.297 MiB. Backbone state SHA-256 before and after was identical: `c6fc025cae67cee295fa18eff0998f3ae5f0e7a7d662ca38ab7be77ad49665c9`. |
| `for fold in 0 1 2 3 4; do PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e7_frozen_resnet18/train_cv.py --fold "$fold" || exit; done` | Five fresh, independent head fits completed. Fold runtimes, memory, and selected epochs appear below. |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e7_frozen_resnet18/train_cv.py --aggregate` | Wrote 1,402 unique-path OOF predictions, pooled metrics, and E6 comparison. |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v` | 36 tests passed in 35.300 s. E7 checks bundled offline loading, RGB ImageNet normalization, frozen backbone and BatchNorm tensors after a head optimizer step, cached versus fresh features, offline image inference versus OOF, all reloaded head checkpoints and metrics, and pooled-macro-F1 selection. Earlier E0–E6 tests passed. |
| E7 fold-0 `train_fold(0)` rerun into a temporary directory from the bundled feature cache | Checkpoint SHA-256 matched `9612ef102214093292e3d3024d3f0d27acb5bc317eabc196009185774bc4521f`; fold OOF CSV SHA-256 matched `f2ea7a8b37684fc2cf2a6e85ed58c6295822f66cfa2a071c4fb6be2dcf731f0e`. Rerun fold time 2.608 s and peak RSS 257.0 MiB. Retained E7 files were not rewritten. |

The five retained head fits totaled **12.388 s** inside `train_fold`, in addition to the **98.661 s** shared feature extraction. Their maximum process peak RSS was **259.203 MiB**; extraction peaked at **601.297 MiB**. These are single CPU runs and exclude command startup and aggregation time. E6's full CNN fold fits took 1,925.400 s in total, but that is a different architecture and workload, so the runtimes are not per-epoch speed comparisons.

## Fold metrics and direct E6 comparison

Metrics use the E2 **folder labels as an explicit single-label diagnostic assumption**. There are known folder/CSV label disagreements. Class order is **clipper, grasper, hook, scissor**. Fold checkpoints use validation macro-F1; pooled OOF macro-F1 is the primary E6-versus-E7 experiment metric.

| Fold | n | Selected epoch (zero-based) | Epochs run | E7 accuracy | E7 macro-F1 | E6 macro-F1 | Head runtime s | Peak RSS MiB | Changed OOF labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 260 | 18 | 30 | 0.876923 | 0.821867 | 0.847507 | 2.773 | 257.797 | 44 |
| 1 | 461 | 5 | 30 | 0.696312 | 0.598483 | 0.641411 | 2.085 | 259.141 | 188 |
| 2 | 174 | 8 | 30 | 0.902299 | 0.836187 | 0.789237 | 2.565 | 258.562 | 34 |
| 3 | 266 | 6 | 30 | 0.864662 | 0.829730 | 0.819542 | 2.418 | 258.703 | 67 |
| 4 | 241 | 15 | 30 | 0.796680 | 0.660062 | 0.694316 | 2.546 | 259.203 | 65 |

Per-class E7 values are **precision / recall / F1**:

| Fold | Clipper | Grasper | Hook | Scissor |
| --- | --- | --- | --- | --- |
| 0 | .898/.815/.854 | .916/.870/.892 | .870/.967/.916 | .625/.625/.625 |
| 1 | .338/.781/.472 | .928/.517/.664 | .909/.896/.903 | .305/.426/.356 |
| 2 | .934/.851/.891 | .939/1.000/.969 | .861/1.000/.925 | .636/.500/.560 |
| 3 | .750/.857/.800 | .966/.878/.920 | .886/.963/.923 | .727/.632/.676 |
| 4 | .930/.667/.777 | .857/.795/.825 | .955/.923/.939 | .061/.286/.100 |

E7 confusion matrices, **rows = actual folder labels; columns = predictions**:

| Fold | Matrix, four rows |
| --- | --- |
| 0 | `[[44,4,2,4],[3,87,8,2],[0,3,87,0],[2,1,3,10]]` |
| 1 | `[[25,1,3,3],[21,77,6,45],[4,0,190,18],[24,5,10,29]]` |
| 2 | `[[57,3,3,4],[0,62,0,0],[0,0,31,0],[4,1,2,7]]` |
| 3 | `[[42,0,0,7],[3,86,7,2],[0,3,78,0],[11,0,3,24]]` |
| 4 | `[[40,8,2,10],[2,66,1,14],[0,0,84,7],[1,3,1,2]]` |

**Pooled E7 OOF:** 1,402 samples, accuracy **0.804565**, macro-F1 **0.738166**. E6 pooled accuracy was **0.817404** and macro-F1 **0.754132**. E7 fell by **0.012839 accuracy** and **0.015966 macro-F1**. By the declared primary metric, `comparison_to_e6.json` selects **E6**. Pooled macro-F1 comes from all OOF predictions, rather than the mean of fold F1 values.

| Pooled class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| clipper | 262 | 0.734982 | 0.793893 | 0.763303 |
| grasper | 492 | 0.921951 | 0.768293 | 0.838137 |
| hook | 505 | 0.902111 | 0.930693 | 0.916179 |
| scissor | 143 | 0.382979 | 0.503497 | 0.435045 |

Pooled E7 confusion matrix: `[[208,16,10,28],[29,378,22,63],[4,6,470,25],[42,10,19,72]]`.

Joining E6 and E7 OOF CSVs by complete relative path found **398 changed predictions** (folds 0–4: 44, 188, 34, 67, 65). Under the folder-label assumption, **164 went from wrong to correct**, **182 from correct to wrong**, and **52 remained wrong**. The other 1,004 predicted labels were unchanged.

## Artifacts and limits

- Official bundled weights SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` (46,830,571 bytes). Source URL and expected hash are in `weights/weights_manifest.json`; the local file is loaded directly for offline training and inference.
- E2 manifest SHA-256: `d82c7984e1ebcbaaf85ba8ca1ec7561cb141ae40ad4cb379d13c661c88fcaf18`. E7 feature cache SHA-256: `3df7b344b20a6179d6038f24a22f2eff238243cc85b4271f2bef285ba9791e12`.
- E7 `oof_predictions.csv` SHA-256: `20d744c452884804cb3473e17086007d5fbd78f4de4e8a36abda0772fe4e5d18`. E7 `results.json` SHA-256: `3a35ae43452a1db896a8202a4eddb50ce930f3aa509d1822402792f57655ccff`. E7 `comparison_to_e6.json` SHA-256: `5b8b6c9549be8e536d8ab37e708d2a8101c60b0566c1a144c9f7bc8fa5470dc1`.
- E7 checkpoint SHA-256 by fold: `9612ef102214093292e3d3024d3f0d27acb5bc317eabc196009185774bc4521f`, `5b8921c873fd349cf9a6950cfa254995d542f9ca2a321b755867ed9a3b798142`, `dce1e87cc011938ed3bf2c5c6d995c8999ee5e9e46c4b1c10260e71e3135a43c`, `165337c8873e2a9463b5e42dc73de801400107c7a31cf7761954f0aa82f6dc3e`, `4d83764220a56ff1caa6bf6f21cac5e258ca02fa64035de755f38238669a0f01`.
- This is an OOF comparison on ten filename-derived video groups, under the folder-label assumption. It is not a private-test result. The head is trained on features extracted in batches of 16; the offline image inference check compared three fold-0 images individually and reproduced their predictions. No claim of byte-identical feature tensors across every inference batch size is made. The existing `predict.py` and deployment checkpoint remain unchanged.

# E8 — partial ResNet18 fine tuning

E8 uses E7's bundled official ResNet18 `IMAGENET1K_V1` weights, RGB ImageNet preprocessing, and seed-42 initial four-class head tensors. The E2 video folds, batch size 2, fold-training-only mean-one inverse-frequency class weights, raw-logit weighted cross entropy, Adam, 30-epoch ceiling, training-loss patience 3 with 0.0001 minimum improvement, and highest-validation-macro-F1 checkpoint rule remain as in E7. **Only layer-4 convolution/downsample parameters and the head train.** Layer 4 uses learning rate **0.0001**, ten times lower than the head's **0.001**. Every earlier-layer parameter and **every BatchNorm affine parameter, running mean, running variance, and batch counter** stays frozen; all BatchNorm modules remain in eval mode. No augmentation or deployment inference changed.

E8 extracts the fixed activations through layer 3 once, then trains layer 4 and the head independently for each fold. The cache contains 1,402 complete relative paths and activations, without labels. This preserves the full network forward path while avoiding repeated frozen-prefix computation; a test compares the prefix/suffix composition with the full model. The activation cache is **281,440,826 bytes (268.403 MiB)**, excluded from Git and regenerable offline with `--cache` using the dataset images and E7's bundled weights. Each selected checkpoint stores only trainable layer-4 and head tensors; offline single-image inference reconstructs the remaining state from E7's bundled weights.

## Commands and verification

Commands ran from `surgical_tool_id/` on CPU. The retained folds ran serially in five separate Python processes:

| Command | Measured result |
| --- | --- |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e8_partial_resnet18/train_cv.py --cache` | Produced `(1402,256,14,14)` float32 activations in 79.776 s; peak process RSS 816.281 MiB. Frozen-state SHA-256 before and after was `6c344528608ded2e880a4f337df3613846570225a1def765d10d60c2500445ec`. |
| `for fold in 0 1 2 3 4; do PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e8_partial_resnet18/train_cv.py --fold "$fold" > "experiments/e8_partial_resnet18/fold_${fold}_console.log" 2>&1 || exit 1; done` | Five corrected, independent fold fits completed. Each fold's frozen-state SHA-256 before and after training equaled the cache value above. Per-fold selected epochs, runtime, and memory appear below. Console logs are local and excluded from Git. |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python experiments/e8_partial_resnet18/train_cv.py --aggregate` | Wrote 1,402 unique-path OOF rows, pooled metrics, and direct E7/E6 comparison. |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v` | **42 tests passed in 38.841 s.** E8 tests check exact E7 initial head tensors/settings/preprocessing/fold weights, the trainable parameter set and learning-rate ratio, all frozen/BatchNorm tensors after an optimizer step, offline bundled loading, full-versus-split forward equivalence, OOF coverage and metrics, selected checkpoints, offline single-image inference, and primary pooled-macro-F1 selection. Earlier E0–E7 tests also passed. |

The first set of E8 runs was discarded before this retained experiment: a review found that layer-4 BatchNorm affine weight and bias tensors still had gradients. The loader now freezes those tensors as well as all BatchNorm buffers, and the cache and all five folds were regenerated. Earlier results are excluded from every E8 artifact and comparison here.

The five retained fold fits totaled **1,424.002 s**, plus **79.776 s** for shared feature extraction. Maximum fold-process peak RSS was **769.297 MiB**; feature extraction peaked at **816.281 MiB**. These are single CPU-run observations and exclude shell startup and aggregation time. `frozen_state_hash` covers all parameters outside the trainable layer-4 convolution/downsample and head set, including every BatchNorm tensor. It was byte-identical before and after every fold; the optimizer-step test also checks each frozen tensor and absent gradients directly.

## Fold metrics and direct comparison

Metrics use E2 **folder labels as an explicit single-label diagnostic assumption**; known folder/CSV disagreements remain. Class order is **clipper, grasper, hook, scissor**. Fold checkpoints use validation macro-F1. **Pooled OOF macro-F1 is the primary metric** for comparing E8 with E7 and the current E6 candidate.

| Fold | n | Selected epoch (zero-based) | Epochs run | E8 accuracy | E8 macro-F1 | E7 macro-F1 | E6 macro-F1 | Runtime s | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 260 | 8 | 10 | 0.934615 | 0.870037 | 0.821867 | 0.847507 | 302.691 | 608.891 |
| 1 | 461 | 11 | 12 | 0.876356 | 0.813998 | 0.598483 | 0.641411 | 330.473 | 608.797 |
| 2 | 174 | 8 | 9 | 0.971264 | 0.942050 | 0.836187 | 0.789237 | 279.163 | 769.297 |
| 3 | 266 | 5 | 7 | 0.894737 | 0.885021 | 0.829730 | 0.819542 | 226.382 | 733.938 |
| 4 | 241 | 2 | 9 | 0.921162 | 0.775124 | 0.660062 | 0.694316 | 285.293 | 737.141 |

Per-class E8 values are **precision / recall / F1**:

| Fold | Clipper | Grasper | Hook | Scissor |
| --- | --- | --- | --- | --- |
| 0 | .879/.944/.911 | .979/.940/.959 | .947/.989/.967 | .750/.562/.643 |
| 1 | .644/.906/.753 | .897/.872/.884 | .971/.953/.962 | .683/.632/.656 |
| 2 | .971/.985/.978 | .984/1.000/.992 | 1.000/.968/.984 | .846/.786/.815 |
| 3 | .880/.898/.889 | .915/.878/.896 | .860/.988/.920 | .966/.737/.836 |
| 4 | .961/.817/.883 | .901/.988/.943 | .957/.978/.967 | .333/.286/.308 |

E8 confusion matrices, **rows = actual folder labels; columns = predictions**:

| Fold | Matrix, four rows |
| --- | --- |
| 0 | `[[51,0,1,2],[4,94,1,1],[0,1,89,0],[3,1,3,9]]` |
| 1 | `[[29,1,1,1],[6,130,1,12],[0,3,202,7],[10,11,4,43]]` |
| 2 | `[[66,0,0,1],[0,62,0,0],[0,0,30,1],[2,1,0,11]]` |
| 3 | `[[44,3,1,1],[6,86,6,0],[0,1,80,0],[0,4,6,28]]` |
| 4 | `[[49,4,3,4],[1,82,0,0],[0,2,89,0],[1,3,1,2]]` |

**Pooled E8 OOF:** 1,402 samples, accuracy **0.910128**, macro-F1 **0.869483**. E7 pooled accuracy/macro-F1 were **0.804565 / 0.738166**; E6's were **0.817404 / 0.754132**. E8 gains **0.131317** macro-F1 versus E7 and **0.115351** versus E6. `comparison_to_e6_e7.json` selects **E8** by the declared primary metric. Pooled macro-F1 is computed from all OOF predictions, not averaged from fold F1 values.

| Pooled class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| clipper | 262 | 0.878676 | 0.912214 | 0.895131 |
| grasper | 492 | 0.928425 | 0.922764 | 0.925586 |
| hook | 505 | 0.945946 | 0.970297 | 0.957967 |
| scissor | 143 | 0.756098 | 0.650350 | 0.699248 |

Pooled E8 confusion matrix: `[[239,8,6,9],[17,454,8,13],[0,7,490,8],[16,20,14,93]]`.

Joining OOF CSVs by complete relative path found **271 E8-versus-E7 changed predictions** (folds 0–4: 32, 132, 15, 51, 41). Under the folder-label assumption, 193 changed from wrong to correct, 45 from correct to wrong, and 33 remained wrong. E8 versus E6 changed **281 predictions** (folds 0–4: 29, 121, 31, 54, 46): 191 wrong to correct, 61 correct to wrong, and 29 still wrong. The unchanged counts were 1,131 versus E7 and 1,121 versus E6.

## Artifacts and limits

- E7 bundled official pretrained weight SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. E2 manifest SHA-256: `d82c7984e1ebcbaaf85ba8ca1ec7561cb141ae40ad4cb379d13c661c88fcaf18`. Regenerable E8 activation cache SHA-256 for this run: `4a89a32e0b05b7ccb49d9c4e8412ff210aba44eab869db837676a21b9a4d1770`.
- E8 `oof_predictions.csv` SHA-256: `d734bd3c9fcef6f7d5fc85c9ac744393dd750404043bb8d544ecbee616ece2e9`. E8 `results.json` SHA-256: `a25d502a8ef9dfdfc999b7d91793506987f4e0b3f3b118c7debafd8cb7b362c9`. E8 `comparison_to_e6_e7.json` SHA-256: `b51736c4370a87ea76f762c015b35e464e0bad5cf6eb598710d2a151ec9c31df`.
- E8 checkpoint SHA-256 by fold: `0a0cb4e0cdcdc839ceea9bf4854fbb3c820f5abb040228a0f9203080611e4c19`, `4d9caf6142a48f7bcf0bf12748d326b5261ea7ace9e592a1797e4070f2055148`, `1f00be2e0628a603f9d14ede24f026cf433a216a4eb8001fcaf6c525e4513e00`, `cd5261b84bbb7cab3726a30ec633f66be70082905a032af6eb56e6d3007ea281`, `4799dda62ee506c73a1e2c2dadde9595bb5b8e93aa178b8de55cc6de08753ba9`.
- These are video-held-out folder-label diagnostics on ten filename-derived groups, not private-test results. Fold 4 has only seven scissor validation images; its scissor F1 is 0.308. The activation cache is not committed, so `--cache` must be rerun before training from a fresh checkout. The offline inference test reloaded each selected checkpoint and reproduced the first validation image of every fold. No claim of byte-identical logits across every batch size is made. Existing `predict.py` and its checkpoint remain unchanged.
