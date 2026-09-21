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
