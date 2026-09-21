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
