# E11 release audit

Audit scope: local `main` through E10 commit `6a26422c2014349b1adc39da98d60a88fae5146a`, with E11 documentation added separately. No remote was configured or created. No files were removed, models retrained, predictions changed, or content pushed. This is a release inventory and verification record, **not an authorization to publish**.

## Git and tracked-file inventory

| Check | Measured finding |
| --- | --- |
| `git log --oneline --all` | Eleven commits: untouched baseline, then E0–E10 as separate commits. |
| `git ls-files` and file sizes | 1,613 tracked files before E11 documentation, 787,395,331 working-tree bytes. 1,402 PNGs total 28,990,177 bytes; 47 `.pt` files total 708,650,442 bytes; one `.pth` is 46,830,571 bytes; 51 CSVs total 1,812,369 bytes; 62 JSONs total 808,260 bytes; one notebook. |
| Historical objects via `git rev-list --objects --all` and `git cat-file --batch-check` | 1,628 distinct history blobs; largest blob 46,830,571 bytes. No current tracked file or historical blob exceeded 50 MiB or 100 MiB. The `.git` directory occupied about 704 MiB (`du -sh .git`) at audit time. |
| Dataset, caches, and outputs | All 1,402 source PNGs and `labels.csv` are tracked. One E7 `feature_cache.pt` and E7/E8 cache metadata are tracked. The larger E8 layer-3 cache is ignored locally. At least 149 experiment CSV/JSON/PT artifacts are tracked, including E9 fold checkpoints and E10 full-state checkpoints. The supplied baseline also tracked older checkpoints and `.DS_Store`. |
| Environments and remotes | No tracked `.env`, virtualenv, `__pycache__`, or `.pyc` files were found. `git remote -v` produced no configured remote. |
| Pattern-based secret scan | History-wide `git grep -I -E -l` found no matches for AWS access-key IDs, common GitHub/OpenAI token prefixes, PEM private-key headers, or simple `password`, `api_key`, and `access_token` assignments. This limited pattern scan cannot rule out all secrets, binary payloads, or sensitive imagery. |

[GitHub's current documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) says regular Git files over 50 MiB trigger a warning and files over 100 MiB are blocked. [GitHub repository guidance](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits) recommends small objects and using Git LFS or external storage for large assets. The measured largest file is below those upload thresholds, but the 704 MiB local Git history and many 33–47 MB checkpoints are material clone-size and maintenance costs. A future release may need a separately reviewed lightweight distribution or LFS migration; rewriting this history was outside E11.

## Licensing and provenance

No `LICENSE`, `COPYING`, or `NOTICE` file is tracked. The repository does not document the source license or redistribution authorization for the 1,402 surgical images or `labels.csv`. That missing authorization is a **release blocker for public distribution** until the dataset owner confirms permitted use and privacy handling. No external provenance check can establish those permissions from the local files alone.

The E7 bundled pretrained file matches SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. [TorchVision's official ResNet18 source](https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html) identifies the matching `IMAGENET1K_V1` weight URL, and [TorchVision identifies its code license as BSD-3-Clause](https://github.com/pytorch/vision/blob/main/CITATION.cff). This supports provenance of the source code and URL; it does **not by itself resolve** redistribution terms for the bundled weight file, ImageNet-trained assets, or this repository's copied/custom architecture. A human maintainer must confirm applicable notices and permissions before publication.

## Verification and interpretation

The release checks compared checkpoint bytes to [`checkpoints/e10/manifest.json`](surgical_tool_id/checkpoints/e10/manifest.json), ran the full tests, checked local Markdown links, and ran the CLI from a fresh isolated Git clone. Deployment requires only [`predict.py`](surgical_tool_id/predict.py), [`final_model.py`](surgical_tool_id/final_model.py), [`requirements-inference.txt`](surgical_tool_id/requirements-inference.txt), and the E10 checkpoint bundle. The isolated clone used already installed PyTorch/Pillow; package-index installation and CUDA execution were not verified.

| Command/check | Measured result |
| --- | --- |
| Parse Markdown links in the six E11 documents and check non-HTTP paths against the repository | **45 local links checked; zero missing.** |
| `shasum -a 256 surgical_tool_id/checkpoints/e10/seed_17.pt surgical_tool_id/checkpoints/e10/seed_42.pt surgical_tool_id/checkpoints/e10/seed_123.pt` | All three hashes matched the package manifest: seed 17 `9dd2105b617fccba565a678ec54f40cac6006a2317075edd65e8379a036dc3bf`, seed 42 `97cd6548719638a4a46d53a67d2d6cbff72b16a3bb60b840c0edcd9269eb46d4`, seed 123 `fb3e313f0faf99d701f45cb222d53c615f2dd534aededc0d454583f78006cef5`. |
| `git clone --no-local --quiet . /tmp/e11_release_clone` from committed E10, then `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v` in the clone | Fresh clone contained **no ignored E8 feature cache**. All **49 tests passed in 39.594 s**. |
| `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python predict.py --data-dir data/cholec-tinytools/validation --out /tmp/e11_clone_predictions.csv` in the clone | CLI exited 0 and wrote **277 rows**. `validate_schema` returned `{'ok': True, 'reason': None, 'n_rows': 277}`. `cmp` found the clone CSV byte identical to the E10 reference; both have SHA-256 `b8e7214847d6b4f67b369d9b7a5638fae90da4bea9efb2778f56e1be1481d46a`. No labels were scored. |

E3 pooled video-grouped OOF macro-F1 was **0.616462**; E9 ensemble pooled video-grouped OOF macro-F1 was **0.888744** under folder labels. **0.888744 is grouped cross-validation performance, not private held-out performance.** The E10 fits used all 1,402 images and have no independent accuracy or F1 estimate.

## Remaining manual release steps

1. Obtain written dataset and label redistribution/privacy authorization; inspect the imagery and metadata for sensitive content before public distribution.
2. Select and document a repository license and verify upstream model-weight and code notices with the applicable rights holders.
3. Decide whether the full dataset and historical fold checkpoints belong in the public repository; if a smaller release is needed, review its artifact manifest and history strategy before changing Git history.
4. Install the pinned inference requirements in a truly fresh CPU environment and a supported CUDA environment; run the documented CLI and compare the output contract on each.
5. Obtain an independent, source-verified private or external video holdout before making a generalization or clinical performance claim. Review folder/CSV label conflicts and the filename-to-video mapping.
