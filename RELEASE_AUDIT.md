# E12 sanitized public release audit

This repository was created as a separate filtered copy. The original assessment repository was not modified. E12 did not create a remote or push content.

## Public contents and exclusions

The history filter removed these classes from every rewritten revision:

- `surgical_tool_id/data/` and `surgical_tool_id/labels.csv`;
- JSON split manifests containing per-image paths or labels;
- OOF and aligned-probability CSVs;
- feature caches and cache metadata;
- every `.pt` and `.pth` historical model artifact;
- supplied `.DS_Store` metadata.

E12 then added only the three required E10 full-state checkpoints. Source code, documentation, aggregate metric JSON, per-fold metric explanations, and non-filename-level experiment records remain. `.gitignore` prevents the private dataset, generated manifests, OOF files, caches, and extra model binaries from being added accidentally.

The dataset is not distributed. An authorized user must place it outside Git or under the ignored local `surgical_tool_id/data/` tree and run [`video_folds.py`](surgical_tool_id/splits/video_folds.py) with `--data-root` and `--out` to regenerate the ignored split manifest. No open source license was added because no license terms were provided.

## Rewritten commit map

| Original | Sanitized | Subject |
| --- | --- | --- |
| `b41f1b71bb8234581223669b2b0755f1133336d4` | `b0c8afea9a4ea84d9cc7aaacc8e4288102af6fcc` | Baseline: untouched supplied repository |
| `eae2df9590cb2df185ee237935e22d4138031970` | `926ebc4864f9e0f77866ca2af997dcfcae99d5d4` | E0 inference decoupling |
| `10c573998a6e5c3bc850471f422b4d2735d51840` | `ef0635323c8ad4472447c37c3ba393529b2cadd9` | E1 class decoding |
| `d05a1dc82f19d192dacfe8f6cf5eb9691eb2a1af` | `37f9d091e50d7a7806e8b1fcdb0e59c051b7138e` | E2 grouped splits |
| `533122dcf23b40da148ffbfbb091506018394ef0` | `002b0bb3a3b50625df690ee6b67a30af1262efdd` | E3 legacy baseline |
| `dbb37f20ae13d6f0465c6aca38036b65847e2280` | `dd45f7f77ddbceb47afadb137439cc36ce43801c` | E4 cross entropy |
| `d2f7409bcbab0a70db1ec2e538cf1ca9d126f1e8` | `bec026b2d4269af073fdd2b96e682f2e1b446be1` | E5 weighted cross entropy |
| `a52a160af9821da9c5ccb7601d193b5554d6cb4d` | `4568a042df7e6de7548160cce9c03b4d8f70f288` | E6 RGB input |
| `c0ca9836bb5ee615e9e7612b4dac9ef53be3294c` | `c88cd031880314be546839996b066c06001faca2` | E7 frozen ResNet18 |
| `250f21e22a0a81e7e0ec9b53ec05ef43497e87f6` | `66c05a989c70c92f373dcaad74a5e9c52b9ba009` | E8 partial fine tuning |
| `b60c0804a0b5c8241e7b43df022240f305311707` | `91b2d73678b206ddfe9eb7a9342749b129c8bdaa` | E9 seed ensemble |
| `6a26422c2014349b1adc39da98d60a88fae5146a` | `c050c43db06ba72956ed33df4099bfe162848ee6` | E10 deployment |
| `e5b68d2d9ce796b2ea9f184c56c6aca51a3b69f6` | `cfdf2423545b7667f37941a14c4169b687d235ab` | E11 documentation audit |

## Verification evidence

| Check | Measured result |
| --- | --- |
| Public test suite | 49 tests discovered in 3.534 s: **25 passed, 24 skipped**, zero failures. Skips name the required private dataset or removed historical artifact. |
| Documentation links | 41 repository-local Markdown links checked; zero missing. |
| Reachable Git objects after reference expiry and aggressive GC | 232 reachable objects, including 148 blobs totaling 135,527,399 uncompressed bytes. Only `refs/heads/main` remained. |
| Original dataset object comparison | Zero of the 1,402 original dataset image blob IDs remained reachable. |
| Reachable path scan | Zero dataset, `labels.csv`, split-manifest JSON, OOF, probability, or cache paths. Exactly three model paths remained: the E10 seeds 17, 42, and 123. |
| Reachable content scan | Zero image-extension paths, image-magic blobs, or occurrences of any original dataset basename. |
| Reachable secret scan | Zero matches for AWS access-key IDs, common GitHub/OpenAI token prefixes, PEM private-key headers, and simple password/API-key/access-token assignments. This is a bounded pattern scan, not a guarantee that no sensitive semantic content exists. |
| Checkpoint SHA-256 | Seed 17 `9dd2105b617fccba565a678ec54f40cac6006a2317075edd65e8379a036dc3bf`; seed 42 `97cd6548719638a4a46d53a67d2d6cbff72b16a3bb60b840c0edcd9269eb46d4`; seed 123 `fb3e313f0faf99d701f45cb222d53c615f2dd534aededc0d454583f78006cef5`. All matched the deployment manifest. |
| Post-GC size | Working tree approximately 248.2 MiB; `.git` approximately 119.2 MiB; packed Git data approximately 119.1 MiB. |

Fresh-clone offline inference and final commit checks were run after creating the final E12 commit and are reported with the release handoff.

## Performance interpretation and remaining limitations

E9's **0.888744 pooled grouped OOF macro-F1** is cross-validation performance under the private folder-label assumption. It is not private held-out performance. The OOF rows are excluded from the public repository; aggregate metrics and experiment explanations remain for audit context.

The public checkout supports offline inference with the three E10 checkpoints. It cannot reproduce private-data experiments without the authorized dataset, regenerated manifest, and removed pretrained/historical training assets. Private test methods are retained but skipped when those artifacts are absent. CPU inference is verified; CUDA execution remains unmeasured. Dataset provenance, label disagreements, external validation, clinical suitability, and checkpoint redistribution rights remain unresolved. The repository has no license grant.
