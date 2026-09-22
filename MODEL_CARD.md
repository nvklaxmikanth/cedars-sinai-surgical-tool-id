# Model card: E10 surgical tool ensemble

## Model and intended use

The E10 package classifies a PNG frame into one of four labels in this exact output order: **clipper, grasper, hook, scissor**. It is a research assessment on the supplied Cholec Tinytools subset, not a clinical device or a validated surgical decision aid. It does not identify multiple simultaneous tools, localization, procedure stage, or patient outcome. Deployment is through [`predict.py`](surgical_tool_id/predict.py) and three full-state checkpoints in [`checkpoints/e10/`](surgical_tool_id/checkpoints/e10/manifest.json).

Each model is a ResNet18 initialized from the bundled official TorchVision `IMAGENET1K_V1` weights. During training, layers through layer 3 and every BatchNorm parameter and running state stayed frozen; layer 4 and a new four-output linear head were trained. At inference each model produces logits, softmax is applied per model, and the three probability vectors are averaged equally before argmax. No class threshold or abstention rule was calibrated.

## Input and preprocessing

Images are opened with Pillow and converted to true RGB. The shorter side is resized to 256 pixels with bilinear interpolation, the center 224×224 crop is taken, pixels are divided by 255, and channels are normalized with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. This is implemented in [`final_model.py`](surgical_tool_id/final_model.py). The output CSV has columns `filename,predicted_class`; paths with duplicate basenames retain their relative paths.

## Training and evaluation evidence

The E2 split manifest covers 1,402 images in ten filename-derived video groups. Each of five validation folds holds out entire groups, with zero train/validation video-ID overlap and exactly one OOF appearance per image. The group ID is assumed to be the first filename number; it was not independently checked against source video metadata. Folder names supply the evaluation labels, although the supplied CSV and folders disagree for some files.

| Experiment | Change | Pooled grouped OOF macro-F1 |
| --- | --- | ---: |
| E3 | Legacy grayscale CNN, softmax plus one-hot MSE | 0.616462 |
| E4 | Raw-logit cross entropy | 0.713121 |
| E5 | Training-fold inverse-frequency weights | 0.728552 |
| E6 | True RGB input | 0.754132 |
| E7 | Frozen pretrained ResNet18 head | 0.738166 |
| E8 | Partial ResNet18 fine tuning | 0.869483 |
| E9 | Equal-weight OOF ensemble of seeds 17, 42, 123 | **0.888744** |

E9 pooled grouped OOF accuracy was **0.923680**. Its individual seed macro-F1 values were 0.883778, 0.869483, and 0.868940; the ensemble was selected because 0.888744 exceeded the best individual value. Selection and reported OOF evaluation used the same folds and labels, so the E9 score may be optimistic after experiment selection. **0.888744 is grouped cross-validation performance, not private held-out performance.** No private-test score is available. See [`E9 results.json`](surgical_tool_id/experiments/e9_seed_ensemble/results.json) and the [experiment log](surgical_tool_id/EXPERIMENT_LOG.md).

E10 trained the three selected seeds on all 1,402 frames for nine complete epochs, derived from the median zero-based E9 selected epoch of 8. It used batch size 2, weighted cross entropy, Adam with layer-4/head learning rates 0.0001/0.001, and no augmentation. E10 has **no training-set accuracy or F1 report** and no independent validation set. A CPU output-contract run on 277 images took 53.620 s and peaked at 749.047 MiB RSS on the measured host; CUDA execution was not measured. Individual E10 fits took 278.953, 279.133, and 280.505 s with 740.016, 780.281, and 776.703 MiB peak RSS respectively.

## Integrity and limitations

| Seed | Full-state checkpoint SHA-256 |
| ---: | --- |
| 17 | `9dd2105b617fccba565a678ec54f40cac6006a2317075edd65e8379a036dc3bf` |
| 42 | `97cd6548719638a4a46d53a67d2d6cbff72b16a3bb60b840c0edcd9269eb46d4` |
| 123 | `fb3e313f0faf99d701f45cb222d53c615f2dd534aededc0d454583f78006cef5` |

The deployed loader checks these hashes against the [package manifest](surgical_tool_id/checkpoints/e10/manifest.json). Expected behavior outside this PNG corpus, across hospitals/procedures, on other image formats, and on CUDA hardware is unmeasured. Temporal correlation remains possible within video groups, and the ten-group sample is small. Folder/CSV label disagreements, source-video assumptions, dataset provenance, and asset redistribution rights remain unresolved; see [CODE_REVIEW.md](CODE_REVIEW.md) and [RELEASE_AUDIT.md](RELEASE_AUDIT.md).
