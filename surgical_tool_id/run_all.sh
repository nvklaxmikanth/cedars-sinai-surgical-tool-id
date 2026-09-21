#!/usr/bin/env bash
# Kicks off the full pipeline: trains the current experiment, checks the
# production checkpoint's accuracy, and reminds you what config is live.
set -e

echo "=== training exp02 (efficientnet) ==="
python experiments/exp02_efficientnet/run_experiment.py

echo "=== evaluating production checkpoint ==="
python evaluate_model.py

echo "=== current config.py values ==="
python -c "import config; print('lr=', config.LEARNING_RATE); print('batch_size=', config.BATCH_SIZE); print('epochs=', config.NUM_EPOCHS)"
