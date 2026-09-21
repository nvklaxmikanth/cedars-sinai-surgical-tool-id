"""Central config. Some of this has drifted from configs/default.yaml since
that file was set up for the EfficientNet experiment - this module is what
train.py actually imports.
"""

LEARNING_RATE = 3e-4
BATCH_SIZE = 32
NUM_EPOCHS = 15
NUM_CLASSES = 4
IMAGE_SIZE = 224
WEIGHT_DECAY = 1e-5
DATA_ROOT = "data/cholec-tinytools"
