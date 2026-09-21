import torch.nn as nn
from torchvision import models


def build_efficientnet(num_classes):
    """EfficientNet-B0 backbone, ImageNet-pretrained, with a fresh
    classifier head for our 4 tool classes."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
