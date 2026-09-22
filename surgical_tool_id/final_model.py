"""Self-contained ResNet18 architecture and E8/E10 RGB preprocessing."""

import torch
import torch.nn as nn
from PIL import Image


MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        layers = [BasicBlock(self.inplanes, channels, stride, downsample)]
        self.inplanes = channels
        layers.extend(BasicBlock(channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return torch.flatten(self.avgpool(x), 1)

    def forward(self, x):
        return self.fc(self.forward_features(x))


def preprocess_image(path):
    """IMAGENET1K_V1 PIL resize-short-side-256, center-crop-224, normalize."""
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if width < height:
        new_size = (256, int(256 * height / width))
    else:
        new_size = (int(256 * width / height), 256)
    image = image.resize(new_size, Image.Resampling.BILINEAR)
    left = round((image.width - 224) / 2)
    top = round((image.height - 224) / 2)
    image = image.crop((left, top, left + 224, top + 224))
    tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).view(224, 224, 3)
    tensor = tensor.permute(2, 0, 1) / 255.0
    return (tensor - MEAN) / STD
