"""E8 ResNet18: train layer 4 and head, with all BatchNorm state frozen."""

import hashlib
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn


E7_MODEL = Path(__file__).resolve().parents[1] / "e7_frozen_resnet18" / "model.py"
spec = importlib.util.spec_from_file_location("e7_resnet18_model_for_e8", E7_MODEL)
e7_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e7_model)

WEIGHTS_PATH = e7_model.WEIGHTS_PATH
WEIGHTS_SHA256 = e7_model.WEIGHTS_SHA256
preprocess_image = e7_model.preprocess_image


def load_partial_model(num_classes=4):
    """Load E7's bundled official weights and permit only layer 4/head gradients."""
    rng_state = torch.random.get_rng_state()
    model = e7_model.load_frozen_backbone(num_classes=num_classes)
    # E7 creates a fresh Linear head immediately after seed 42. Restore the
    # state consumed by backbone construction so E8 starts with that same head.
    torch.random.set_rng_state(rng_state)
    model.fc.reset_parameters()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("layer4.", "fc.")))
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    model.eval()
    return model


def forward_to_layer3(model, x):
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    x = model.layer1(x)
    x = model.layer2(x)
    return model.layer3(x)


def forward_from_layer3(model, x):
    x = model.layer4(x)
    return model.fc(torch.flatten(model.avgpool(x), 1))


def frozen_state_hash(model):
    """Hash every frozen parameter and buffer, including all BatchNorm state."""
    digest = hashlib.sha256()
    trainable_names = {name for name, parameter in model.named_parameters()
                       if parameter.requires_grad}
    for name, tensor in model.state_dict().items():
        if name in trainable_names:
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def trainable_state(model):
    trainable_names = {name for name, parameter in model.named_parameters()
                       if parameter.requires_grad}
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            if name in trainable_names}


def load_trainable_state(model, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected = set(trainable_state(model))
    if set(state) != expected:
        raise ValueError("checkpoint trainable tensor keys do not match layer 4 and head")
    model.load_state_dict(state, strict=False)
    model.eval()


def predict_image(path, checkpoint, class_names):
    """Reconstruct an E8 fold model offline from E7 weights and its E8 checkpoint."""
    model = load_partial_model(num_classes=len(class_names))
    load_trainable_state(model, checkpoint)
    with torch.no_grad():
        index = model(preprocess_image(path).unsqueeze(0)).argmax(dim=1).item()
    return index, class_names[index]
