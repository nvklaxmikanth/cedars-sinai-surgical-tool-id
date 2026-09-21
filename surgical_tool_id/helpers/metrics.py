"""Small metrics helpers used by evaluate_model.py and the training loops."""
import torch


def batch_accuracy(logits, targets):
    """Accuracy for a single batch."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def epoch_accuracy(batch_accs):
    """Average accuracy across an epoch's batches."""
    return sum(batch_accs) / len(batch_accs)


def topk_confidence(logits):
    """Mean top-1 softmax confidence, used as a quick calibration sanity
    check when eyeballing predictions."""
    probs = torch.softmax(logits, dim=0)
    conf, _ = probs.max(dim=0)
    return conf.mean().item()
