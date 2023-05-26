import numpy as np
from torch import tensor
from torch.nn import functional as F
from torch.optim import SGD
from torchmetrics.functional.classification.accuracy import binary_accuracy

from experiments.model import TorchTabularModel
from experiments.torch_classifier import MLP, TorchClassifier
from inflorescence.metrics import (
    binary_confusion_matrix,
    disparate_impact_ratio,
    equal_odds_ratio,
    false_negatives,
    false_positives,
    group_binary_confusion_matrix,
    positive_predictive_parity_ratio,
    true_negatives,
    true_positives,
)


def test_metrics():
    module = TorchClassifier(
        model=MLP(3),
        criterion=F.binary_cross_entropy,
        metrics={"accuracy": binary_accuracy},
        group_metrics={"accuracy": binary_accuracy},
        optimizer_class=SGD,
        lr=0.001,
    )
    x = tensor(
        [
            [0.13, 0.07, 0.01],
            [0.14, 0.98, 0.02],
            [0.92, 0.05, 0.04],
            [0.11, 0.47, 0.01],
            [0.73, 0.04, 0.63],
        ]
    )
    y = tensor([[0, 1, 0, 1, 0], [0, 0, 0, 1, 1]]).T
    result = module._shared_eval_step((x, y), 0, mode="train")
    assert "train_accuracy" in result[1]
    assert "train_accuracy_group_01" in result[1]


def test_metrics_validation():
    module = TorchClassifier(
        model=MLP(3),
        criterion=F.binary_cross_entropy,
        metrics={"accuracy": binary_accuracy},
        group_metrics={"accuracy": binary_accuracy},
        optimizer_class=SGD,
        lr=0.001,
    )
    x = tensor(
        [
            [0.13, 0.07, 0.01],
            [0.14, 0.98, 0.02],
            [0.92, 0.05, 0.04],
            [0.11, 0.47, 0.01],
            [0.73, 0.04, 0.63],
        ]
    )
    y = tensor([[0, 1, 0, 1, 0], [0, 0, 0, 1, 1]]).T
    result = module.validation_step((x, y), 0)
    assert "valid_accuracy" in result
    assert "valid_accuracy_group_01" in result


def test_torch_tabular_evaluate():
    module = TorchClassifier(
        model=MLP(3),
        criterion=F.binary_cross_entropy,
        metrics={"accuracy": binary_accuracy},
        group_metrics={"accuracy": binary_accuracy},
        optimizer_class=SGD,
        lr=0.001,
    )
    model = TorchTabularModel(module=module)
    x = np.array(
        [
            [0.13, 0.07, 0.01],
            [0.14, 0.98, 0.02],
            [0.92, 0.05, 0.04],
            [0.11, 0.47, 0.01],
            [0.73, 0.04, 0.63],
        ]
    )
    y = np.array([[0, 1, 0, 1, 0], [0, 0, 0, 1, 1]]).T
    result = model.evaluate(x, y, batch_size=5)
    assert "valid_accuracy" in result[1]
    assert "valid_accuracy_group_01" in result[1]


def test_torch_tabular_evaluate_cm():
    module = TorchClassifier(
        model=MLP(3),
        criterion=F.binary_cross_entropy,
        metrics={
            "tn": true_negatives,
            "fp": false_positives,
            "fn": false_negatives,
            "tp": true_positives,
        },
        optimizer_class=SGD,
        lr=0.001,
    )
    model = TorchTabularModel(module=module)
    x = np.array(
        [
            [0.13, 0.07, 0.01],
            [0.14, 0.98, 0.02],
            [0.92, 0.05, 0.04],
            [0.11, 0.47, 0.01],
            [0.73, 0.04, 0.63],
        ]
    )
    y = np.array([[0, 1, 0, 1, 0], [0, 0, 0, 1, 1]]).T
    loss, metrics = model.evaluate(x, y, batch_size=5)
    assert "valid_tn" in metrics
    assert "valid_fp" in metrics
    assert "valid_fn" in metrics
    assert "valid_tp" in metrics
