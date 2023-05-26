from inspect import isfunction
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from flwr.common.typing import Metrics, Scalar
from torch import nn, optim, sigmoid


def make_layers(*sizes, activation):
    """Build a MLP graph of linear layers based on given sizes per layer."""
    if len(sizes) < 2:
        raise ValueError(
            "A neural network must have at least two layer sizes, an input and output size."
        )
    layers = []
    layers.append(nn.Linear(sizes[0], sizes[1]))
    for i in range(1, len(sizes) - 1):
        layers.append(activation())
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
    return nn.ModuleList(layers)


class LogisticRegression(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, 1)

    def forward(self, x):
        y_pred = sigmoid(self.linear(x))
        return y_pred


class MLP(nn.Module):
    def __init__(self, n_inputs, hidden_units=10, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return sigmoid(x)


class CNN(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        n_channels: int = 3,
        conv1_kernels=10,
        conv2_kernels=20,
        kernel_size=5,
        # dropout=0.5,
        linear_hidden_size=50,
        num_classes=10,
    ):
        """CNN for use with the CelebA experiments."""
        super().__init__()
        conv_out_dim = (((input_shape[1] - 4) // 2) - 4) // 2
        self.linear_input_size = conv2_kernels * conv_out_dim**2
        self.conv1 = nn.Conv2d(n_channels, conv1_kernels, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(conv1_kernels, conv2_kernels, kernel_size=kernel_size)
        # self.conv2_drop = nn.Dropout2d(dropout)
        self.fc1 = nn.Linear(self.linear_input_size, linear_hidden_size)
        self.fc2 = nn.Linear(linear_hidden_size, num_classes)
        self.epoch = 0
        self.criterion = nn.NLLLoss()
        self.num_classes = num_classes

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        # x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, self.linear_input_size)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        if self.num_classes > 1:
            return F.log_softmax(x, dim=-1)
        else:
            return torch.sigmoid(x)


class TorchClassifier(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: Callable[..., Union[Scalar, torch.Tensor]],
        optimizer_class: Type[optim.Optimizer],
        metrics: Optional[Dict[str, Callable[..., Scalar]]] = None,
        group_metrics: Optional[Dict[str, Callable[..., Any]]] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        train_on_gpu: bool = False,
        test_on_gpu: bool = False,
        device_num: int = 0,
        **optimizer_kwargs,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.metrics = metrics or {}
        self.group_metrics = group_metrics or {}
        self.seed = seed
        if seed:
            torch.manual_seed(seed)
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.train_on_gpu = train_on_gpu
        self.test_on_gpu = test_on_gpu
        self.device_num = device_num

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        loss, metrics = self._shared_eval_step(batch, batch_idx, mode="train")
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._shared_eval_step(batch, batch_idx, mode="valid")
        metrics.update({"valid_loss": loss})
        self.log("valid_loss", loss, on_epoch=True, on_step=False)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, metrics = self._shared_eval_step(batch, batch_idx, mode="test")
        # metrics = {f"test_{k}": v for k, v in metrics.items()}
        metrics.update({"test_loss": loss})
        self.log("test_loss", loss, on_epoch=True, on_step=False)
        return metrics

    def calculate_metrics(self, metrics, pred, y, mode, group=None):
        metrics_dict = {}
        g = f"_group_{group:02}" if group is not None else ""
        for k, v in metrics.items():
            train_gpu = mode == "train" and self.train_on_gpu
            test_gpu = mode in ["valid", "test"] and self.test_on_gpu
            if (train_gpu or test_gpu) and not isfunction(v):
                v.to(torch.device("cuda", self.device_num))
            metric_value = v(pred, y)
            if repr(v) == "BinaryConfusionMatrix()":
                cm_dict = {
                    f"{mode}_{k}_tn{g}": metric_value[0][0].float(),
                    f"{mode}_{k}_fp{g}": metric_value[0][1].float(),
                    f"{mode}_{k}_fn{g}": metric_value[1][0].float(),
                    f"{mode}_{k}_tp{g}": metric_value[1][1].float(),
                }
                metrics_dict.update(cm_dict)
                self.log_dict(cm_dict, reduce_fx=torch.sum, on_epoch=True, on_step=False)
            elif repr(v) == "MulticlassConfusionMatrix()":
                # Confusion matrix returns a tensor, split into a dict
                cm_dict = {}
                for i, row in enumerate(metric_value):
                    for j, col in enumerate(row):
                        cm_dict[f"{mode}_{k}_{i:02}_{j:02}{g}"] = col.float()
                metrics_dict.update(cm_dict)
                self.log_dict(cm_dict, reduce_fx=torch.sum, on_epoch=True, on_step=False)
            else:
                key = f"{mode}_{k}{g}"
                metrics_dict[key] = metric_value
                self.log(key, metric_value, on_epoch=True, on_step=False)
        return metrics_dict

    def calculate_group_metrics(self, pred, y, g, mode):
        group_metrics = {}
        # groups = np.unique(g)
        groups = torch.unique(g)
        for group in groups:
            metrics_dict = self.calculate_metrics(
                self.group_metrics, pred=pred[g == group], y=y[g == group], mode=mode, group=group
            )
            # metrics_dict = {f"{k}_group_{i:02}": v for k, v in metrics_dict.items()}
            group_metrics.update(metrics_dict)
        return group_metrics

    def _shared_eval_step(self, batch, batch_idx, mode) -> Tuple[Scalar, Metrics]:
        x, labels = batch
        y_hat = self.model(x).squeeze(-1)
        pred = y_hat.gt(0.5) * 1
        metrics = {}
        if labels.dim() > 1:
            y = labels[:, 0]
            g = labels[:, 1:]
            if g.shape[1] == 1:
                g = g.squeeze()
            elif g.shape[1] > 1:
                g = torch.argmax(labels, dim=1)

            metrics_dict = self.calculate_metrics(self.metrics, pred, y, mode)
            metrics.update(metrics_dict)
            if self.group_metrics:
                group_metrics = self.calculate_group_metrics(pred, y, g, mode)
                metrics.update(group_metrics)
        else:
            y = labels
            metrics_dict = self.calculate_metrics(self.metrics, pred, y, mode)
            metrics.update(metrics_dict)
        loss = self.criterion(y_hat.float(), y.float())
        return loss, metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        return self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)
