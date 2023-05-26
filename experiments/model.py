from collections import OrderedDict
from typing import Tuple

import pytorch_lightning as pl
from flwr.common.typing import Metrics, NDArray, Scalar
from torch import from_numpy, tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.datasets import VisionDataset

from inflorescence.model import Model


class TorchModel(Model):
    def get_weights(self):
        """"""
        # return [val.cpu().numpy() for _, val in self.module.state_dict().items()]
        return [val.cpu().detach().numpy() for val in self.module.parameters()]

    def set_weights(self, weights):
        """"""
        params_dict = zip(self.module.state_dict().keys(), weights)
        state_dict = OrderedDict({k: tensor(v) for k, v in params_dict})
        self.module.load_state_dict(state_dict, strict=True)


class TorchTabularModel(TorchModel):
    def __init__(self, module):
        super().__init__(module)

    def fit(
        self,
        X_train: NDArray,
        y_train: NDArray,
        epochs: int,
        batch_size: int = 32,
        **kwargs,
    ) -> Tuple[Scalar, Metrics]:
        """Fit the model."""
        # TODO: configure callbacks for logging
        trainer = pl.Trainer(
            max_epochs=epochs, enable_progress_bar=False, enable_checkpointing=False, logger=False
        )  # , accelerator="gpu")
        train_loader = DataLoader(
            TensorDataset(
                from_numpy(X_train.astype("float32")),
                from_numpy(y_train.astype("float32")),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        trainer.fit(self.module, train_loader)
        metrics = trainer.logged_metrics
        loss = float(metrics.pop("train_loss"))
        return loss, metrics

    def evaluate(
        self, X_test: NDArray, y_test: NDArray, batch_size: int = 32
    ) -> Tuple[Scalar, Metrics]:
        """"""
        trainer = pl.Trainer(
            enable_progress_bar=False, enable_checkpointing=False, logger=False
        )  # accelerator="gpu")
        val_loader = DataLoader(
            TensorDataset(
                from_numpy(X_test.astype("float32")),
                from_numpy(y_test.astype("float32")),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        res = trainer.validate(self.module, val_loader)[0]
        print(res)
        loss = res.pop("valid_loss")
        return loss, res


class TorchVisionModel(TorchModel):
    def __init__(self, module, use_gpu: bool = False, device_num: int = 0):
        super().__init__(module)
        self.use_gpu = use_gpu
        self.accel = "gpu" if self.use_gpu else None
        self.device_num = device_num

    def fit(
        self, X_train: VisionDataset, y_train=None, *, epochs, batch_size=32, **kwargs
    ) -> Tuple[Scalar, Metrics]:
        """"""
        trainer = pl.Trainer(
            max_epochs=epochs,
            enable_progress_bar=False,
            enable_checkpointing=False,
            # logger=False,
            accelerator=self.accel,
            devices=[self.device_num] if self.use_gpu else None,
        )
        train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
        trainer.fit(self.module, train_loader)
        metrics = trainer.logged_metrics
        loss = float(metrics.pop("train_loss"))
        return loss, {}

    def evaluate(self, X_test, y_test, batch_size: int = 32) -> Tuple[Scalar, Metrics]:
        """"""
        trainer = pl.Trainer(
            enable_progress_bar=False,
            # accelerator=self.accel,
            enable_checkpointing=False,
            logger=False,
        )
        val_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False)
        res = trainer.validate(self.module, val_loader)[0]
        loss = res.pop("valid_loss")
        return loss, res

    def predict():
        """"""
        # TODO
