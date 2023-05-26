from typing import Callable

from flwr.client import NumPyClient
from flwr.common.typing import Config, NDArray, NDArrays
from loguru import logger

from inflorescence.model import Model


class FlowerClient(NumPyClient):
    """A minimal Flower NumPy client implementation."""

    def __init__(
        self,
        model_fn: Callable[[], Model],
        cid: str,
        x_train: NDArray,
        y_train: NDArray,
        x_test: NDArray,
        y_test: NDArray,
        epochs: int,
        batch_size: int,
        *model_args,
        **model_kwargs,
    ):
        self.model = model_fn(*model_args, **model_kwargs)
        self.cid = cid
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = batch_size

    def get_parameters(self, config: Config):
        return self.model.get_weights()

    def fit(self, parameters: NDArrays, config):
        logger.debug("fit(): {}", self.cid)
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=2,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters: NDArrays, config: Config):
        logger.debug("evaluate(): {}", self.cid)
        self.model.set_weights(parameters)
        loss, metrics = self.model.evaluate(self.x_test, self.y_test)
        logger.debug(
            "evaluate() cid {}: loss: {}, num_examples: {}",
            self.cid,
            loss,
            len(self.x_test),
        )
        return loss, len(self.x_test), metrics
