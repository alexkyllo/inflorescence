"""IFCA client implementation for Flower.
Ghosh et al, An Efficient Framework for Clustered Federated Learning, 2020.
https://arxiv.org/pdf/2006.04088.pdf
https://github.com/jichan3751/ifca
"""

from typing import Callable, Dict, Tuple

import numpy as np
from flwr.client.numpy_client import NumPyClient
from flwr.common.typing import Config, NDArray, NDArrays, Scalar

from ..model import Model


class IFCAClient(NumPyClient):
    """A client designed to work with the Iterative Federated Clustering Algorithm
    (IFCA) strategy from Ghosh et al 2020 (model averaging version).

    Initialize k models with different random seeds
    Broadcast all k models to random subset of n clients
    Each client computes loss on each model and picks model w/ argmin loss
    and then trains that model for t steps
    Client sends updated model and cluster identity to server
    Server averages together updated models from clients within same cluster
    """

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
        self.model_fn = model_fn  # TODO: how to pass random seed for initialization?
        self.models = []
        self.k_clusters = 1
        self.cluster_num = 0
        self.cid = cid
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_args = model_args
        self.model_kwargs = model_kwargs

    def get_parameters(self, config: Config) -> NDArrays:
        if len(self.models) == 0:
            self.models = [
                self.model_fn(*self.model_args, **self.model_kwargs) for _ in range(self.k_clusters)
            ]
        return self.models[self.cluster_num].get_weights()

    def set_weights(self, parameters: NDArrays):
        """Set the client's model weights to the provided parameters."""
        self.models = [self.model_fn(*self.model_args, **self.model_kwargs)]  # set first model
        len_weights = len(self.models[0].get_weights())  # check how many layers it has
        self.k_clusters = len(parameters) // len_weights
        self.models[0].set_weights(parameters[0:len_weights])
        for i in range(1, self.k_clusters):
            j = i * len_weights
            model = self.model_fn(*self.model_args, **self.model_kwargs)
            model.set_weights(parameters[j : j + len_weights])
            self.models.append(model)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        _, _, metrics = self.evaluate(parameters=parameters, config=config)
        j = metrics["cluster"]
        self.models[int(j)].fit(
            self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=2
        )
        return self.get_parameters(config), len(self.x_train), {"cluster": j}

    def evaluate(
        self, parameters: NDArrays, config: Config
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_weights(parameters)
        losses = []
        metrics = []
        for i, model in enumerate(self.models):
            loss, metric_dict = model.evaluate(self.x_test, self.y_test)
            losses.append(loss)
            metric_dict.update({"cluster": i})
            metrics.append(metric_dict)
        self.cluster_num = np.argmin(losses)
        return (losses[self.cluster_num], len(self.x_test), metrics[self.cluster_num])
