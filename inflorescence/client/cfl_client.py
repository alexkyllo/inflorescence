"""CFL client implementation for Flower."""

from typing import Callable, Dict, Tuple

from flwr.client.numpy_client import NumPyClient
from flwr.common.typing import NDArray, NDArrays, Scalar

from ..model import Model


class CFLClient(NumPyClient):
    """A client designed to work with Clustered FL where each cluster has its
    own parameters."""

    def __init__(
        self,
        model_fn: Callable[[], Model],
        cid: int,
        x_train: NDArray,
        y_train: NDArray,
        x_test: NDArray,
        y_test: NDArray,
        epochs: int,
        batch_size: int,
        *model_args,
        **model_kwargs,
    ):
        self.model_fn = model_fn
        self.model = None
        self.k_clusters = 1
        self.cid = cid
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_args = model_args
        self.model_kwargs = model_kwargs

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        # Flatten the nested array of weights per model since ragged object ndarrays are not allowed.
        if self.model is None:
            self.model = self.model_fn(*self.model_args, **self.model_kwargs)
        return self.model.get_weights()

    def set_weights(self, parameters: NDArrays, cluster_num: int):
        """Set the client's model weights to the provided parameters."""
        # Server parameters contain one model per cluster.
        self.model = self.model_fn(*self.model_args, **self.model_kwargs)  # create a model
        len_weights = len(self.model.get_weights())  # check how many layers it has
        self.k_clusters = len(parameters) // len_weights  # check how many clusters there are
        start = cluster_num * len_weights  # get index where this model's params start
        self.model.set_weights(parameters[start : start + len_weights])

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        # Server sent FitIns with this client's cluster assignment.
        cluster_num = config["cluster"]
        self.set_weights(parameters, cluster_num)
        self.model.fit(
            self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=2
        )
        # Need to send back this client's cluster number because the server doesn't remember it
        return self.get_parameters(config), len(self.x_train), {"cluster": cluster_num}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        cluster_num = config["cluster"]
        self.set_weights(parameters, cluster_num)
        loss, metric_dict = self.model.evaluate(self.x_test, self.y_test)
        metric_dict.update({"cluster": cluster_num})
        return loss, len(self.x_test), metric_dict
