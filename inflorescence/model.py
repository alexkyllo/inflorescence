"""An abstract interface class defining a wrapper for an ML model for federated training.
"""
from abc import ABC, abstractmethod
from typing import Tuple

from flwr.common.typing import Metrics, NDArrays, Scalar


class Model(ABC):
    def __init__(self, module):
        self.module = module

    @abstractmethod
    def fit(self, X_train, y_train, **kwargs):
        """"""

    @abstractmethod
    def evaluate(self, X_test, y_test) -> Tuple[Scalar, Metrics]:
        """Evaluate the model on a validation or test set."""

    @abstractmethod
    def get_weights(self) -> NDArrays:
        """"""

    @abstractmethod
    def set_weights(self, weights: NDArrays):
        """"""
