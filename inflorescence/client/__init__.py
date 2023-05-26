from typing import Callable, Type

from flwr.client import NumPyClient

from inflorescence.dataset import Dataset
from inflorescence.model import Model

from .cfl_client import CFLClient
from .flower_client import FlowerClient
from .ifca_client import IFCAClient


def client_factory(
    client_class: Type[NumPyClient],
    model_class: Type[Model],
    dataset: Dataset,
    *args,
    epochs: int,
    batch_size: int,
    **kwargs,
) -> Callable:
    """Function that returns a function to generate client instances"""

    def client_fn(cid: str):
        """"""
        train, test = dataset.get_split(cid)  # TODO: implement this method
        return client_class(
            model_class(*args, **kwargs),
            cid,
            train,
            test,
            epochs=epochs,
            batch_size=batch_size,
        )

    return client_fn
