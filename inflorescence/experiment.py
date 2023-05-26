"""Experiment run configuration."""
import re
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from flwr.common.typing import Metrics
from flwr.server import ServerConfig
from flwr.server.history import History
from flwr.server.strategy import Strategy
from loguru import logger

from inflorescence.simulation import start_simulation

from .client import FlowerClient
from .dataset import Dataset
from .model import Model


class Experiment:
    """
    A class for running federated learning simulation experiments.

    Parameters
    ----------
    strategy : Strategy
        The Flower strategy to use for the experiment.
    model_fn : Callable[..., Model]
        A function that returns a model.
    dataset : Dataset
        The dataset to use for the experiment.
    group_var : Optional[str], optional
        The name of the variable in the dataset to calculate group fairness by, by default None.
    metrics : Optional[Dict[str, Callable[..., float]]], optional
        A dictionary of metric names and functions that calculate them,
        by default None.
    group_metrics : Optional[Dict[str, Callable[..., float]]], optional
        A dictionary of group metric names and functions that calculate them,
        grouped by the group_var, by default None.
    concentrations : Optional[List[float]], optional
        A list of Dirichlet concentration parameters in range [0,inf] to use for the experiment,
        by default None.
    test_size : float, optional
        The proportion of the dataset to use for testing, by default 0.2.

    Returns
    -------
    Experiment
        An instance of the Experiment class.

    Examples
    --------
    >>> from inflorescence import Experiment, Model, Dataset
    >>> from inflorescence.strategy import IFCA

    >>> def model_fn():
            return Model()

    >>> dataset = Dataset()

    >>> strategy = IFCA(k_clusters=5)

    >>> experiment = Experiment(
            strategy=strategy
            model_fn=model_fn,
            dataset=dataset,
            group_var='group',
            metrics={'accuracy': lambda y_true, y_pred: (y_true == y_pred).mean()},
            group_metrics={'accuracy': lambda y_true, y_pred: (y_true == y_pred).mean()},
            concentrations=[0.1, 1.0, 10.0],
            test_size=0.3,
        )
    """

    def __init__(
        self,
        strategy: Strategy,
        model_fn: Callable[..., Model],
        dataset: Dataset,
        group_var: Optional[str] = None,
        metrics: Optional[Dict[str, Callable[..., float]]] = None,
        group_metrics: Optional[Dict[str, Callable[..., float]]] = None,
        concentrations: Optional[List[float]] = None,
        test_size: float = 0.2,
    ):
        """A federated learning experiment setup."""
        self.model_fn = model_fn
        self.dataset = dataset
        self.strategy = strategy
        self.trials = []
        self.metrics = metrics or {}
        self.group_metrics = group_metrics or {}
        self.group_var = group_var
        self.concentrations = concentrations or [np.inf]  # if None, then IID. np.inf = sentinel val
        self.test_size = test_size

    def run(
        self,
        num_rounds: int,
        num_clients: int,
        epochs: int,
        batch_size: int,
        seed: Optional[Union[int, np.random.Generator]] = None,
        num_cpus: Union[int, float] = 1,
        num_gpus: Union[int, float] = 0,
        *model_args,
        **model_kwargs,
    ):
        """
        Run a federated learning simulation experiment.

        Parameters
        ----------
        num_rounds : int
            Number of communication rounds.
        num_clients : int
            Number of clients.
        epochs : int
            Number of local training epochs.
        batch_size : int
            Batch size for local training.
        seed : Optional[Union[int, np.random.Generator]], optional (default=None)
            Random seed for reproducibility.
        num_cpus : Union[int, float], optional (default=1)
            Number of CPUs to use for each client.
        num_gpus : Union[int, float], optional (default=0)
            Number of GPUs to use for each client.
        *model_args :
            Positional arguments for the model.
        **model_kwargs :
            Keyword arguments for the model.

        Returns
        -------
        self
            The experiment instance with trials list containing the results.
        """
        """Simulation."""
        for i, concentration in enumerate(self.concentrations):
            if concentration == np.inf:
                self.dataset.split_iid(
                    num_clients=num_clients,
                    group_var=self.group_var,
                    test_size=self.test_size,
                    seed=seed,
                )
            else:
                self.dataset.split_non_iid(
                    group_var=self.group_var,
                    num_clients=num_clients,
                    concentration=concentration,
                    test_size=self.test_size,
                    seed=seed,
                )
            logger.info(
                "Starting trial # {} with strategy {}, concentration {}",
                i,
                str(self.strategy),
                concentration,
            )
            trial = Trial(
                model_fn=self.model_fn,
                strategy=self.strategy,
                dataset=self.dataset,
                group_var=self.group_var,
                metrics=self.metrics,
                group_metrics=self.group_metrics,
                number=i,
                concentration=concentration,
            )
            self.trials.append(trial)
            model_kwargs["seed"] = seed
            trial.run(
                num_rounds=num_rounds,
                epochs=epochs,
                batch_size=batch_size,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                *model_args,
                **model_kwargs,
            )
            logger.info(
                "Finished trial # {} with strategy {}, concentration {}",
                i,
                str(self.strategy),
                concentration,
            )

        return self

    def to_pandas(self):
        """Get experiment trial results as a pandas DataFrame."""
        return (
            pd.concat([t.to_pandas() for t in self.trials])
            if self.trials
            else dataframe_from_history(None)
        )


class Trial:
    """An experiment run with a single concentration."""

    def __init__(
        self,
        model_fn: Callable[..., Model],
        strategy: Strategy,
        dataset: Dataset,
        group_var: Optional[str] = None,
        metrics: Optional[Dict[str, Callable[..., float]]] = None,
        group_metrics: Optional[Dict[str, Callable[..., float]]] = None,
        number: Optional[int] = 1,
        concentration: Optional[Union[float, List[float]]] = None,
    ):
        """A single trial within a federated learning experiment."""
        self.model_fn = model_fn
        self.dataset = dataset
        self.group_var = group_var
        self.strategy = strategy
        self.metrics = metrics or {}
        self.group_metrics = group_metrics or {}
        self.history = None
        self.number = number
        self.concentration = concentration

    def run(
        self,
        num_rounds: int,
        epochs: int,
        batch_size: int,
        num_cpus: Union[int, float] = 1,
        num_gpus: Union[int, float] = 0,
        *model_args,
        **model_kwargs,
    ):
        def client_fn(cid: str) -> FlowerClient:
            try:
                # check if the strategy has a special strategy-specific client class
                client_class: Type[FlowerClient] = self.strategy.client_class
            except AttributeError:
                client_class = FlowerClient
            ((x_train, y_train), (x_test, y_test)) = self.dataset.get_split(int(cid))

            model_kwargs.update(dict(metrics=self.metrics))
            model_kwargs.update(dict(group_metrics=self.group_metrics))
            return client_class(
                model_fn=self.model_fn,
                cid=cid,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                epochs=epochs,
                batch_size=batch_size,
                *model_args,
                **model_kwargs,
            )

        client_resources = {"num_cpus": num_cpus, "num_gpus": num_gpus}

        logger.info("Starting FL simulation with {}", client_resources)
        self.history = start_simulation(
            client_fn=client_fn,
            num_clients=len(self.dataset.splits),
            config=ServerConfig(num_rounds=num_rounds),
            strategy=self.strategy,
            client_resources=client_resources,
        )
        return self

    def to_pandas(self):
        """Return a Pandas DataFrame containing the aggregated metrics at each round."""
        df = dataframe_from_history(
            history=self.history,
            dataset=self.dataset,
            strategy=self.strategy,
            trial=self.number,
            concentration=self.concentration,
        )
        return df


def parse_group_metric_name(metric_name: str) -> Tuple[str, Optional[int], Optional[int]]:
    """Parse a metric name like val_acc_group_00_cluster_01 into a tuple like ("val_acc", 0, 1)."""
    metric = metric_name
    group = None
    cluster = None
    match = re.search("(_group_[0-9]+)", metric_name)
    if match:
        metric = metric_name[0 : match.span()[0]]
        group_str = match.group()
        group_num = re.search("([0-9]+)", group_str)
        if group_num:
            group = int(group_num.group())
    match_cluster = re.search("(_cluster_[0-9]+)", metric_name)
    if match_cluster:
        if group is None:
            metric = metric_name[0 : match_cluster.span()[0]]
        cluster_str = match_cluster.group()
        cluster_num = re.search("([0-9]+)", cluster_str)
        if cluster_num:
            cluster = int(cluster_num.group())
    is_cluster = not match and not match_cluster and re.search("(^cluster_[0-9]+)", metric_name)
    if is_cluster:
        cluster_str = is_cluster.group()
        cluster_num = re.search("([0-9]+)", cluster_str)
        if cluster_num:
            cluster = int(cluster_num.group())
            metric = "clients"
    return (metric, group, cluster)


def parse_group_metrics(metrics: Metrics, group_levels: List[str] = None):
    """Parse a metric dict like {"val_acc_cluster_00_group_01": 0.5}
    into a record dict like:
    [{"metric": "val_acc", "value": 0.5, "cluster": 0, "group": 1}]
    so that we can easily read into a Pandas DataFrame and analyze it.
    """
    result = []
    for k, v in metrics.items():
        metric_name, group, cluster = parse_group_metric_name(k)
        record = {"metric": metric_name, "value": v}
        if group is not None:  # could be 0 which is falsey
            record.update({"group": group})
        if cluster is not None:
            record.update({"cluster": cluster})
        result.append(record)
    return result


def dataframe_from_history(
    history: Optional[History],
    dataset: Optional[Union[Dataset, str]] = None,
    strategy: Optional[Union[Strategy, str]] = None,
    group_levels: Optional[List[str]] = None,
    trial: Optional[int] = None,
    concentration: Optional[Union[float, List[float]]] = None,
):
    """"""
    if history is None:
        return pd.DataFrame(
            {
                k: []
                for k in [
                    "metric",
                    "round",
                    "value",
                    "cluster",
                    "group",
                    "dataset",
                    "strategy",
                    "concentration",
                    "trial",
                ]
            }
        )
    metrics = history.metrics_distributed
    result = []
    for k, v in metrics.items():
        metric_name, group, cluster = parse_group_metric_name(k)
        for rnd in v:
            record = {"metric": metric_name, "round": rnd[0], "value": rnd[1]}
            if group is not None:  # could be 0 which is falsey
                record.update({"group": group})
            if cluster is not None:
                record.update({"cluster": cluster})
            result.append(record)
    df = pd.DataFrame.from_records(result)
    if group_levels is not None:
        df = df.assign(group=df["group"].replace(dict(enumerate(group_levels))))
    if dataset:
        if isinstance(dataset, Dataset):
            dataset_str = dataset.__class__.__name__
        else:
            dataset_str = dataset
        df = df.assign(dataset=dataset_str)
    if strategy:
        if isinstance(strategy, Strategy):
            strategy_str = strategy.__class__.__name__
        else:
            strategy_str = strategy
        df = df.assign(strategy=strategy_str)
    if trial is not None:
        df = df.assign(trial=trial)
    if concentration is not None:
        df = df.assign(concentration=[concentration] * len(df))
    return df
