"""IFCA strategy implementation for Flower.
Ghosh et al, An Efficient Framework for Clustered Federated Learning, 2020.
https://arxiv.org/pdf/2006.04088.pdf
https://github.com/jichan3751/ifca
"""

from logging import INFO, WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from loguru import logger

from inflorescence.client import IFCAClient


def concatenate_ndarrays(ndarrays_list: List[NDArrays]) -> Parameters:
    """Helper function for concatenating a list of Parameters together.
    IFCA requires all cluster parameters to be packed into one Parameters instance
    for compatibility with the Flower Strategy API.
    """
    ndarrays_flat = [_ for _ in ndarrays_list for _ in _]
    return ndarrays_to_parameters(ndarrays_flat)


class IFCA(FedAvg):
    """A Flower Strategy for the Iterative Federated Clustering Algorithm
    from Ghosh et al 2020 (model averaging version).

    Initialize k models with different random seeds
    Broadcast all k models to random subset of n clients
    Each client computes loss on each model and picks model w/ argmin loss
    and then trains that model for t steps
    Client sends updated model and cluster identity to server
    Server averages together updated models from clients within same cluster
    """

    def __init__(
        self,
        k_clusters: int,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ):
        """Iterative Federated Clustering Algorithm.

        Parameters
        ----------
        k_clusters: int
            The number of cluster-wise models to initialize. When k = 1 this is
            equivalent to FedAvg.
        fraction_fit : float, optional
            Fraction of clients used during training. In case `min_fit_clients`
            is larger than `fraction_fit * available_clients`, `min_fit_clients`
            will still be sampled. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. In case `min_evaluate_clients`
            is larger than `fraction_evaluate * available_clients`, `min_evaluate_clients`
            will still be sampled. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        """
        self.k_clusters = k_clusters
        self.weights_len: Optional[int] = None
        if initial_parameters:
            self.weights_len = len(parameters_to_ndarrays(initial_parameters)) // self.k_clusters
            if self.weights_len < 1:
                raise ValueError("IFCA strategy requires one set of weights per cluster.")
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            initial_parameters=initial_parameters,
        )

    @property
    def client_class(self):
        return IFCAClient

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize k sets of global model parameters, one for each cluster."""
        return self.initial_parameters

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average per cluster."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        weight_results: NDArrays = []
        cluster_weights = {}
        cluster_metrics = {}
        metrics_aggregated = {}
        # Collect client responses by cluster number
        cluster_counts = {}
        for _, fit_res in results:
            cluster_num = fit_res.metrics.get("cluster")
            # TODO: debug
            cluster_counts.setdefault(cluster_num, 0)
            cluster_counts[cluster_num] += 1
            # TODO: debug
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            if self.weights_len is None:
                self.weights_len = len(client_weights)
            cluster_weights.setdefault(cluster_num, []).append(
                (client_weights, fit_res.num_examples)
            )
            # Aggregate custom metrics if aggregation fn was provided
            if self.fit_metrics_aggregation_fn:
                cluster_metrics.setdefault(cluster_num, []).append(
                    (fit_res.num_examples, fit_res.metrics)
                )
        logger.debug("aggregate_fit: round {} cluster counts: {}", server_round, cluster_counts)
        if self.fit_metrics_aggregation_fn:
            metrics_aggregated = self.fit_metrics_aggregation_fn(sum(cluster_metrics.values(), []))
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Each cluster's new weights are either average of clients in that cluster
        # or if the cluster is empty, initial weights.
        for i in range(self.k_clusters):
            if i in cluster_weights:
                weight_results.extend(aggregate(cluster_weights[i]))
                if self.fit_metrics_aggregation_fn:
                    metrics_aggregated.update(
                        {
                            f"{key}_cluster_{i:02}": val
                            for key, val in self.fit_metrics_aggregation_fn(
                                cluster_metrics[i]
                            ).items()
                        }
                    )
            elif self.weights_len is not None:  # empty cluster
                j = i * self.weights_len
                if self.initial_parameters is not None:
                    initial_weights = parameters_to_ndarrays(self.initial_parameters)[
                        j : j + self.weights_len
                    ]
                    weight_results.extend(initial_weights)

        parameters_aggregated = ndarrays_to_parameters(weight_results)
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Aggregate losses and metrics overall and cluster-wise.
        cluster_metrics = {}
        metrics_aggregated = {}
        cluster_counts = {}
        ares = np.zeros((len(results), 3))
        for i, (_, res) in enumerate(results):
            cluster_num = res.metrics.get("cluster", 0)
            # TODO: debug
            cluster_counts.setdefault(cluster_num, 0)
            cluster_counts[cluster_num] += 1
            # TODO: debug
            if self.evaluate_metrics_aggregation_fn:
                cluster_metrics.setdefault(cluster_num, []).append((res.num_examples, res.metrics))
            row = np.array([res.num_examples, res.loss, cluster_num])
            ares[i, :] = row
        loss_aggregated = np.average(ares[:, 1], weights=ares[:, 0])
        logger.debug(
            "aggregate_evaluate: round {} cluster counts: {}", server_round, cluster_counts
        )
        for i in range(self.k_clusters):
            if np.sum(ares[ares[:, 2] == i, 0]) > 0:
                metrics_aggregated[f"loss_cluster_{i:02}"] = np.average(
                    ares[ares[:, 2] == i, 1], weights=ares[ares[:, 2] == i, 0]
                )
                if self.evaluate_metrics_aggregation_fn:
                    metrics_aggregated.update(
                        {
                            f"{key}_cluster_{i:02}": val
                            for key, val in self.evaluate_metrics_aggregation_fn(
                                cluster_metrics[i]
                            ).items()
                            if not key.startswith("cluster")
                        }
                    )
            else:  # empty cluster
                metrics_aggregated[f"loss_cluster_{i:02}"] = np.nan

        if self.evaluate_metrics_aggregation_fn:
            metrics_aggregated.update(
                self.evaluate_metrics_aggregation_fn(sum(cluster_metrics.values(), []))
            )
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        return loss_aggregated, metrics_aggregated

    def __str__(self):
        return f"IFCA(k_clusters={self.k_clusters}, accept_failures={self.accept_failures})"
