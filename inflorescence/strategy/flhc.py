"""FL+HC Federated Learning + Hierarchical Clustering
from Briggs et al 2020.
https://arxiv.org/abs/2004.11791
"""
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    FitIns,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Metrics,
    MetricsAggregationFn,
    NDArray,
    NDArrays,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from loguru import logger
from sklearn.cluster import AgglomerativeClustering

from inflorescence.client import FlowerClient


def cluster_clients(
    weights: Dict[str, NDArrays],
    metric: str,
    linkage: str,
    k_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
) -> Dict[str, int]:
    """Cluster client models based on distance in parameter space."""
    flat_weights = [np.concatenate([m.flatten() for m in w]) for w in weights.values()]
    clustering = AgglomerativeClustering(
        n_clusters=k_clusters,
        distance_threshold=distance_threshold,
        metric=metric,
        linkage=linkage,
    ).fit(flat_weights)

    return {cid: clustering.labels_[i] for i, cid in enumerate(weights.keys())}


class FLHC(FedAvg):
    """FL+HC Federated Learning + Hierarchical Clustering from Briggs et al 2020.

    Initialize weights
    Sample clients
    Train each client for n rounds with FedAvg
    Perform Agglomerative Clustering on the weights with a distance threshold to split clients into clusters
    Continue FedAvg within the clusters until convergence
    """

    def __init__(
        self,
        k_clusters: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        metric: Union[str, Callable] = "euclidean",
        linkage: str = "complete",
        cluster_after_round: int = 20,
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
        """FL+HC Federated Learning + Hierarchical Clustering from Briggs et al 2020.

        Parameters
        ----------
        k_clusters: int, optional
            The number of clusters to find in the data. Mutually exclusive with `distance_threshold`.
        distance_threshold: float, optional
            The distance threshold at or above which clusters will not be merged. Mutually
            exclusive with `k_clusters`.
        distance: str, optional
            Distance metric to use for agglomerative clustering.
            Options are "manhattan" (L1), "euclidean" (L2) and "cosine".
        linkage: str, optional
            Linkage to use in the agglomerative clustering.
            Options are "complete", "average", "single" and "ward".
        cluster_after_round: int, optional
            Conduct hierarchical clustering after this round of federated model fitting.
            Defaults to 20.
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
        if not bool(k_clusters) ^ bool(distance_threshold):
            raise ValueError("Exactly one of k_clusters and distance_threshold must be provided.")
        if bool(k_clusters) and k_clusters < 1:
            raise ValueError("If provided, k_clusters must be >= 1.")
        if bool(distance_threshold) and distance_threshold <= 0:
            raise ValueError("If provided, distance_threshold must be positive.")
        self.metric = metric
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.cluster_after_round = cluster_after_round
        self.clusters: Dict[str, int] = {}
        self.k_clusters = k_clusters
        self.num_clusters = 1  # starts with one cluster
        self.cluster_parameters: Dict[int, Parameters] = {}
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
        return FlowerClient

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        weights: Dict[str, NDArrays] = {}  # key: cid, value: weights
        metrics: Dict[str, Metrics] = {}  # key: cid, value: metrics
        weight_results: List[NDArrays] = []
        num_examples: Dict[str, int] = {}  # key: cid, value: num examples per cid
        metrics_aggregated: Metrics = {}
        for client, fit_res in results:
            cid = client.cid
            weights[cid] = parameters_to_ndarrays(fit_res.parameters)
            num_examples[cid] = fit_res.num_examples
            if self.fit_metrics_aggregation_fn:
                metrics[cid] = fit_res.metrics
        if server_round == self.cluster_after_round:
            log(1, f"FLHC clustering clients at round {server_round}")
            self.clusters = cluster_clients(
                weights=weights,
                metric=self.metric,
                linkage=self.linkage,
                k_clusters=self.k_clusters,
                distance_threshold=self.distance_threshold,
            )
            self.num_clusters = len(set(self.clusters.values()))
        clients_in_cluster = {}
        if len(self.clusters) == 0:  # all clients are still in cluster 0
            clients_in_cluster = {0: list(weights.keys())}
            # TODO: If we got a client that hasn't been sampled yet,
            # use k nearest neighbors to assign it to the nearest cluster.
        else:
            for k, v in self.clusters.items():
                clients_in_cluster.setdefault(v, []).append(k)
        logger.info("aggregate_evaluate: round {} clusters: {}", server_round, clients_in_cluster)
        for i in range(self.num_clusters):
            clients = clients_in_cluster[i]
            if len(clients) > 0:
                cluster_weights: List[Tuple[NDArrays, int]] = []
                cluster_metrics: List[Tuple[int, Metrics]] = []
                for c in clients:
                    # Skip clients who failed
                    if c in weights:
                        client_weights = weights[c]
                        client_examples = num_examples[c]
                        cluster_weights.append((client_weights, client_examples))
                        if self.fit_metrics_aggregation_fn:
                            cluster_metrics.append((client_examples, metrics[c]))
                cluster_weights_agg = aggregate(cluster_weights)
                weight_results.append(cluster_weights_agg)
                if self.fit_metrics_aggregation_fn:
                    metrics_aggregated.update(
                        {
                            f"{key}_cluster_{i:02}": val
                            for key, val in self.fit_metrics_aggregation_fn(cluster_metrics).items()
                        }
                    )
                self.cluster_parameters[i] = ndarrays_to_parameters(cluster_weights_agg)
        params_flat = ndarrays_to_parameters([_ for _ in weight_results for _ in _])
        return params_flat, metrics_aggregated

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
        metrics_aggregated: Dict[str, Scalar] = {}
        ares = np.zeros((len(results), 3))
        for i, (client, res) in enumerate(results):
            cluster_num = self.clusters.get(client.cid, 0)
            if self.evaluate_metrics_aggregation_fn:
                cluster_metrics.setdefault(cluster_num, []).append((res.num_examples, res.metrics))
            row = np.array([res.num_examples, res.loss, cluster_num])
            ares[i, :] = row
        loss_aggregated = float(np.average(ares[:, 1], weights=ares[:, 0]))

        for i in range(self.num_clusters):
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

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config: Dict[str, Scalar] = {"server_round": server_round}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        if server_round > self.cluster_after_round or self.num_clusters > 1:
            logger.info(
                "configure_fit: round {} cluster memberships: {}", server_round, self.clusters
            )

            sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
            self.clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
            # Return client/config pairs including each client's current cluster assignment
            configs = []
            cluster_counts = {}
            for client in self.clients:
                client_config = {}
                client_config.update(config)
                cluster_num = self.clusters.get(client.cid, -1)
                # TODO: implement handling for new clients who weren't in the clustering round.
                client_params = self.cluster_parameters.get(cluster_num, parameters)
                cluster_counts.setdefault(cluster_num, 0)
                cluster_counts[cluster_num] += 1
                configs.append((client, FitIns(client_params, client_config)))
            logger.info("configure_fit: round {}, clusters: {}", server_round, cluster_counts)
            return configs
        else:
            return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if server_round > self.cluster_after_round or self.num_clusters > 1:
            # Do not configure federated evaluation if fraction eval is 0.
            if self.fraction_evaluate == 0.0:
                return []
            logger.info(
                "configure_evaluate: round {} cluster memberships: {}", server_round, self.clusters
            )
            sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
            self.clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
            # Parameters and config
            config = {}
            if self.on_evaluate_config_fn is not None:
                # Custom evaluation config function provided
                config = self.on_evaluate_config_fn(server_round)
            # Return client/config pairs including each client's current cluster assignment
            configs = []
            for client in self.clients:
                client_config = {}
                client_config.update(config)
                # send cluster centroid model to each client in the cluster for eval.
                cluster_num = self.clusters.get(str(client.cid), -1)
                client_params = self.cluster_parameters.get(cluster_num, parameters)
                configs.append((client, EvaluateIns(client_params, client_config)))
                logger.info(
                    "round {} configure_evaluate() client {}, cluster {}, client_config: {}",
                    server_round,
                    client.cid,
                    cluster_num,
                    client_config,
                )
            return configs
        else:
            return super().configure_evaluate(server_round, parameters, client_manager)

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize k sets of global model parameters, one for each cluster."""
        return self.initial_parameters

    def __str__(self):
        return f"FLHC(k_clusters={self.k_clusters}, distance_threshold={self.distance_threshold}, metric={self.metric}, linkage={self.linkage}, accept_failures={self.accept_failures})"
