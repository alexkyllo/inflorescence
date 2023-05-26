"""Weighted Clustered Federated Learning (WeCFL) from Ma et al 2022.
https://arxiv.org/abs/2202.06187
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
from sklearn.cluster import KMeans


def flatten_weights(weights: NDArrays):
    return np.concatenate([w.flatten() for w in weights])


class WeCFL(FedAvg):
    """WeCFL from Ma et al 2022.
    https://arxiv.org/abs/2202.06187
    https://github.com/jie-ma-ai/FedBase/blob/main/fedbase/baselines/wecfl.py
    Initialize k models with different random seeds
    Repeat until convergence:
        Broadcast all k models to random subset of n clients
        Expectation step: Each client computes loss on each model and picks model w/ argmin loss
        to assign itself to a cluster.
        Maximization step: Server computes cluster center H_k
        Distribution step: Server sends H_k to clients in cluster k
        Local update step: Client trains H_k for n epochs on local data

    1. randomly initialize server model
    2. randomly initialize client model on each client (?)
    3. assign server model to each client
    4. randomly initialize k cluster models on the server
    for each round:
        5. run local update epochs on each client and send back to server
        6. run k-means clustering on the server https://github.com/jie-ma-ai/FedBase/blob/2c43137177d528516d6c195482dcb0fbd7a0ae57/fedbase/server/server.py#L81
        7. weighted average together the models from clients in the same k-means cluster
        8. send cluster average model to each client assigned to that cluster
    """

    def __init__(
        self,
        k_clusters: int,
        initial_parameters: Parameters,
        fraction_fit: float = 1,
        fraction_evaluate: float = 1,
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
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ):
        """Weighted Clustered Federated Learning (WeCFL)

        Parameters
        ----------
        k_clusters: int
            The number of cluster-wise models to initialize. When k = 1 this is
            equivalent to FedAvg.
        initial_parameters : Parameters
            Initial global model parameters.
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
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        """
        self.k_clusters = k_clusters
        self.weights_len: Optional[int] = None
        self.initial_parameters = initial_parameters
        self.cluster_parameters = {}  # key: cluster #, value: parameters
        self.cluster_members = {}  # key: cid, value: cluster #
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

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.
        If server_round > 1, send each client the weights
        """
        config: Dict[str, Scalar] = {"server_round": server_round}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        logger.info(
            "configure_fit: round {} cluster members: {}", server_round, self.cluster_members
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
            cluster_num = self.cluster_members.get(str(client.cid), -1)
            client_params = self.cluster_parameters.get(cluster_num, parameters)
            cluster_counts.setdefault(cluster_num, 0)
            cluster_counts[cluster_num] += 1
            configs.append((client, FitIns(client_params, client_config)))
        logger.info("configure_fit: round {}, clusters: {}", server_round, cluster_counts)
        return configs

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        logger.info(
            "configure_evaluate: round {} cluster members: {}", server_round, self.cluster_members
        )
        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        # Return client/config pairs including each client's current cluster assignment
        configs = []
        # Do not resample clients for evaluation, always use the same set of
        # clients as the preceding fit() round.
        for client in self.clients:
            client_config = {}
            client_config.update(config)
            # send cluster centroid model to each client in the cluster for eval.
            cluster_num = self.cluster_members.get(str(client.cid), -1)
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

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        client_weights = {}
        client_ns = {}
        client_metrics = {}
        # weight_results: NDArrays = []
        weight_results: List[NDArrays] = []
        for client, res in results:
            client_weights[client.cid] = parameters_to_ndarrays(res.parameters)
            client_ns[client.cid] = res.num_examples
            client_metrics[client.cid] = res.metrics
        ns = np.array(list(client_ns.values()))
        n = ns.sum()
        # all_weights = np.concatenate([flatten_weights(w) for w in client_weights.values()])
        all_weights = np.asarray([flatten_weights(w) for w in client_weights.values()])
        # Clustering step
        labels = (
            KMeans(n_clusters=self.k_clusters, n_init=5)
            .fit(all_weights, sample_weight=np.divide(ns, n))
            .labels_
        )
        cluster_counts = {}  # for logging
        cluster_weights = {}
        cluster_metrics = {}
        metrics_aggregated = {}
        for i, (cid, weights) in enumerate(client_weights.items()):
            cluster_num = labels[i]
            self.cluster_members[cid] = cluster_num
            cluster_counts.setdefault(cluster_num, 0)
            cluster_counts[cluster_num] += 1
            num_examples = client_ns[cid]
            cluster_weights.setdefault(cluster_num, []).append((weights, num_examples))
            if self.fit_metrics_aggregation_fn:
                cluster_metrics.setdefault(cluster_num, []).append(
                    (num_examples, client_metrics[cid])
                )

        logger.info("aggregate_fit: round {} cluster counts: {}", server_round, cluster_counts)
        logger.info(
            "aggregate_fit: round {} cluster members: {}", server_round, self.cluster_members
        )
        if self.fit_metrics_aggregation_fn:
            metrics_aggregated = self.fit_metrics_aggregation_fn(sum(cluster_metrics.values(), []))
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # Each cluster's new weights are either average of clients in that cluster
        # or if the cluster is empty, initial weights.
        for i in range(self.k_clusters):
            if i in cluster_weights:
                # weight_results.extend(aggregate(cluster_weights[i]))
                cluster_weights_agg = aggregate(cluster_weights[i])
                weight_results.append(cluster_weights_agg)
                if self.fit_metrics_aggregation_fn:
                    metrics_aggregated.update(
                        {
                            f"{key}_cluster_{i:02}": val
                            for key, val in self.fit_metrics_aggregation_fn(
                                cluster_metrics[i]
                            ).items()
                        }
                    )
            else:
                # weight_results.extend(parameters_to_ndarrays(self.initial_parameters))
                cluster_weights_agg = parameters_to_ndarrays(self.initial_parameters)
                weight_results.append(cluster_weights_agg)
            self.cluster_parameters[i] = ndarrays_to_parameters(cluster_weights_agg)
            # parameters_aggregated = ndarrays_to_parameters(weight_results)
            # self.cluster_parameters[k] = {
            #     k: ndarrays_to_parameters(w) for k, w in enumerate(weight_results)
            # }
        # self.initial_parameters = parameters_aggregated
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
        metrics_aggregated = {}
        cluster_counts = {}
        ares = np.zeros((len(results), 3))
        for i, (client, res) in enumerate(results):
            cluster_num = self.cluster_members.get(client.cid, -1)
            # TODO: debug
            cluster_counts.setdefault(cluster_num, 0)
            cluster_counts[cluster_num] += 1
            # TODO: debug
            if self.evaluate_metrics_aggregation_fn:
                cluster_metrics.setdefault(cluster_num, []).append((res.num_examples, res.metrics))
            row = np.array([res.num_examples, res.loss, cluster_num])
            ares[i, :] = row
        loss_aggregated = np.average(ares[:, 1], weights=ares[:, 0])
        logger.info("aggregate_evaluate: round {} cluster counts: {}", server_round, cluster_counts)
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
        return f"WeCFL(k_clusters={self.k_clusters}, accept_failures={self.accept_failures})"
