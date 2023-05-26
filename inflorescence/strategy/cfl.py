"""Clustered Federated Learning
from Sattler et al 2019.
Recursive bi-partitioning of clients based on cosine similarity of
updates.
https://arxiv.org/abs/1910.01991
https://github.com/felisat/clustered-federated-learning
"""
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
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
from numpy.linalg import norm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from inflorescence.client import CFLClient


def flatten_weights(weights: NDArrays):
    return np.concatenate([w.flatten() for w in weights])


def pairwise_cosine(x: List[NDArrays]) -> NDArray:
    """Get the pairwise cosine distances between every pair of vectors in X."""
    # xf = np.array([np.asarray(a).flatten() for a in x])
    xf = [flatten_weights(a) for a in x]
    return cosine_distances(xf)


def cluster_clients(dist: NDArray) -> Tuple[NDArray, NDArray]:
    """Cluster clients based on the pairwise cosine distance matrix."""
    clustering = AgglomerativeClustering(
        n_clusters=2, metric="precomputed", linkage="complete"
    ).fit(dist)
    cluster_1 = np.argwhere(clustering.labels_ == 0).flatten()
    cluster_2 = np.argwhere(clustering.labels_ == 1).flatten()
    return cluster_1, cluster_2


class CFL(FedAvg):
    """Clustered Federated Learning from Sattler et al 2019.
    Sample clients
    Initialize one cluster with all clients in it
    Train each client for one epoch
    Subtract each client's weights - old weights to get weight update
    Compute pairwise cosine similarities between all pairs of clients
    For each cluster:
        Check stopping criteria:
        if mean(norm(weight_update)) and eps_1 or max(norm(weight_update)) > eps_2
            and there are >2 clusters and round > 20:
            Use AgglomerativeClustering with complete linkage on cosine
            similarities to split the cluster in two
        Aggregate parameters with FedAvg clusterwise
    """

    def __init__(
        self,
        eps_1: float = 0.4,  # default used in CFL paper
        eps_2: float = 1.6,  # default used in CFL paper
        cluster_after_round: int = 20,  # default used in CFL paper
        k_clusters: int = 1,  # If > 1, then the data is already clustered
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
        """Clustered Federated Learning (CFL) from Sattler et al 2019.
        Parameters
        ----------
        eps_1: float, optional
            Epsilon for mean norm of weight update in stopping criterion.
            Bi-partitioning will only happen if the weight update mean norm < eps_1,
            indicating that the cluster's model is close to convergence.
            Defaults to 0.4 as in the CFL paper.
        eps_2: float, optional
            Epsilon for max norm of weight update in stopping criterion
            Bi-partitioning will only happen if the weight update max norm > eps_2,
            indicating that at least one client within the cluster is far from
            convergence. Defaults to 1.6 as in the CFL paper.
        cluster_after_round: int, optional
            Start attempting recursive bipartitioning after this round number.
            Defaults to 20 as in the CFL paper.
        k_clusters: int, optional
            The number of clusters that the initial parameters represent. If >1
            then cluster_after_round will be ignored. Defaults to 1.
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
        if eps_1 <= 0 or eps_2 <= 0:
            raise ValueError("eps_1 and eps_2 must be strictly positive.")
        self.eps_1 = eps_1
        self.eps_2 = eps_2
        self.cluster_after_round = cluster_after_round
        # Dictionary to track which cluster each client is in.
        # key = cid, value = cluster number.
        # Cluster number is an index into the Parameters array.
        self.clusters: Dict[int, int] = {}
        # The algorithm starts with all clients in one big cluster
        if k_clusters < 1:
            raise ValueError(f"k_clusters must be a positive number, but was {k_clusters}.")
        self.k_clusters = k_clusters
        # Keep track of how many layers are in the model so we can split the params clusterwise
        self.weights_len: Optional[int] = None
        if initial_parameters:
            param_len = len(parameters_to_ndarrays(initial_parameters))
            self.weights_len = param_len // self.k_clusters
            if self.weights_len < 1:
                raise ValueError("CFL strategy requires one set of weights per cluster.")
        # Each client starts in cluster 0 but the server doesn't know all the
        # client IDs until they connect. Upon each round, each new client will be
        # assigned to cluster 0 as a default.
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
        return CFLClient

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize k sets of global model parameters, one for each cluster."""
        return self.initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        # Return client/config pairs including each client's current cluster assignment
        configs = [
            (client, FitIns(parameters, {**config, "cluster": self.clusters.get(client.cid, 0)}))
            for client in clients
        ]
        return configs

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        configs = [
            (
                client,
                EvaluateIns(parameters, {**config, "cluster": self.clusters.get(client.cid, 0)}),
            )
            for client in clients
        ]

        # Return client/config pairs
        return configs

    def get_cluster_weights(self, cluster_num: int, parameters: Parameters):
        weights = parameters_to_ndarrays(parameters)
        if self.weights_len is None:
            self.weights_len = len(weights) // self.k_clusters
        weights_start = cluster_num * self.weights_len
        weights_end = weights_start + self.weights_len
        return weights[weights_start:weights_end]

    def delta(self, cluster_num: int, new_weights: NDArrays):
        """Get the weight deltas from previous round for one client."""
        prev_weights = self.get_cluster_weights(cluster_num, self.initial_parameters)
        return list(np.subtract(new_weights, prev_weights))
        # ValueError: could not broadcast input array from shape (24,) into shape (1,)

    def should_cluster(self, updates: List[NDArrays]) -> bool:
        """Test if the splitting criteria are met."""
        len_criterion = len(updates) > 2
        update_norm = [norm(flatten_weights(u)) for u in updates]
        max_norm = np.max(update_norm)
        mean_norm = np.mean(update_norm)
        norm_criterion = mean_norm < self.eps_1 and max_norm > self.eps_2
        logger.info(
            "should_cluster if {} < {} and {} > {} and {} > 2",
            mean_norm,
            self.eps_1,
            max_norm,
            self.eps_2,
            len(updates),
        )
        return len_criterion and norm_criterion

    def cluster(
        self,
        weight_updates: Dict[int, List[Tuple[int, NDArrays]]],
        new_weights: Dict[int, Tuple[NDArrays, int]],
    ) -> List[List[Tuple[NDArrays, int]]]:
        """"""
        new_cluster_params = []
        # Iterate over the existing clusters and their client updates,
        # splitting the clusters that meet the splitting criteria
        for _, client_updates in weight_updates.items():
            updates = []
            cluster_cids = []
            for cu in client_updates:
                cluster_cids.append(cu[0])
                updates.append(cu[1])
            # Splitting criteria: only bipartition this cluster if it is incongruent
            # and there are >2 clients in it
            cluster_1_weights = []
            if self.should_cluster(updates):
                cluster_2_weights = []
                cluster_1, cluster_2 = cluster_clients(pairwise_cosine(updates))
                for idx in cluster_1:
                    cid = cluster_cids[idx]
                    self.clusters[cid] = len(new_cluster_params)
                    cluster_1_weights.append(new_weights[cid])
                for idx in cluster_2:
                    cid = cluster_cids[idx]
                    self.clusters[cid] = len(new_cluster_params) + 1
                    cluster_2_weights.append(new_weights[cid])
                new_cluster_params += [cluster_1_weights, cluster_2_weights]
            else:
                for cid in cluster_cids:
                    self.clusters[cid] = len(new_cluster_params)
                    cluster_1_weights.append(new_weights[cid])
                new_cluster_params += [cluster_1_weights]
        self.k_clusters = len(new_cluster_params)
        return new_cluster_params

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        For each client:
            Compute weight update for each client
        For each cluster:
            Check stopping criteria for bipartitioning
            If cluster is incongruent, compute pairwise cosine similarities of updates
            Then split the cluster into two with AgglomerativeClustering
            Average together weights clusterwise and update self.initial_parameters
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        weight_results: NDArrays = []
        new_client_weights: Dict[int, Tuple[NDArrays, int]] = {}  # cid: (weights, num_examples)
        # client_num_examples = {}  # cid: num_examples
        client_metrics: Dict[int, Tuple[int, Metrics]] = {}
        weight_updates = {}  # index: cluster number, value: tuple (cid, NDArrays of weight updates)
        cluster_weights = {}
        cluster_counts = {}
        # First, gather the client weights into a dict
        for client, fit_res in results:
            # Compute client's weight update delta
            # get the existing weights for this client's cluster
            # subtract new weights - old weights
            # insert into weight_updates dict for norm calculation and clustering
            # Also insert into client_weights dict for clusterwise averaging
            cid = int(client.cid)
            cluster_num = self.clusters.get(cid, 0)
            cluster_counts.setdefault(cluster_num, 0)
            cluster_counts[cluster_num] += 1
            new_weights = parameters_to_ndarrays(fit_res.parameters)
            cluster_weights.setdefault(cluster_num, []).append((new_weights, fit_res.num_examples))
            new_client_weights[cid] = (new_weights, fit_res.num_examples)
            # client_num_examples[client.cid] = fit_res.num_examples
            if self.fit_metrics_aggregation_fn:
                # keep metrics per client because cluster assignments may change
                client_metrics[cid] = (fit_res.num_examples, fit_res.metrics)
            if server_round > self.cluster_after_round:
                # We only need the updates if we're clustering
                deltas = self.delta(cluster_num, new_weights)
                weight_updates.setdefault(cluster_num, [])
                weight_updates[cluster_num].append((cid, deltas))
        logger.info("aggregate_fit: round {} cluster counts: {}", server_round, cluster_counts)
        if server_round <= self.cluster_after_round and self.k_clusters == 1:
            # just federated averaging within cluster 0
            for i in range(self.k_clusters):
                weight_results.extend(aggregate(cluster_weights[i]))
        else:
            new_cluster_params = self.cluster(weight_updates, new_client_weights)
            for clust_w in new_cluster_params:
                weight_results.extend(aggregate(clust_w))
        parameters_aggregated = ndarrays_to_parameters(weight_results)
        self.initial_parameters = parameters_aggregated
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            # fit_metrics = [(client_num_examples[cid], client_metrics[cid]) for cid in self.clusters]
            # fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            # Overall metric aggregation
            metrics_aggregated = self.fit_metrics_aggregation_fn(list(client_metrics.values()))
            # Cluster-wise aggregation
            if self.k_clusters > 1:
                client_clusters = {}
                for cid, cluster_num in self.clusters.items():
                    client_clusters.setdefault(cluster_num, []).append(cid)
                for i in range(self.k_clusters):
                    cluster_metrics = []
                    cluster_clients = client_clusters[i]
                    for cid in cluster_clients:
                        cluster_metrics.append(client_metrics[cid])
                    metrics_agg = self.fit_metrics_aggregation_fn(cluster_metrics)
                    metrics_agg = {f"{key}_cluster_{i:02}": val for key, val in metrics_agg.items()}
                    metrics_aggregated.update(metrics_agg)
            else:
                metrics_aggregated.update(
                    {f"{metric}_cluster_00": value for metric, value in metrics_aggregated}
                )
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
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
        metrics_aggregated: Dict[str, Scalar] = {}
        ares = np.zeros((len(results), 3))
        cluster_counts = {}
        for i, (client, res) in enumerate(results):
            cluster_num = self.clusters.get(int(client.cid), 0)
            cluster_counts.setdefault(cluster_num, 0)
            cluster_counts[cluster_num] += 1
            if self.evaluate_metrics_aggregation_fn:
                cluster_metrics.setdefault(cluster_num, []).append((res.num_examples, res.metrics))
            row = np.array([res.num_examples, res.loss, cluster_num])
            ares[i, :] = row
        logger.info("aggregate_evaluate: round {} cluster counts: {}", server_round, cluster_counts)
        loss_aggregated = float(np.average(ares[:, 1], weights=ares[:, 0]))

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

    def assign_new_client():
        """TODO: Algorithm 4 from the CFL paper, to assign new / clients to existing clusters."""

    def __str__(self):
        return (
            f"CFL(eps_1={self.eps_1}, eps_2={self.eps_2}, accept_failures={self.accept_failures})"
        )
