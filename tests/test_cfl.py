"""Tests for strategy/cfl.py"""
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Metrics,
    NDArrays,
    ReconnectIns,
    Status,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from inflorescence.client import CFLClient
from inflorescence.metrics import weighted_average
from inflorescence.strategy.cfl import CFL, cluster_clients, pairwise_cosine
from inflorescence.model import Model

def test_cosine():
    """Test that function to calculate pairwise cosine of parameter updates works."""
    updates: List[NDArrays] = [[np.array([[0, 1], [1, 0]])], [np.array([[1, 0], [0, 1]])]]
    cdist = pairwise_cosine(updates)

    assert (cdist == np.array([[0, 1], [1, 0]])).all()


def test_cluster_clients():
    updates = [np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, 1]]), np.array([[1, 1], [0, 1]])]
    cdist = pairwise_cosine(updates)
    c1, c2 = cluster_clients(cdist)
    assert np.array_equal(c1, np.array([1, 2]))
    assert np.array_equal(c2, np.array([0]))


def test_delta():
    initial_weights = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    new_weights = [np.array([4.0, 6.0])]
    expected = [np.array([1.0, 2.0])]
    cluster_num = 1
    cfl = CFL(initial_parameters=ndarrays_to_parameters(initial_weights), k_clusters=2)
    actual = cfl.delta(cluster_num, new_weights)
    for i, a in enumerate(actual):
        assert np.array_equal(a, expected[i])


def test_cfl_matches_fedavg_before_first_clustering_round():
    """CFL result should match FedAvg result within first n rounds, except
    that the parameters have an extra layer of nesting."""
    # nesting: outer List is one np.array per cluster
    # within each np.array, first dimension is layers, second is parameters within layer
    # This example is one cluster, one layer, one parameter
    initial_params_cfl = ndarrays_to_parameters([np.array([[1.0]])])
    initial_params_fedavg = ndarrays_to_parameters([np.array([1.0])])
    expected_results = ndarrays_to_parameters([np.array([[1.5]])])
    cfl = CFL(initial_parameters=initial_params_cfl)
    fedavg = FedAvg(initial_parameters=initial_params_fedavg)
    # ifca = FedAvg(initial_parameters=initial_params)
    ok = Status(code=Code.OK, message="OK")
    results = [1.0, 1.0, 2.0, 2.0]
    cfl_results = [[np.array([[i]])] for i in results]
    fedavg_results = [[np.array([i])] for i in results]
    actuals, _ = cfl.aggregate_fit(
        server_round=1,
        results=[
            (
                MagicMock(),
                FitRes(
                    status=ok,
                    parameters=ndarrays_to_parameters(r),
                    num_examples=10,
                    metrics={},
                ),
            )
            for r in cfl_results
        ],
        failures=[],
    )
    fedavg_actuals, _ = fedavg.aggregate_fit(
        server_round=1,
        results=[
            (
                MagicMock(),
                FitRes(
                    status=ok,
                    parameters=ndarrays_to_parameters(r),
                    num_examples=10,
                    metrics={},
                ),
            )
            for r in fedavg_results
        ],
        failures=[],
    )
    assert parameters_to_ndarrays(actuals) == parameters_to_ndarrays(expected_results)
    assert parameters_to_ndarrays(actuals) == parameters_to_ndarrays(fedavg_actuals)


def test_cfl_should_cluster():
    initial_weights = [[a] for a in list(np.ones((6, 2)))]
    weight_updates = [
        [np.array([1.1, 1.2])],
        [np.array([-0.01, -0.01])],
        [np.array([-0.01, 0.01])],
        [np.array([-0.01, 0.01])],
        [np.array([-0.01, 0.01])],
        [np.array([-0.01, 0.01])],
    ]
    initial_params = ndarrays_to_parameters(initial_weights[0])
    cfl = CFL(initial_parameters=initial_params)
    assert cfl.should_cluster(weight_updates)


def test_cfl_cluster_clients():
    weight_updates = [
        np.array([1.1, 1.2]),
        np.array([-0.01, -0.01]),
        np.array([-0.01, 0.01]),
        np.array([-0.01, 0.01]),
        np.array([-0.01, 0.01]),
        np.array([-0.01, 0.01]),
    ]
    c1, c2 = cluster_clients(pairwise_cosine(weight_updates))
    assert np.array_equal(c1, np.array([0, 2, 3, 4, 5]))
    assert np.array_equal(c2, np.array([1]))


ok = Status(code=Code.OK, message="OK")


def test_cfl_cluster():
    initial_weights = [np.ones((2))]
    updates = [
        [np.array([1.1, 1.2])],
        [np.array([-0.01, -0.01])],
        [np.array([-0.01, 0.01])],
        [np.array([-0.01, 0.01])],
        [np.array([-0.01, 0.01])],
        [np.array([-0.01, 0.01])],
    ]
    new_weights = [list(np.add(initial_weights, u)) for u in updates]
    new_client_weights = {i: (w, 10) for i, w in enumerate(new_weights)}
    weight_updates = {0: [(i, w) for i, w in enumerate(updates)]}
    initial_params = ndarrays_to_parameters(initial_weights)
    cfl = CFL(initial_parameters=initial_params)
    results = cfl.cluster(weight_updates, new_client_weights)
    expected = [
        [
            ([np.array([2.1, 2.2])], 10),
            ([np.array([0.99, 1.01])], 10),
            ([np.array([0.99, 1.01])], 10),
            ([np.array([0.99, 1.01])], 10),
            ([np.array([0.99, 1.01])], 10),
        ],
        [([np.array([0.99, 0.99])], 10)],
    ]
    assert len(results) == len(expected) == cfl.k_clusters
    for i, res in enumerate(results):
        for j, (ary, _) in enumerate(res):
            assert np.array_equal(ary, expected[i][j][0])


class CustomClientProxy(ClientProxy):
    """Subclass of ClientProxy."""

    def get_properties(
        self,
        ins: GetPropertiesIns,
        timeout: Optional[float],
    ) -> GetPropertiesRes:
        """Returns the client's properties."""

    def get_parameters(
        self,
        ins: GetParametersIns,
        timeout: Optional[float],
    ) -> GetParametersRes:
        """Return the current local model parameters."""

    def fit(
        self,
        ins: FitIns,
        timeout: Optional[float],
    ) -> FitRes:
        """Refine the provided weights using the locally held dataset."""

    def evaluate(
        self,
        ins: EvaluateIns,
        timeout: Optional[float],
    ) -> EvaluateRes:
        """Evaluate the provided weights using the locally held dataset."""

    def reconnect(
        self,
        ins: ReconnectIns,
        timeout: Optional[float],
    ) -> DisconnectRes:
        """Disconnect and (optionally) reconnect later."""


def test_cfl_aggregate_fit():
    initial_weights = [np.ones((2))]
    updates = [
        [np.array([1.1, 1.2])],
        [np.array([-0.01, -0.01])],
        [np.array([-0.01, 0.01])],
        [np.array([-0.01, 0.01])],
        [np.array([-0.01, 0.01])],
        [np.array([-0.01, 0.01])],
    ]
    new_weights = [list(np.add(initial_weights, u)) for u in updates]
    initial_params = ndarrays_to_parameters(initial_weights)
    cfl = CFL(initial_parameters=initial_params, cluster_after_round=0)
    res = [
        (
            CustomClientProxy(str(i)),
            FitRes(
                status=ok,
                parameters=ndarrays_to_parameters(nw),
                num_examples=10,
                metrics={},
            ),
        )
        for i, nw in enumerate(new_weights)
    ]
    parameters_aggregated, _ = cfl.aggregate_fit(server_round=1, results=res, failures=[])
    results = parameters_to_ndarrays(parameters_aggregated)
    expected = [np.array([1.212, 1.248]), np.array([0.99, 0.99])]
    assert len(results) == len(expected) == cfl.k_clusters
    assert np.isclose(results, expected).all()


def test_cfl_aggregate_fit_metrics():
    initial_weights = [np.ones((2))]
    updates = [
        [np.array([1.1, 1.2])],
        [np.array([-0.01, -0.01])],
        [np.array([-0.01, 0.01])],
        [np.array([-0.01, 0.01])],
        [np.array([-0.01, 0.01])],
        [np.array([-0.01, 0.01])],
    ]
    new_weights = [list(np.add(initial_weights, u)) for u in updates]
    initial_params = ndarrays_to_parameters(initial_weights)
    cfl = CFL(
        initial_parameters=initial_params,
        cluster_after_round=0,
        fit_metrics_aggregation_fn=weighted_average,
    )
    res = [
        (
            CustomClientProxy(str(i)),
            FitRes(
                status=ok,
                parameters=ndarrays_to_parameters(nw),
                num_examples=10,
                metrics={"accuracy": 0.8 + (i * 0.02)},
            ),
        )
        for i, nw in enumerate(new_weights)
    ]
    _, metrics_aggregated = cfl.aggregate_fit(server_round=1, results=res, failures=[])
    assert np.isclose(metrics_aggregated["accuracy"], 0.85)
    assert np.isclose(metrics_aggregated["accuracy_cluster_00"], 0.856)
    assert np.isclose(metrics_aggregated["accuracy_cluster_01"], 0.82)


def test_cfl_aggregate_evaluate():
    """Test that CFL.aggregate_evaluate() works"""
    initial_weights = [np.ones(2), np.ones(2)]
    initial_params = ndarrays_to_parameters(initial_weights)
    cfl = CFL(
        initial_parameters=initial_params,
        cluster_after_round=0,
        evaluate_metrics_aggregation_fn=weighted_average,
        k_clusters=2,
    )
    # Manually assign clients to clusters. This happens in aggregate_fit()
    cfl.clusters = {0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    res = [
        (
            CustomClientProxy(str(i)),
            EvaluateRes(
                status=ok,
                loss=4.0 + (i * 0.2),
                num_examples=10,
                metrics={"accuracy": 0.8 + (i * 0.02)},
            ),
        )
        for i in range(6)
    ]
    loss_aggregated, metrics_aggregated = cfl.aggregate_evaluate(
        server_round=1, results=res, failures=[]
    )
    assert np.isclose(loss_aggregated, 4.5)
    assert np.isclose(metrics_aggregated["loss_cluster_00"], 4.56)
    assert np.isclose(metrics_aggregated["loss_cluster_01"], 4.2)
    assert np.isclose(metrics_aggregated["accuracy"], 0.85)
    assert np.isclose(metrics_aggregated["accuracy_cluster_00"], 0.856)
    assert np.isclose(metrics_aggregated["accuracy_cluster_01"], 0.82)


class lrmodel(Model):
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def evaluate(self, x_test, y_test):
        preds = x_test * self.slope + self.intercept
        loss = np.average(np.sqrt((preds - y_test) ** 2))
        # breakpoint()
        return loss, {"accuracy": 1.0}

    def get_weights(self):
        """"""
        return [
            np.array([[self.slope]], dtype=np.float32),
            np.array([self.intercept], dtype=np.float32),
        ]

    def set_weights(self, weights):
        """"""
        self.slope = weights[0][0][0]
        self.intercept = weights[1][0]

    def fit(self, *args, **kwargs):
        return

def test_cfl_client_fit():
    client = CFLClient(
        model_fn=lambda: lrmodel(1, 1),
        cid=0,
        x_train=np.array([[1.0]]),
        y_train=np.array([2.9]),
        x_test=np.array([[1.0]]),
        y_test=np.array([2.9]),
        epochs=1,
        batch_size=1,
    )
    weights, num_examples, metrics = client.fit(parameters=lrmodel(1, 1).get_weights(), config={"cluster": 0})
    assert weights == [np.array([[1]]), np.array([1])]
    assert num_examples == 1
    assert metrics["cluster"] == 0


def test_cfl_client_get_weights():
    client = CFLClient(
        model_fn=lambda: lrmodel(1, 1),
        cid=0,
        x_train=np.array([[1.0]]),
        y_train=np.array([2.9]),
        x_test=np.array([[1.0]]),
        y_test=np.array([2.9]),
        epochs=1,
        batch_size=1,
    )
    weights = client.get_parameters(config={})
    assert np.array_equal(weights[0], np.array([[1]], dtype=np.float32),)
    assert np.array_equal(weights[1], np.array([1], dtype=np.float32),)
