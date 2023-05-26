"""Tests for strategy/ifca.py"""
from typing import List, Tuple
from unittest.mock import MagicMock

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import Code, EvaluateRes, FitRes, Metrics, Status
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy

from inflorescence.client.ifca_client import IFCAClient
from inflorescence.metrics import weighted_average
from inflorescence.model import Model
from inflorescence.strategy.ifca import IFCA


def eval_res(**kwargs):
    return (MagicMock(), EvaluateRes(status=Status(code=Code.OK, message="OK"), **kwargs))


def test_ifca_initializes_params():
    ifca = IFCA(
        k_clusters=2,
        initial_parameters=ndarrays_to_parameters(
            [np.random.normal(size=(2, 2)), np.random.normal(size=(2, 2))]
        ),
    )
    client_manager = SimpleClientManager()
    initial_params = parameters_to_ndarrays(
        ifca.initialize_parameters(client_manager=client_manager)
    )
    assert initial_params[0].shape == initial_params[1].shape == (2, 2)


def test_ifca_aggregate_fit_works():
    """Test that only params from clients in the same cluster are averaged together."""
    # two initial models with one layer and one parameter each
    ndarrays = [np.array([[1.0]]), np.array([[2.0]])]
    initial_params = ndarrays_to_parameters(ndarrays)
    ifca = IFCA(k_clusters=2, initial_parameters=initial_params)
    # ifca = FedAvg(initial_parameters=initial_params)
    ok = Status(code=Code.OK, message="OK")
    c1: Metrics = {"cluster": 0}
    c2: Metrics = {"cluster": 1}
    client1res = (
        MagicMock(),
        FitRes(
            status=ok,
            parameters=ndarrays_to_parameters(ndarrays[0]),
            num_examples=10,
            metrics=c1,
        ),
    )
    client2res = (
        MagicMock(),
        FitRes(
            status=ok,
            parameters=ndarrays_to_parameters(ndarrays[0]),
            num_examples=10,
            metrics=c1,
        ),
    )
    client3res = (
        MagicMock(),
        FitRes(
            status=ok,
            parameters=ndarrays_to_parameters(ndarrays[1]),
            num_examples=10,
            metrics=c2,
        ),
    )
    client4res = (
        MagicMock(),
        FitRes(
            status=ok,
            parameters=ndarrays_to_parameters(ndarrays[1]),
            num_examples=10,
            metrics=c2,
        ),
    )
    results: List[Tuple[ClientProxy, FitRes]] = [client1res, client2res, client3res, client4res]
    actuals, _ = ifca.aggregate_fit(server_round=1, results=results, failures=[])
    assert parameters_to_ndarrays(actuals) == ndarrays


def test_ifca_aggregate_fit_works_with_different_sample_sizes():
    """Test that only params from clients in the same cluster are averaged together."""
    # two initial models with one layer and one parameter each
    initial_weights = [np.array([[1.0]]), np.array([[2.0]])]
    initial_params = ndarrays_to_parameters(initial_weights)
    expected_weights = [np.array([[1.2]]), np.array([[2.4]])]
    expected_results = ndarrays_to_parameters(expected_weights)
    ifca = IFCA(k_clusters=2, initial_parameters=initial_params)
    # ifca = FedAvg(initial_parameters=initial_params)
    ok = Status(code=Code.OK, message="OK")
    c1: Metrics = {"cluster": 0}
    c2: Metrics = {"cluster": 1}
    client1res = (
        MagicMock(),
        FitRes(
            status=ok,
            parameters=ndarrays_to_parameters([np.array([[1.0]])]),
            num_examples=20,
            metrics=c1,
        ),
    )
    client2res = (
        MagicMock(),
        FitRes(
            status=ok,
            parameters=ndarrays_to_parameters([np.array([[2.0]])]),
            num_examples=5,
            metrics=c1,
        ),
    )
    client3res = (
        MagicMock(),
        FitRes(
            status=ok,
            parameters=ndarrays_to_parameters([np.array([[2.0]])]),
            num_examples=15,
            metrics=c2,
        ),
    )
    client4res = (
        MagicMock(),
        FitRes(
            status=ok,
            parameters=ndarrays_to_parameters([np.array([[3.0]])]),
            num_examples=10,
            metrics=c2,
        ),
    )
    results: List[Tuple[ClientProxy, FitRes]] = [client1res, client2res, client3res, client4res]
    actuals, _ = ifca.aggregate_fit(server_round=1, results=results, failures=[])
    assert actuals == expected_results


def test_ifca_aggregate_fit_works_with_layers():
    """Test that only params from clients in the same cluster are averaged together."""
    # two initial models with one layer and one parameter each
    initial_weights = [np.array([[1.0, 2.0], [2.0, 3.0]]), np.array([[3.0, 4.0], [4.0, 5.0]])]
    initial_params = ndarrays_to_parameters(initial_weights)
    ifca = IFCA(
        k_clusters=2, initial_parameters=initial_params, fit_metrics_aggregation_fn=weighted_average
    )
    # ifca = FedAvg(initial_parameters=initial_params)
    ok = Status(code=Code.OK, message="OK")
    client1res = (
        MagicMock(),
        FitRes(
            status=ok,
            parameters=ndarrays_to_parameters([initial_weights[0]]),
            num_examples=10,
            metrics={"cluster": 0, "accuracy": 0.78},
        ),
    )
    client2res = (
        MagicMock(),
        FitRes(
            status=ok,
            parameters=ndarrays_to_parameters([initial_weights[0]]),
            num_examples=10,
            metrics={"cluster": 0, "accuracy": 0.76},
        ),
    )
    client3res = (
        MagicMock(),
        FitRes(
            status=ok,
            parameters=ndarrays_to_parameters([initial_weights[1]]),
            num_examples=10,
            metrics={"cluster": 1, "accuracy": 0.82},
        ),
    )
    client4res = (
        MagicMock(),
        FitRes(
            status=ok,
            parameters=ndarrays_to_parameters([initial_weights[1]]),
            num_examples=10,
            metrics={"cluster": 1, "accuracy": 0.84},
        ),
    )
    results: List[Tuple[ClientProxy, FitRes]] = [client1res, client2res, client3res, client4res]
    actuals, metrics = ifca.aggregate_fit(server_round=1, results=results, failures=[])
    assert actuals == initial_params
    assert np.isclose(metrics["accuracy"], 0.8)
    assert np.isclose(metrics["accuracy_cluster_00"], 0.77)
    assert np.isclose(metrics["accuracy_cluster_01"], 0.83)


def test_ifca_aggregate_fit_works_with_nonhomogeneous_layers():
    """Test that only params from clients in the same cluster are averaged together."""
    # two initial models with one layer and one parameter each
    initial_weights = [np.array([[1.0]]), np.array([2.0]), np.array([[3.0]]), np.array([4.0])]
    initial_params = ndarrays_to_parameters(initial_weights)
    ifca = IFCA(
        k_clusters=2, initial_parameters=initial_params, fit_metrics_aggregation_fn=weighted_average
    )
    # ifca = FedAvg(initial_parameters=initial_params)
    ok = Status(code=Code.OK, message="OK")
    client1res = (
        MagicMock(),
        FitRes(
            status=ok,
            parameters=ndarrays_to_parameters(initial_weights[0:2]),
            num_examples=10,
            metrics={"cluster": 0, "accuracy": 0.78},
        ),
    )
    client2res = (
        MagicMock(),
        FitRes(
            status=ok,
            parameters=ndarrays_to_parameters(initial_weights[0:2]),
            num_examples=10,
            metrics={"cluster": 0, "accuracy": 0.76},
        ),
    )
    client3res = (
        MagicMock(),
        FitRes(
            status=ok,
            parameters=ndarrays_to_parameters(initial_weights[2:4]),
            num_examples=10,
            metrics={"cluster": 1, "accuracy": 0.82},
        ),
    )
    client4res = (
        MagicMock(),
        FitRes(
            status=ok,
            parameters=ndarrays_to_parameters(initial_weights[2:4]),
            num_examples=10,
            metrics={"cluster": 1, "accuracy": 0.84},
        ),
    )
    results: List[Tuple[ClientProxy, FitRes]] = [client1res, client2res, client3res, client4res]
    actuals, metrics = ifca.aggregate_fit(server_round=1, results=results, failures=[])
    assert actuals == initial_params
    assert np.isclose(metrics["accuracy"], 0.8)
    assert np.isclose(metrics["accuracy_cluster_00"], 0.77)
    assert np.isclose(metrics["accuracy_cluster_01"], 0.83)


def test_ifca_aggregate_evaluate_works_with_layers():
    """Test that losses are weighted averaged overall and cluster-wise."""
    initial_params = ndarrays_to_parameters([np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]])])
    ifca = IFCA(
        k_clusters=2,
        initial_parameters=initial_params,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    ok = Status(code=Code.OK, message="OK")

    client1res = (
        MagicMock(),
        EvaluateRes(
            status=ok,
            loss=0.2,
            num_examples=10,
            metrics={"cluster": 0, "accuracy": 0.78},
        ),
    )
    client2res = (
        MagicMock(),
        EvaluateRes(
            status=ok,
            loss=0.3,
            num_examples=10,
            metrics={"cluster": 0, "accuracy": 0.76},
        ),
    )
    client3res = (
        MagicMock(),
        EvaluateRes(
            status=ok,
            loss=0.4,
            num_examples=10,
            metrics={"cluster": 1, "accuracy": 0.82},
        ),
    )
    client4res = (
        MagicMock(),
        EvaluateRes(
            status=ok,
            loss=0.5,
            num_examples=10,
            metrics={"cluster": 1, "accuracy": 0.84},
        ),
    )
    results: List[Tuple[ClientProxy, EvaluateRes]] = [
        client1res,
        client2res,
        client3res,
        client4res,
    ]
    loss, metrics = ifca.aggregate_evaluate(server_round=1, results=results, failures=[])
    assert loss == 0.35
    assert metrics["loss_cluster_00"] == 0.25
    assert metrics["loss_cluster_01"] == 0.45
    assert metrics["accuracy"] == 0.8
    assert metrics["accuracy_cluster_00"] == 0.77
    assert np.isclose(metrics["accuracy_cluster_01"], 0.83)


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


def test_ifca_client_fit():
    client = IFCAClient(
        model_fn=lambda: lrmodel(1, 1),
        cid=0,
        x_train=np.array([[1.0]]),
        y_train=np.array([2.9]),
        x_test=np.array([[1.0]]),
        y_test=np.array([2.9]),
        epochs=1,
        batch_size=1,
    )
    weights, num_examples, metrics = client.fit(parameters=lrmodel(1, 1).get_weights(), config={})
    assert weights == [np.array([[1]]), np.array([1])]
    assert num_examples == 1
    assert metrics["cluster"] == 0


def test_ifca_client_evaluate():
    """IFCAClient should return evaluation result for the model with lowest loss."""
    client = IFCAClient(
        model_fn=lambda: lrmodel(1, 1),
        cid=0,
        x_train=None,
        y_train=None,
        x_test=np.array([[1.0]]),
        y_test=[2.9],
        epochs=1,
        batch_size=1,
    )
    loss, _, metrics = client.evaluate(
        parameters=[*lrmodel(1, 1).get_weights(), *lrmodel(2, 1).get_weights()], config={}
    )
    assert metrics["cluster"] == 1
    assert np.isclose(loss, 0.1)


def test_ifca_client_fit_cluster_metric():
    """IFCAClient.fit() should evaluate and return cluster #."""
    client = IFCAClient(
        model_fn=lambda: lrmodel(1, 1),
        cid=0,
        x_train=np.array([[1.0]]),
        y_train=[2.9],
        x_test=np.array([[1.0]]),
        y_test=[2.9],
        epochs=1,
        batch_size=1,
    )
    weights_1 = lrmodel(1, 1).get_weights()
    weights_2 = lrmodel(2, 1).get_weights()
    params, _, metrics = client.fit(parameters=[*weights_1, *weights_2], config={})
    assert metrics["cluster"] == 1
    assert np.array_equal(params[0], np.array([[2]]))
    assert np.array_equal(params[1], np.array([1]))


class DummyModel:
    def __init__(self, weights):
        self.weights = weights

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def fit(self, *args, **kwargs):
        return


def test_ifca_client_set_weights():
    weights_1 = [
        np.array(
            [[0.08, -0.04, 0.006]],
            dtype=np.float32,
        ),
        np.array([0.03], dtype=np.float32),
    ]
    weights_2 = [
        np.array(
            [[0.09, -0.02, 0.004]],
            dtype=np.float32,
        ),
        np.array([0.04], dtype=np.float32),
    ]
    client = IFCAClient(
        model_fn=lambda: DummyModel(weights_1),
        cid=0,
        x_train=np.array([[1.0]]),
        y_train=[2.9],
        x_test=np.array([[1.0]]),
        y_test=[2.9],
        epochs=1,
        batch_size=1,
    )
    client.set_weights([*weights_1, *weights_2])
    # breakpoint()
    assert client.models[0].get_weights() == weights_1
    assert client.models[1].get_weights() == weights_2


def test_aggregate_client_specific_metrics():
    """Test that metrics are averaged over clients that reported that metric."""
    initial_params = ndarrays_to_parameters([np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]])])
    ifca = IFCA(
        k_clusters=2,
        initial_parameters=initial_params,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    client1res = eval_res(
        loss=0.2, num_examples=10, metrics={"cluster": 0, "accuracy": 0.78, "precision": 0.5}
    )
    client2res = eval_res(
        loss=0.3, num_examples=10, metrics={"cluster": 0, "accuracy": 0.76, "precision": 0.7}
    )
    client3res = eval_res(loss=0.4, num_examples=10, metrics={"cluster": 0, "accuracy": 0.82})
    client4res = eval_res(loss=0.5, num_examples=10, metrics={"cluster": 1, "accuracy": 0.84})
    results: List[Tuple[ClientProxy, EvaluateRes]] = [
        client1res,
        client2res,
        client3res,
        client4res,
    ]
    _, metrics = ifca.aggregate_evaluate(server_round=1, results=results, failures=[])
    assert metrics["precision_cluster_00"] == 0.6


# TODO: test empty clusters case

# def test_ifca_server_client_integration():
#     """IFCA works end to end in simulated training."""
#     client = IFCAClient(
#         model_fn=lambda: lrmodel(1, 1),
#         cid=0,
#         x_train=np.arange(20),
#         y_train=np.arange(20),
#         x_test=np.arange(20),
#         y_test=np.arange(20),
#         epochs=1,
#         batch_size=1,
#     )
#     params = [
#         *lrmodel(1, 1).get_weights(),
#         *lrmodel(2, 2).get_weights(),
#         *lrmodel(1, 1).get_weights(),
#         *lrmodel(2, 2).get_weights(),
#     ]
#     strategy = IFCA(
#         k_clusters=2,
#         initial_parameters=params,
#         fraction_fit=1.0,  # Proportion of clients to sample in each training round
#         fraction_evaluate=0.5,  # Proportion of clients to calculate accuracy on after each round
#         min_fit_clients=8,  # Minimum number of clients to train on in each round
#         min_evaluate_clients=4,  # Minimum number of clients to evaluate accuracy on after each round
#         min_available_clients=8,
#     )
#     new_params, num_examples, metrics = client.fit(params, {})
#     ok = Status(code=Code.OK, message="OK")
#     params_agg, metrics_agg = strategy.aggregate_fit(
#         server_round=0,
#         results=[
#             (
#                 MagicMock(),
#                 FitRes(
#                     status=ok,
#                     parameters=ndarrays_to_parameters(new_params),
#                     num_examples=num_examples,
#                     metrics=metrics,
#                 ),
#             )
#         ],
#     )
