"""Tests for strategy/wecfl.py"""
from typing import List, Tuple
from unittest.mock import MagicMock

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import Code, EvaluateRes, FitRes, Metrics, NDArrays, Status
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from inflorescence.client import FlowerClient
from inflorescence.strategy.wecfl import WeCFL
from inflorescence.model import Model
from tests.test_cfl import CustomClientProxy
from tests.test_ifca import weighted_average


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


def test_configure_fit_odd_round():
    init_params = [np.array([[1.0, 2.0, 3.0]]), np.array([[3.0, 1.0, 2.0]])]
    strategy = WeCFL(k_clusters=2, initial_parameters=ndarrays_to_parameters(init_params))
    clim = SimpleClientManager()
    for i in range(3):
        clim.register(CustomClientProxy(str(i)))
    config = strategy.configure_fit(server_round=3, parameters=init_params, client_manager=clim)

    for c in config:
        assert c[1].parameters == init_params
        assert c[1].config == {"server_round": 3}


def test_configure_fit_even_round():
    init_params = [np.array([[1.0, 2.0, 3.0]]), np.array([[3.0, 1.0, 2.0]])]
    strategy = WeCFL(k_clusters=2, initial_parameters=ndarrays_to_parameters(init_params))
    clim = SimpleClientManager()
    for i in range(3):
        clim.register(CustomClientProxy(str(i)))
    config1 = strategy.configure_fit(server_round=1, parameters=init_params, client_manager=clim)
    strategy.cluster_members = {0: 0, 1: 0, 2: 1}
    config2 = strategy.configure_fit(server_round=2, parameters=init_params, client_manager=clim)
    for c in config2:
        assert c[1].parameters == init_params
        assert c[1].config == {
            "server_round": 2,
            "cluster": strategy.cluster_members[int(c[0].cid)],
        }


def test_configure_fit_two_rounds():
    init_weights = [lrmodel(1, 1).get_weights(), lrmodel(1.1, 1.2).get_weights()]
    init_weights_flat = [_ for _ in init_weights for _ in _]
    strategy = WeCFL(k_clusters=2, initial_parameters=ndarrays_to_parameters(init_weights_flat))
    clim = SimpleClientManager()
    client = FlowerClient(
        model_fn=lambda: lrmodel(1, 1),
        cid=0,
        x_train=np.array([]),
        x_test=np.array([]),
        y_train=np.array([]),
        y_test=np.array([]),
        epochs=1,
        batch_size=1,
    )

    for i in range(3):
        clim.register(CustomClientProxy(str(i)))
    config1 = strategy.configure_fit(
        server_round=1, parameters=init_weights_flat, client_manager=clim
    )
    ok = Status(code=Code.OK, message="OK")
    params, _, metrics = client.fit(init_weights_flat, config=config1[0][1].config)
    params_aggregated, metrics_aggregated = strategy.aggregate_fit(
        server_round=1,
        results=[
            (
                clim.clients["0"],
                FitRes(
                    status=ok,
                    parameters=ndarrays_to_parameters(params),
                    num_examples=1,
                    metrics=metrics,
                ),
            ),
            (
                clim.clients["1"],
                FitRes(
                    status=ok,
                    parameters=ndarrays_to_parameters(params),
                    num_examples=1,
                    metrics=metrics,
                ),
            ),
            (
                clim.clients["2"],
                FitRes(
                    status=ok,
                    parameters=ndarrays_to_parameters(init_weights[1]),
                    num_examples=1,
                    metrics={"cluster": 1},
                ),
            ),
        ],
        failures=[],
    )
    config2 = strategy.configure_fit(
        server_round=2, parameters=init_weights_flat, client_manager=clim
    )
    assert strategy.clusters[0] == 0
    assert strategy.clusters[1] == 0
    assert strategy.clusters[2] == 1


def test_wecfl_aggregate_fit_works_with_layers():
    """Test that only params from clients in the same cluster are averaged together.
    Same behavior as IFCA for odd rounds.
    """
    # two initial models with one layer and one parameter each
    initial_weights = [
        np.array([[1.0, 2.0], [2.0, 3.0]]),
        np.array([[3.0, 4.0], [4.0, 5.0]]),
    ]
    initial_params = ndarrays_to_parameters(initial_weights)
    strategy = WeCFL(
        k_clusters=2,
        initial_parameters=initial_params,
        fit_metrics_aggregation_fn=weighted_average,
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
    results: List[Tuple[ClientProxy, FitRes]] = [
        client1res,
        client2res,
        client3res,
        client4res,
    ]
    actuals, metrics = strategy.aggregate_fit(server_round=1, results=results, failures=[])
    assert actuals == initial_params
    assert np.isclose(metrics["accuracy"], 0.8)
    assert np.isclose(metrics["accuracy_cluster_00"], 0.77)
    assert np.isclose(metrics["accuracy_cluster_01"], 0.83)
