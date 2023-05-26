"""Experiments on the Adult dataset."""
import os
from datetime import datetime
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import Metrics, NDArray, NDArrays, Scalar
from flwr.server.strategy import FedAvg
from loguru import logger
from torchmetrics import Accuracy, ConfusionMatrix

from experiments.model import TorchTabularModel
from experiments.torch_classifier import MLP, TorchClassifier
from inflorescence.dataset import AdultDataset, Dataset
from inflorescence.experiment import Experiment
from inflorescence.metrics import aggregate_binary_cm
from inflorescence.model import Model
from inflorescence.strategy import CFL, FLHC, IFCA, WeCFL


class AdultMLP(TorchTabularModel):
    def __init__(
        self,
        num_features: int = 61,
        learning_rate: float = 0.001,
        hidden_units: int = 100,
        dropout: float = 0.2,
        metrics: Optional[Dict[str, Callable[..., Scalar]]] = None,
        group_metrics: Optional[Dict[str, Callable[..., Scalar]]] = None,
        seed: Optional[int] = None,
    ):
        """An MLP model for classifying the UCI Adult dataset."""
        self.num_features = num_features
        self.learning_rate = learning_rate
        module = TorchClassifier(
            model=MLP(self.num_features, hidden_units=hidden_units, dropout=dropout),
            criterion=F.binary_cross_entropy,
            metrics=metrics,
            group_metrics=group_metrics,
            optimizer_class=torch.optim.SGD,
            lr=self.learning_rate,
            seed=seed,
        )
        super().__init__(module)


def run_ifca(
    k_clusters,
    num_clients,
    num_rounds,
    epochs,
    batch_size,
    concentration,
    seed,
    common_params,
):
    adult = AdultDataset()
    initial_weights = [AdultMLP(seed=seed + i).get_weights() for i in range(k_clusters)]
    initial_weights_flat = [_ for _ in initial_weights for _ in _]
    strategy = IFCA(
        k_clusters=k_clusters,
        initial_parameters=ndarrays_to_parameters(initial_weights_flat),
        **common_params,
    )
    metrics = {"acc": Accuracy(task="binary"), "cm": ConfusionMatrix(task="binary")}
    experiment = Experiment(
        strategies=[strategy],
        model_fn=AdultMLP,
        dataset=adult,
        group_var="sex",
        metrics=metrics,
        group_metrics=metrics,
        concentrations=[concentration],
    )
    dt = datetime.now()
    logger.info(f"Starting experiment for Adult with seed {seed}")
    experiment.run(
        num_rounds=num_rounds,
        num_clients=num_clients,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    dtend = datetime.now()
    span = dtend - dt
    dts = dt.strftime("%Y-%m-%d_%H%M%S")
    fname = f"results/adult_ifca_seed_{seed}_concentration_{concentration:.02f}_{k_clusters}_{dts}.parquet"
    logger.info("Finished experiments for Adult IFCA in {}. Saving results to {}", span, fname)
    df = experiment.to_pandas()
    os.makedirs("results", exist_ok=True)
    df.to_parquet(fname)


def run_wecfl(
    k_clusters,
    num_clients,
    num_rounds,
    epochs,
    batch_size,
    concentration,
    seed,
    common_params,
):
    adult = AdultDataset()
    initial_weights = [AdultMLP(seed=seed + i).get_weights() for i in range(k_clusters)]
    initial_weights_flat = [_ for _ in initial_weights for _ in _]
    strategy = WeCFL(
        k_clusters=k_clusters,
        initial_parameters=ndarrays_to_parameters(initial_weights_flat),
        **common_params,
    )
    metrics = {"acc": Accuracy(task="binary"), "cm": ConfusionMatrix(task="binary")}
    experiment = Experiment(
        strategies=[strategy],
        model_fn=AdultMLP,
        dataset=adult,
        group_var="sex",
        metrics=metrics,
        group_metrics=metrics,
        concentrations=[concentration],
    )
    dt = datetime.now()
    logger.info(f"Starting experiment for Adult WeCFL with seed {seed}")
    experiment.run(
        num_rounds=num_rounds,
        num_clients=num_clients,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    dtend = datetime.now()
    span = dtend - dt
    dts = dt.strftime("%Y-%m-%d_%H%M%S")
    fname = f"results/adult_wecfl_seed_{seed}_concentration_{concentration:.02f}_{k_clusters}_{dts}.parquet"
    logger.info("Finished experiments for Adult WeCFL in {}. Saving results to {}", span, fname)
    df = experiment.to_pandas()
    os.makedirs("results", exist_ok=True)
    df.to_parquet(fname)


def run_fedavg(
    num_clients,
    num_rounds,
    epochs,
    batch_size,
    concentration,
    seed,
    common_params,
):
    adult = AdultDataset()
    initial_weights = AdultMLP(seed=seed).get_weights()
    strategy = FedAvg(
        initial_parameters=ndarrays_to_parameters(initial_weights),
        **common_params,
    )
    metrics = {"acc": Accuracy(task="binary"), "cm": ConfusionMatrix(task="binary")}
    experiment = Experiment(
        strategies=[strategy],
        model_fn=AdultMLP,
        dataset=adult,
        group_var="sex",
        metrics=metrics,
        group_metrics=metrics,
        concentrations=[concentration],
    )
    dt = datetime.now()
    logger.info(f"Starting experiment for Adult with seed {seed}")
    experiment.run(
        num_rounds=num_rounds,
        num_clients=num_clients,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    dtend = datetime.now()
    span = dtend - dt
    dts = dt.strftime("%Y-%m-%d_%H%M%S")
    fname = f"results/adult_fedavg_seed_{seed}_concentration_{concentration:.02f}_{dts}.parquet"
    logger.info("Finished experiments for Adult FedAvg in {}. Saving results to {}", span, fname)
    df = experiment.to_pandas()
    os.makedirs("results", exist_ok=True)
    df.to_parquet(fname)


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/adult.log", format="{time:YYYY-MM-DDTHH:mm:ss} | {level} | {message}")
    NUM_ROUNDS = 20
    NUM_CLIENTS = 10
    EPOCHS = 10
    BATCH_SIZE = 64
    K_CLUSTERS = 5
    SEEDS = [43, 59, 67, 79, 97]
    CONCENTRATIONS = np.logspace(-2, 3, 6)
    common_params = dict(
        accept_failures=False,
        fraction_fit=1.0,  # Proportion of clients to sample in each training round
        fraction_evaluate=1.0,  # Proportion of clients to calculate accuracy on after each round
        min_fit_clients=NUM_CLIENTS,  # Minimum number of clients to train on in each round
        min_evaluate_clients=NUM_CLIENTS,  # Minimum number of clients to evaluate accuracy on after each round
        min_available_clients=NUM_CLIENTS,  # Minimum number of available clients needed to start a round
        evaluate_metrics_aggregation_fn=aggregate_binary_cm,  # <-- pass the metric aggregation function
    )
    for seed in SEEDS:
        for concentration in CONCENTRATIONS:
            run_ifca(
                k_clusters=K_CLUSTERS,
                num_clients=NUM_CLIENTS,
                num_rounds=NUM_ROUNDS,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                seed=seed,
                concentration=concentration,
                common_params=common_params,
            )
            run_fedavg(
                num_clients=NUM_CLIENTS,
                num_rounds=NUM_ROUNDS,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                seed=seed,
                concentration=concentration,
                common_params=common_params,
            )
            run_wecfl(
                k_clusters=K_CLUSTERS,
                num_clients=NUM_CLIENTS,
                num_rounds=NUM_ROUNDS,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                seed=seed,
                concentration=concentration,
                common_params=common_params,
            )

