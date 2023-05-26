"""Experiments on the COMPAS dataset."""
from datetime import datetime
from typing import Callable, Dict, Optional

import numpy as np
import torch
from flwr.common import ndarrays_to_parameters
from flwr.common.typing import Scalar
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation
from loguru import logger
from torch.nn import functional as F
from torchmetrics import Accuracy, ConfusionMatrix
from torchmetrics.functional.classification.accuracy import binary_accuracy

from experiments.model import TorchTabularModel
from experiments.torch_classifier import MLP, LogisticRegression, TorchClassifier
from inflorescence.dataset import Compas
from inflorescence.experiment import Experiment
from inflorescence.metrics import aggregate_binary_cm, weighted_average
from inflorescence.strategy import CFL, FLHC, IFCA, WeCFL


class CompasMLP(TorchTabularModel):
    """"""

    def __init__(
        self,
        num_features: int = 24,
        learning_rate: float = 0.001,
        hidden_units: int = 100,
        dropout: float = 0.5,
        metrics: Optional[Dict[str, Callable[..., Scalar]]] = None,
        group_metrics: Optional[Dict[str, Callable[..., Scalar]]] = None,
        seed: Optional[int] = None,
    ):
        """An MLP model for classifying the COMPAS dataset."""
        self.num_features = num_features
        self.learning_rate = learning_rate
        module = TorchClassifier(
            model=LogisticRegression(self.num_features),
            criterion=F.binary_cross_entropy,
            metrics=metrics,
            group_metrics=group_metrics,
            optimizer_class=torch.optim.SGD,
            lr=self.learning_rate,
            seed=seed,
        )
        super().__init__(module)


def centralized():
    compas = Compas()
    x, y = compas.get_xy()
    mlp = CompasMLP(
        hidden_units=100,
        learning_rate=0.01,
        dropout=0.5,
        metrics={"acc": Accuracy(task="binary"), "cm": ConfusionMatrix(task="binary")},
        group_metrics={"acc": Accuracy(task="binary"), "cm": ConfusionMatrix(task="binary")},
    )
    from sklearn.model_selection import train_test_split

    (x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2)
    accs = []
    losses = []
    for i in range(20):
        print(f"epoch {i}")
        mlp.fit(x_train, y_train, epochs=1)
        loss, metrics_dict = mlp.evaluate(x_test, y_test)
        accs.append(metrics_dict["valid_acc"])
        losses.append(loss)
    import matplotlib.pyplot as plt

    plt.plot(accs)
    plt.ylim(0.5, 0.75)
    plt.show()
    max(accs)


if __name__ == "__main__":
    NUM_ROUNDS = 20
    NUM_CLIENTS = 10
    EPOCHS = 10
    BATCH_SIZE = 64
    K_CLUSTERS = 5
    SEEDS = [43, 59, 67, 79, 97]
    CONCENTRATIONS = np.logspace(-2, 3, 6)
    logger.add(
        "logs/compasaalogistic.log", format="{time:YYYY-MM-DDTHH:mm:ss} | {level} | {message}"
    )
    for SEED in SEEDS:
        compas = Compas()
        initial_weights = [CompasMLP(seed=SEED + i).get_weights() for i in range(K_CLUSTERS)]
        initial_weights_flat = [_ for _ in initial_weights for _ in _]
        logger.debug("initial_weight: {}", [i[0][0][0] for i in initial_weights])
        common_params = dict(
            accept_failures=False,
            fraction_fit=1.0,  # Proportion of clients to sample in each training round
            fraction_evaluate=1.0,  # Proportion of clients to calculate accuracy on after each round
            min_fit_clients=NUM_CLIENTS,  # Minimum number of clients to train on in each round
            min_evaluate_clients=NUM_CLIENTS,  # Minimum number of clients to evaluate accuracy on after each round
            min_available_clients=NUM_CLIENTS,  # Minimum number of available clients needed to start a round
            evaluate_metrics_aggregation_fn=aggregate_binary_cm,  # <-- pass the metric aggregation function
        )

        metrics = {"acc": Accuracy(task="binary"), "cm": ConfusionMatrix(task="binary")}
        for concentration in CONCENTRATIONS:
            strategy = IFCA(
                k_clusters=K_CLUSTERS,
                initial_parameters=ndarrays_to_parameters(initial_weights_flat),
                **common_params,
            )
            experiment = Experiment(
                strategies=[strategy],
                model_fn=CompasMLP,
                dataset=compas,
                group_var="race_African-American",
                metrics=metrics,
                group_metrics=metrics,
                concentrations=[concentration],
            )
            dt = datetime.now()
            logger.info("Starting experiments for Compas with seed {}", SEED)
            experiment.run(
                num_rounds=NUM_ROUNDS,
                num_clients=NUM_CLIENTS,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                seed=SEED,
            )
            dtend = datetime.now()
            dts = dt.strftime("%Y-%m-%d_%H%M%S")
            fname = f"results/compasaalogistic_ifca_seed_{SEED}_concentration_{concentration:.02f}_{dts}.parquet"
            logger.info("Finished experiments. Saving results to {}", fname)
            df = experiment.to_pandas()
            df.to_parquet(fname)
            # FedAvg
            strategy = FedAvg(
                initial_parameters=ndarrays_to_parameters(initial_weights[0]), **common_params
            )
            experiment = Experiment(
                strategies=[strategy],
                model_fn=CompasMLP,
                dataset=compas,
                group_var="race_African-American",
                metrics=metrics,
                group_metrics=metrics,
                concentrations=[concentration],
            )
            dt = datetime.now()
            logger.info("Starting experiments for Compas with seed {}", SEED)
            experiment.run(
                num_rounds=NUM_ROUNDS,
                num_clients=NUM_CLIENTS,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                seed=SEED,
            )
            dtend = datetime.now()
            dts = dt.strftime("%Y-%m-%d_%H%M%S")
            fname = f"results/compasaalogistic_fedavg_seed_{SEED}_concentration_{concentration:.02f}_{dts}.parquet"
            logger.info("Finished experiments. Saving results to {}", fname)
            df = experiment.to_pandas()
            df.to_parquet(fname)
            # WeCFL
            strategy = WeCFL(
                k_clusters=K_CLUSTERS,
                initial_parameters=ndarrays_to_parameters(initial_weights_flat),
                **common_params,
            )
            experiment = Experiment(
                strategies=[strategy],
                model_fn=CompasMLP,
                dataset=compas,
                group_var="race_African-American",
                metrics=metrics,
                group_metrics=metrics,
                concentrations=[concentration],
            )
            dt = datetime.now()
            logger.info("Starting experiments for Compas with seed {}", SEED)
            experiment.run(
                num_rounds=NUM_ROUNDS,
                num_clients=NUM_CLIENTS,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                seed=SEED,
            )
            dtend = datetime.now()
            dts = dt.strftime("%Y-%m-%d_%H%M%S")
            fname = f"results/compasaalogistic_wecfl_seed_{SEED}_concentration_{concentration:.02f}_{K_CLUSTERS}_{dts}.parquet"
            logger.info("Finished experiments. Saving results to {}", fname)
            df = experiment.to_pandas()
            df.to_parquet(fname)

            strategy = FLHC(
                k_clusters=k,
                metric="euclidean",
                linkage="complete",
                cluster_after_round=5,
                initial_parameters=ndarrays_to_parameters(initial_weights[0]),
                **common_params,
            )
            experiment = Experiment(
                # strategies=[fedavg, ifca, flhc, cfl],  # TODO: wecfl
                # strategies=[wecfl],
                strategies=[strategy],
                model_fn=CompasMLP,
                dataset=compas,
                group_var="race_African-American",
                metrics=metrics,
                group_metrics=metrics,
                concentrations=[concentration],
            )
            dt = datetime.now()
            logger.info(
                "Starting experiments for Compas with FLHC(k_clusters={}) seed {}", k, SEED
            )
            experiment.run(
                num_rounds=NUM_ROUNDS,
                num_clients=NUM_CLIENTS,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                seed=SEED,
            )
            dtend = datetime.now()
            dts = dt.strftime("%Y-%m-%d_%H%M%S")
            fname = f"results/compasaalogistic_flhck_seed_{SEED}_concentration_{concentration:.02f}_{k:02d}_{dts}.parquet"
            logger.info("Finished experiments. Saving results to {}", fname)
            df = experiment.to_pandas()
            df.to_parquet(fname)
            
            for eps_1, eps_2 in [(0.03, 0.04)]:  # (0.4, 1.6), (0.4, 0.8)]:
                strategy = CFL(
                    eps_1=eps_1,
                    eps_2=eps_2,
                    cluster_after_round=5,
                    k_clusters=1,
                    initial_parameters=ndarrays_to_parameters(initial_weights[0]),
                    **common_params,
                )
                experiment = Experiment(
                    # strategies=[fedavg, ifca, flhc, cfl],  # TODO: wecfl
                    # strategies=[wecfl],
                    strategies=[strategy],
                    model_fn=CompasMLP,
                    dataset=compas,
                    group_var="race_African-American",
                    metrics=metrics,
                    group_metrics=metrics,
                    concentrations=[concentration],
                )
                dt = datetime.now()
                logger.info("Starting experiments for Compas with seed {}", SEED)
                experiment.run(
                    num_rounds=NUM_ROUNDS,
                    num_clients=NUM_CLIENTS,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    seed=SEED,
                )
                dtend = datetime.now()
                dts = dt.strftime("%Y-%m-%d_%H%M%S")
                fname = f"compasaalogistic_cfl_seed_{SEED}_concentration_{concentration:.02f}_{eps_1:.02f}_{eps_2:.02f}_{dts}.parquet"
                logger.info("Finished experiments. Saving results to {}", fname)
                df = experiment.to_pandas()
                df.to_parquet(fname)
