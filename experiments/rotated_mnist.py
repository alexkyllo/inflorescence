import os
import platform
import time
from datetime import datetime
from typing import List, Optional, Tuple, Union
from urllib.parse import urljoin
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf
from flwr.common import ndarrays_to_parameters
from flwr.common.typing import NDArray
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation
from keras.utils.np_utils import to_categorical
from loguru import logger
from numpy.random import Generator
from tensorflow.keras import Input, Sequential, layers, metrics
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

from inflorescence.dataset import Dataset
from inflorescence.dataset.common import XY
from inflorescence.experiment import Experiment
from inflorescence.metrics import aggregate_binary_cm, weighted_average
from inflorescence.model import Model
from inflorescence.strategy import CFL, FLHC, IFCA, WeCFL

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class KerasCNN(Model):
    def __init__(self, learning_rate: float = 0.001, seed=None, **kwargs):
        """"""
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if seed:
            tf.random.set_seed(seed)
        dropout_rate = 0.5
        optimizer = Adam(learning_rate=learning_rate)
        model = Sequential()
        model.add(Input(shape=(28, 28, 1)))
        model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(units=10, activation="softmax"))

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=[
                "sparse_categorical_accuracy",
            ],
        )
        self.model = model

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def fit(self, X_train, y_train, epochs, batch_size, **kwargs):
        """"""
        y = y_train[:, 0]
        history = self.model.fit(X_train, y, batch_size=batch_size, epochs=epochs, verbose=2)
        return history.history["loss"][-1]

    def evaluate(self, X_test, y_test, batch_size=32):
        """"""
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        start_time = time.time()
        epoch_loss = 0.0
        steps = 0
        num_classes = self.model.layers[-1].output_shape[-1]
        cm = tf.zeros((num_classes, num_classes), dtype=tf.dtypes.int32)
        group_cms = {}
        group_correct = {}
        group_n = {}
        for x_batch, labels in test_dataset:
            steps += 1
            y = labels[:, 0]
            g = labels[:, 1]
            with tf.GradientTape() as tape:
                preds = self.model(x_batch, training=False)
                loss_value = loss_fn(y_true=y, y_pred=preds)
                epoch_loss += loss_value
            grads = tape.gradient(loss_value, self.model.trainable_weights)
            batch_cm = tf.math.confusion_matrix(
                y, tf.argmax(preds, axis=1), num_classes=num_classes
            )
            groups = tf.sort(tf.unique(g)[0])
            for group in groups:
                group_num = group.numpy()
                batch_group_cm = tf.math.confusion_matrix(
                    y[g == group], tf.argmax(preds[g == group], axis=1), num_classes=num_classes
                )
                group_cms.setdefault(
                    group_num, tf.zeros((num_classes, num_classes), dtype=tf.dtypes.int32)
                )
                group_cms[group_num] += batch_group_cm
                # TODO: log per-group accuracy here
                diag_sum = tf.reduce_sum(tf.linalg.tensor_diag_part(batch_group_cm))
                cm_sum = tf.reduce_sum(batch_group_cm)
                group_correct.setdefault(group_num, 0)
                group_n.setdefault(group_num, 0)
                group_correct[group_num] += diag_sum
                group_n[group_num] += cm_sum
            cm += batch_cm
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            test_acc_metric.update_state(y_true=y, y_pred=preds)
        valid_acc = test_acc_metric.result()
        print("Validation acc: %.4f" % (float(valid_acc),))
        test_acc_metric.reset_states()
        # process confusion matrix
        cm_dict = {}
        for i in range(num_classes):
            for j in range(num_classes):
                cm_dict[f"valid_cm_{i:02}_{j:02}"] = cm[i][j].numpy().astype(float)
                for k, gcm in group_cms.items():
                    cm_dict[f"valid_cm_{i:02}_{j:02}_group_{k:02}"] = (
                        gcm[i][j].numpy().astype(float)
                    )
        epoch_loss = epoch_loss / steps
        print("Time taken: %.2fs" % (time.time() - start_time))
        metrics_dict = {"valid_acc": valid_acc.numpy().astype(float)}
        for g, n in group_n.items():
            correct = group_correct[g]
            metrics_dict[f"valid_acc_group_{g:02}"] = (correct / n).numpy().astype(float)
        metrics_dict.update(cm_dict)
        return epoch_loss.numpy().astype(float), metrics_dict


class RotatedMNIST(Dataset):
    def __init__(self, root_dir=None):
        self.root_dir = root_dir
        self.download()

    def download(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data(path=self.root_dir)
        x_train = x_train.astype(np.float32) / 255
        x_test = x_test.astype(np.float32) / 255
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        self.x = np.vstack([x_train, x_test])
        self.y = np.hstack([y_train, y_test])
        self.shuffle()
        self.rotate()

    def shuffle(self, seed=None):
        idx = np.arange(70000)
        np.random.default_rng(seed=seed).shuffle(idx)
        self.x = self.x[idx]
        self.y = self.y[idx]

    def split_non_iid(
        self,
        concentration: float,
        num_clients: int,
        group_var: str = "rotation",
        test_size: float = 0.2,
        seed=None,
    ) -> Tuple[List[Tuple[XY, XY]], NDArray]:
        return super().split_non_iid(concentration, num_clients, group_var, test_size, seed)

    def rotate(self):
        """Rotate the clients' samples and add rotation # to y labels"""
        for i, ex in enumerate(self.x):
            # 3/8 0deg, 1/4 90deg, 1/8 180 deg, 1/4 270deg
            self.x[i] = np.rot90(ex, ((((i - 2) % 8) > 0) * ((i % 4))))
        idxs = np.arange(70000)
        rot_labels = (((idxs - 2) % 8) > 0) * ((idxs % 4))
        self.y = np.stack([self.y, rot_labels], axis=1)

    def __getitem__(self, item):
        pass

    def get_xy(self, group_var=None):
        return self.x, self.y

def run_ifca(
    k_clusters,
    num_clients,
    num_rounds,
    epochs,
    batch_size,
    concentration,
    seed,
    num_cpus,
    num_gpus,
    common_params,
):
    initial_weights = []
    with tf.device("/device:cpu:0"):
        for i in range(k_clusters):
            initial_model = KerasCNN(seed=seed + i)
            initial_weights.append(initial_model.get_weights())
            del initial_model
    initial_weights_flat = [_ for _ in initial_weights for _ in _]
    strategy = IFCA(
        k_clusters=k_clusters,
        initial_parameters=ndarrays_to_parameters(initial_weights_flat),
        **common_params,
    )
    dt = datetime.now()
    logger.info("Starting experiments for RotatedMNIST with IFCA(k_clusters={})", k_clusters)
    experiment = Experiment(
        strategies=[strategy],
        model_fn=KerasCNN,
        dataset=rmnist,
        group_var="rotation",
        concentrations=[concentration],
    )

    experiment.run(
        num_rounds=num_rounds,
        num_clients=num_clients,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
    )
    dtend = datetime.now()
    span = dtend - dt
    dts = dt.strftime("%Y-%m-%d_%H%M%S")
    fname = (
        f"results/rotatedmnistimb_ifca_seed_{seed}_{concentration:.02f}_{k_clusters}_{dts}.parquet"
    )
    logger.info("Finished RotatedMNIST IFCA experiment in {}. Saving results to {}", span, fname)
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
    num_cpus,
    num_gpus,
    common_params,
):
    initial_weights = []
    with tf.device("/device:cpu:0"):
        for i in range(k_clusters):
            initial_model = KerasCNN(seed=seed + i)
            initial_weights.append(initial_model.get_weights())
            del initial_model
    strategy = WeCFL(
        k_clusters=k_clusters,
        initial_parameters=ndarrays_to_parameters(initial_weights[0]),
        **common_params,
    )
    dt = datetime.now()
    logger.info("Starting experiments for RotatedMNIST with WeCFL(k_clusters={})", k_clusters)
    experiment = Experiment(
        strategies=[strategy],
        model_fn=KerasCNN,
        dataset=rmnist,
        group_var="rotation",
        concentrations=[concentration],
    )

    experiment.run(
        num_rounds=num_rounds,
        num_clients=num_clients,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
    )
    dtend = datetime.now()
    span = dtend - dt
    dts = dt.strftime("%Y-%m-%d_%H%M%S")
    fname = (
        f"results/rotatedmnistimb_wecfl_seed_{seed}_{concentration:.02f}_{k_clusters}_{dts}.parquet"
    )
    logger.info("Finished RotatedMNIST WeCFL experiment in {}. Saving results to {}", span, fname)
    df = experiment.to_pandas()
    os.makedirs("results", exist_ok=True)
    df.to_parquet(fname)


def run_flhck(
    k_clusters,
    num_clients,
    num_rounds,
    epochs,
    batch_size,
    concentration,
    seed,
    num_cpus,
    num_gpus,
    common_params,
):
    with tf.device("/device:cpu:0"):
        initial_model = KerasCNN(seed=seed)
        initial_weights = initial_model.get_weights()
        del initial_model
    strategy = FLHC(
        k_clusters=k_clusters,
        metric="euclidean",
        linkage="complete",
        cluster_after_round=5,
        initial_parameters=ndarrays_to_parameters(initial_weights),
        **common_params,
    )
    dt = datetime.now()
    logger.info("Starting experiments for RotatedMNIST with FLHC(k_clusters={})", k_clusters)
    experiment = Experiment(
        strategy=strategy,
        model_fn=KerasCNN,
        dataset=rmnist,
        group_var="rotation",
        concentrations=[concentration],
    )

    experiment.run(
        num_rounds=num_rounds,
        num_clients=num_clients,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
    )
    dtend = datetime.now()
    span = dtend - dt
    dts = dt.strftime("%Y-%m-%d_%H%M%S")
    fname = f"results/rotatedmnistimb_flhck_seed_{seed}_{concentration:.02f}_{k_clusters:02d}_{dts}.parquet"
    logger.info("Finished RotatedMNIST FLHC experiment in {}. Saving results to {}", span, fname)
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
    num_cpus,
    num_gpus,
    common_params,
):
    with tf.device("/device:cpu:0"):
        initial_model = KerasCNN(seed=seed)
        initial_weights = initial_model.get_weights()
        del initial_model
    strategy = FedAvg(
        initial_parameters=ndarrays_to_parameters(initial_weights),
        **common_params,
    )
    dt = datetime.now()
    logger.info("Starting experiments for RotatedMNIST with FedAvg")
    experiment = Experiment(
        strategies=[strategy],
        model_fn=KerasCNN,
        dataset=rmnist,
        group_var="rotation",
        concentrations=[concentration],
    )

    experiment.run(
        num_rounds=num_rounds,
        num_clients=num_clients,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
    )
    dtend = datetime.now()
    span = dtend - dt
    dts = dt.strftime("%Y-%m-%d_%H%M%S")
    fname = f"results/rotatedmnistimb_fedavg_seed_{seed}_{concentration:.02f}_{dts}.parquet"
    logger.info("Finished RotatedMNIST fedavg experiment in {}. Saving results to {}", span, fname)
    df = experiment.to_pandas()
    os.makedirs("results", exist_ok=True)
    df.to_parquet(fname)


if __name__ == "__main__":
    NUM_ROUNDS = 10
    BATCH_SIZE = 64
    EPOCHS = 10
    NUM_CLIENTS = 10
    rmnist = RotatedMNIST()
    SEEDS = [43, 59, 67, 79, 97]
    CONCENTRATIONS = np.logspace(-2, 3, 6)
    K_CLUSTERS = 5
    common_params = dict(
        accept_failures=False,
        fraction_fit=1.0,  # Proportion of clients to sample in each training round
        fraction_evaluate=1.0,  # Proportion of clients to calculate accuracy on after each round
        min_fit_clients=NUM_CLIENTS,  # Minimum number of clients to train on in each round
        min_evaluate_clients=NUM_CLIENTS,  # Minimum number of clients to evaluate accuracy on after each round
        min_available_clients=NUM_CLIENTS,  # Minimum number of available clients needed to start a round
        evaluate_metrics_aggregation_fn=aggregate_binary_cm,  # <-- pass the metric aggregation function
    )
    os.makedirs("logs", exist_ok=True)
    logger.add(
        "logs/rotatedmnistimb.log", format="{time:YYYY-MM-DDTHH:mm:ss} | {level} | {message}"
    )

    for seed in SEEDS:
        for concentration in CONCENTRATIONS:
            logger.info(
                "Starting experiments for RotatedMNIST, seed = {}, concentration = {}",
                seed,
                concentration,
            )
            run_fedavg(
                num_clients=NUM_CLIENTS,
                num_rounds=NUM_ROUNDS,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                seed=seed,
                concentration=concentration,
                num_cpus=8,
                num_gpus=0,
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
                num_cpus=8,
                num_gpus=0,
                common_params=common_params,
            )
            run_ifca(
                k_clusters=K_CLUSTERS,
                num_clients=NUM_CLIENTS,
                num_rounds=NUM_ROUNDS,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                seed=seed,
                concentration=concentration,
                num_cpus=8,
                num_gpus=0,
                common_params=common_params,
            )

            run_flhck(
                k_clusters=K_CLUSTERS,
                num_clients=NUM_CLIENTS,
                num_rounds=NUM_ROUNDS,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                seed=seed,
                concentration=concentration,
                num_cpus=8,
                num_gpus=0,
                common_params=common_params,
            )
