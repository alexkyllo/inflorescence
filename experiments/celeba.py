"""Experiments on the CelebA dataset."""
from datetime import datetime

import numpy as np
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from loguru import logger
from torch import Generator, optim
from torch.nn import functional as F
from torch.utils.data import Subset, random_split
from torchmetrics import Accuracy, ConfusionMatrix
from torchvision import transforms
from torchvision.datasets import CelebA

from experiments.model import TorchVisionModel
from experiments.torch_classifier import CNN, TorchClassifier
from inflorescence.dataset import Dataset, sample_non_iid
from inflorescence.experiment import Experiment
from inflorescence.metrics import aggregate_binary_cm
from inflorescence.strategy import CFL, FLHC, IFCA, WeCFL


class CelebADataset(Dataset):
    """"""

    def __init__(self, root_dir, target_name, download=False):
        """"""
        self.root_dir = root_dir
        self.dataset = None
        self.target_name = target_name
        self.attr_names = [
            "5_o_Clock_Shadow",
            "Arched_Eyebrows",
            "Attractive",
            "Bags_Under_Eyes",
            "Bald",
            "Bangs",
            "Big_Lips",
            "Big_Nose",
            "Black_Hair",
            "Blond_Hair",
            "Blurry",
            "Brown_Hair",
            "Bushy_Eyebrows",
            "Chubby",
            "Double_Chin",
            "Eyeglasses",
            "Goatee",
            "Gray_Hair",
            "Heavy_Makeup",
            "High_Cheekbones",
            "Male",
            "Mouth_Slightly_Open",
            "Mustache",
            "Narrow_Eyes",
            "No_Beard",
            "Oval_Face",
            "Pale_Skin",
            "Pointy_Nose",
            "Receding_Hairline",
            "Rosy_Cheeks",
            "Sideburns",
            "Smiling",
            "Straight_Hair",
            "Wavy_Hair",
            "Wearing_Earrings",
            "Wearing_Hat",
            "Wearing_Lipstick",
            "Wearing_Necklace",
            "Wearing_Necktie",
            "Young",
        ]
        assert target_name in self.attr_names
        self.target_index = self.attr_names.index(self.target_name)
        if download:
            self.download()

    def download(self, group_var=None):
        y_idx = self.target_index
        if group_var:
            group_index = self.attr_names.index(group_var)
            y_idx = [self.target_index, group_index]
        self.dataset = CelebA(
            root=self.root_dir,
            download=True,
            split="all",
            target_type="attr",
            transform=transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()]),
            target_transform=transforms.Lambda(lambda y: y[y_idx].float()),
        )

    def __getitem__(self, item):
        try:
            idx = self.attr_names.index(item)
            return self.dataset.attr[idx]
        except ValueError as exc:
            raise KeyError(item) from exc

    def get_xy(self, group_var=None):
        """"""
        # TODO: implement non-iid split
        self.download(group_var=group_var)
        y_idx = self.target_index
        if group_var:
            group_index = self.attr_names.index(group_var)
            y_idx = [self.target_index, group_index]
            return self.dataset, self.dataset.attr[:, [self.target_index, group_index]]
        return self.dataset, self.dataset.attr[:, self.target_index]

    def get_split(self, cid: int):
        """"""
        return self.splits[cid]

    def split_non_iid(
        self,
        concentration: float,  # Union[float, NDArray, List[float]],
        num_clients: int,
        group_var=None,
        test_size=0.2,
        seed=None,
    ):
        """Split the dataset into n_splits by drawing from a Dirichlet distribution
        with parameter _concentration_."""

        X, y = self.get_xy(group_var=group_var)
        if group_var:
            labels = y[:, 1]
        else:
            labels = y
        partitions, dist = sample_non_iid(
            labels=labels,
            n_splits=num_clients,
            concentration=concentration,
            seed=seed,
        )

        for i, part in enumerate(partitions):
            Xp = Subset(X, part)
            yp = y[part]
            # partitions[i] = (Xp, yp)
            # shuffle = ShuffleSplit(n_splits=1, test_size=test_size)
            # train_idx, test_idx = next(shuffle.split(Xp, yp))
            if seed:
                train, test = random_split(
                    Xp, [1 - test_size, test_size], generator=Generator().manual_seed(seed)
                )
            else:
                train, test = random_split(Xp, [1 - test_size, test_size])
            partitions[i] = (
                (Subset(Xp, train.indices), yp[train.indices, :]),
                (Subset(Xp, test.indices), yp[test.indices, :]),
            )
        self.splits = partitions
        return self.splits, dist


class CelebAModel(TorchVisionModel):
    """"""

    def __init__(
        self,
        learning_rate=0.001,
        metrics=None,
        group_metrics=None,
        seed=None,
    ):
        self.learning_rate = learning_rate
        module = TorchClassifier(
            CNN(
                input_shape=(3, 128, 128),
                conv1_kernels=32,
                conv2_kernels=64,
                linear_hidden_size=128,
                # dropout=0.2,
                num_classes=1,
            ),
            criterion=F.binary_cross_entropy,
            metrics=metrics,
            group_metrics=group_metrics,
            optimizer_class=optim.SGD,
            lr=self.learning_rate,
            seed=seed,
            train_on_gpu=True,
            # device_num=2,
        )
        super().__init__(module, use_gpu=True)

if __name__ == "__main__":
    NUM_CLIENTS = 10
    NUM_ROUNDS = 10
    EPOCHS = 5
    BATCH_SIZE = 64
    K_CLUSTERS = 5
    SEED = None
    CONCENTRATIONS = np.logspace(-2, 3, 6)
    # DATA_DIR = "/media/hdd1/data/school"
    celeba = CelebADataset(target_name="Attractive", root_dir=DATA_DIR)
    initial_weights = [CelebAModel(seed=SEED).get_weights() for _ in range(K_CLUSTERS)]
    initial_weights_flat = [_ for _ in initial_weights for _ in _]
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
    logger.add(
        "logs/celeba_attractiveyoung.log", format="{time:YYYY-MM-DDTHH:mm:ss} | {level} | {message}"
    )
    for concentration in CONCENTRATIONS:
        # FLHC
        k = K_CLUSTERS
        strategy = FLHC(
            k_clusters=K_CLUSTERS,
            metric="euclidean",
            linkage="complete",
            cluster_after_round=5,
            initial_parameters=ndarrays_to_parameters(initial_weights[0]),
            **common_params,
        )
        experiment = Experiment(
            strategies=[strategy],
            model_fn=CelebAModel,
            dataset=celeba,
            group_var="Male",
            metrics=metrics,
            group_metrics=metrics,
            concentrations=[concentration],
        )
        dt = datetime.now()
        logger.info("Starting experiments for CelebA with FLHC(k_clusters={})", k)
        experiment.run(
            num_rounds=NUM_ROUNDS,
            num_clients=NUM_CLIENTS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            seed=SEED,
            num_cpus=5,
            num_gpus=1 / 3,
            # num_cpus=8,
            # num_gpus=0,
        )
        dtend = datetime.now()
        dts = dt.strftime("%Y-%m-%d_%H%M%S")
        fname = f"results/celeba_attractiveyoung_flhck_concentration_{concentration:.02f}_{k:02d}_{dts}.parquet"
        logger.info("Finished experiments. Saving results to {}", fname)
        df = experiment.to_pandas()
        df.to_parquet(fname)

        # FedAvg
        fedavg = FedAvg(
            initial_parameters=ndarrays_to_parameters(initial_weights[0]), **common_params
        )
        ifca = IFCA(
            k_clusters=K_CLUSTERS,
            initial_parameters=ndarrays_to_parameters(initial_weights_flat),
            **common_params,
        )
        experiment = Experiment(
            strategies=[fedavg],
            model_fn=CelebAModel,
            dataset=celeba,
            group_var="Young",
            metrics=metrics,
            group_metrics=metrics,
            concentrations=[concentration],
        )
        dt = datetime.now()
        logger.info("Starting experiments for CelebA")
        experiment.run(
            num_rounds=NUM_ROUNDS,
            num_clients=NUM_CLIENTS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            seed=SEED,
            num_cpus=5,
            num_gpus=1 / 3,
        )
        dtend = datetime.now()
        span = dtend - dt
        dts = dt.strftime("%Y-%m-%d_%H%M%S")
        fname = f"results/celeba_attractiveyoung_fedavg_concentration_{concentration:.02f}_{dts}.parquet"
        logger.info("Finished CelebA FedAvg experiment in {}. Saving results to {}", span, fname)
        df = experiment.to_pandas()
        df.to_parquet(fname)

        # IFCA
        experiment = Experiment(
            strategies=[ifca],
            model_fn=CelebAModel,
            dataset=celeba,
            group_var="Young",
            metrics=metrics,
            group_metrics=metrics,
            concentrations=[concentration],
        )
        dt = datetime.now()
        logger.info("Starting experiments for CelebA")
        experiment.run(
            num_rounds=NUM_ROUNDS,
            num_clients=NUM_CLIENTS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            seed=SEED,
            num_cpus=5,
            num_gpus=1 / 3,
        )
        dtend = datetime.now()
        span = dtend - dt
        dts = dt.strftime("%Y-%m-%d_%H%M%S")
        fname = (
            f"results/celeba_attractiveyoung_ifca_concentration_{concentration:.02f}_{dts}.parquet"
        )
        logger.info("Finished CelebA IFCA experiment in {}. Saving results to {}", span, fname)
        df = experiment.to_pandas()
        df.to_parquet(fname)
        
        # WeCFL
        wecfl = WeCFL(
            k_clusters=K_CLUSTERS,
            # initial_parameters=ndarrays_to_parameters(initial_weights[0]),
            initial_parameters=ndarrays_to_parameters(initial_weights_flat),
            **common_params,
        )
        experiment = Experiment(
            strategies=[wecfl],
            model_fn=CelebAModel,
            dataset=celeba,
            group_var="Young",
            metrics=metrics,
            group_metrics=metrics,
            concentrations=[concentration],
        )
        dt = datetime.now()
        logger.info("Starting experiments for CelebA")
        experiment.run(
            num_rounds=NUM_ROUNDS,
            num_clients=NUM_CLIENTS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            seed=SEED,
            num_cpus=5,
            num_gpus=1 / 3,
        )
        dtend = datetime.now()
        span = dtend - dt
        dts = dt.strftime("%Y-%m-%d_%H%M%S")
        fname = f"results/celeba_attractiveyoung_wecfl_concentration_{concentration:.02f}_{K_CLUSTERS}_{dts}.parquet"
        logger.info("Finished CelebA WeCFL experiments in {}. Saving results to {}", span, fname)
        df = experiment.to_pandas()
        df.to_parquet(fname)
