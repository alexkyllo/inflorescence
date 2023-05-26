"""An interface for a dataset wrapper class."""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
from flwr.common.typing import NDArray, NDArrays
from openml import datasets
from sklearn.model_selection import ShuffleSplit, train_test_split

from inflorescence.dataset.common import XY, exclude_classes_and_normalize


class Dataset(ABC):
    """An interface for retrieving and splitting training datasets."""

    def __init__(self):
        self.splits = []

    @abstractmethod
    def download(self):
        """Download the dataset from an internet source."""

    @abstractmethod
    def get_xy(self, group_var: Optional[str] = None) -> XY:
        """Return a tuple of (X, y) or (features, labels)."""

    # @abstractmethod
    # def __getitem__(self, item):
    #     """"""

    def split_iid(
        self,
        num_clients: int,
        group_var: Optional[str] = None,
        test_size: float = 0.2,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> List[Tuple[XY, XY]]:
        """Get a uniform random split of the data with num_clients splits."""
        X, y = self.get_xy(group_var=group_var)
        np.random.default_rng(seed).shuffle(X)
        np.random.default_rng(seed).shuffle(y)
        X = np.array_split(X, num_clients)
        y = np.array_split(y, num_clients)
        splits = []
        for i in range(len(X)):
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], test_size=test_size, random_state=seed
            )
            splits.append(((X_train, y_train), (X_test, y_test)))
        self.splits = splits
        return self.splits

    def split_non_iid(
        self,
        concentration: float,  # Union[float, NDArray, List[float]],
        num_clients: int,
        group_var: Optional[str] = None,
        test_size: float = 0.2,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ) -> Tuple[List[Tuple[XY, XY]], NDArray]:
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
            Xp = X[part]
            yp = y[part]
            # partitions[i] = (Xp, yp)
            shuffle = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_idx, test_idx = next(shuffle.split(Xp, yp))
            partitions[i] = (
                (Xp[train_idx, :], yp[train_idx]),
                (Xp[test_idx, :], yp[test_idx]),
            )
        self.splits = partitions
        return self.splits, dist

    def get_split(self, cid: int):
        return self.splits[cid]


def download_openml(dataset_id: int) -> NDArray:
    return datasets.get_dataset(dataset_id=dataset_id, download_data=True).get_data()


def get_split_sizes(ary: NDArray, n_splits: int):
    """Get the sizes of an array partitioning as evenly as possible."""
    split_sizes = []
    quo, rem = divmod(len(ary), n_splits)
    for i in range(n_splits):
        split_sizes.append(quo)
    for i in range(rem):
        split_sizes[i] += 1
    return split_sizes


def sample_non_iid(
    labels: NDArray,
    n_splits: int,
    concentration: float,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> NDArray:
    """
    Partition a 1D array of labels non-IID and return a 2D array of dimension
    (partition, indices). Returns the indices so that the original dataset X and
    y can be partitioned by them without loading the whole dataset into memory,
    and so that the non-iid split label need not be the class label y.

    """
    # get sizes of each of n splits
    split_sizes = get_split_sizes(labels, n_splits)
    # get the probability of each class in the dataset
    classes, freq = np.unique(labels, return_counts=True)
    prob = freq / freq.sum()
    # then scale it by the concentration parameter to get the alpha for Dirichlet
    concentration *= prob
    # get the indices of each class in labels
    class_idx = [np.flatnonzero(labels == c) for c in classes]
    # Make n_splits draws from a dirichlet distribution of classes.
    dist = np.random.default_rng(seed).dirichlet(alpha=concentration, size=n_splits)
    splits = [(_, _) for _ in range(n_splits)]
    empty_classes = classes.size * [False]
    for split_num in range(n_splits):
        splits[split_num], empty_classes = sample_without_replacement(
            distribution=dist[split_num],
            class_idx=class_idx,
            n=split_sizes[split_num],
            empty_classes=empty_classes,
            seed=seed,
        )
    return splits, dist


def sample_without_replacement(distribution, class_idx, n, empty_classes, seed):
    """Randomly sample from classes without replacement according to Dirichlet distribution."""
    distribution = exclude_classes_and_normalize(distribution.copy(), exclude_dims=empty_classes)
    # sample = np.random.default_rng(seed).choice(class_idx, size=n, p=distribution, replace=False)
    sample = []
    generator = np.random.default_rng(seed)
    for _ in range(n):
        # randomly pick a class according to the Dirichlet distribution
        # sample_class = np.where(np.random.default_rng(seed).multinomial(1, distribution) == 1)[0][0]
        sample_class = generator.choice(len(class_idx), p=distribution)
        # randomly pick an element from the chosen class
        choice_idx = generator.choice(len(class_idx[sample_class]))
        # add that chosen element to the sample
        sample.append(class_idx[sample_class][choice_idx])
        # delete that element from the class
        class_idx[sample_class] = np.delete(class_idx[sample_class], np.array([choice_idx]))
        if len(class_idx[sample_class]) == 0:
            empty_classes[sample_class] = True
            distribution = exclude_classes_and_normalize(
                distribution=distribution, exclude_dims=empty_classes
            )
    return np.asarray(sample), empty_classes
