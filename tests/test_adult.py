import numpy as np

from inflorescence.dataset.adult import AdultDataset


def test_adult_dataset_get_xy_groupvar():
    """"""
    adult = AdultDataset()
    _, y = adult.get_xy(group_var="race")
    assert y.shape[1] == 2
    assert np.array_equal(np.unique(y[:, 1]), np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    # assert np.array_equal(adult["race"].values, y[:, 1])


def test_adult_dataset_get_xy():
    """"""
    adult = AdultDataset()
    _, y = adult.get_xy()
    assert y.shape == (45222,)
    assert np.array_equal(np.unique(y), np.array([0.0, 1.0]))


def test_adult_dataset_split_non_iid():
    """"""
    adult = AdultDataset()
    splits, dist = adult.split_non_iid(
        group_var="sex", num_clients=20, concentration=[0.48, 0.52], test_size=0.2, seed=42
    )
    assert len(splits) == 20
    assert len(splits[0]) == 2
    assert len(splits[0][0]) == 2
    ((X_train_0, y_train_0), (X_test_0, y_test_0)) = splits[0]
    assert y_train_0.shape[1] == 2
