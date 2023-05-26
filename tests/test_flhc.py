import numpy as np

from inflorescence.strategy.flhc import FLHC, cluster_clients


def test_cluster_clients():
    weights = {
        2: [
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]),
        ],
        1: [
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 8.0]]),
        ],
        4: [
            np.array([[-3.0, -2.0, -1.0], [0.0, 1.0, 2.0]]),
            np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]),
        ],
        3: [
            np.array([[-4.0, -3.0, -1.0], [0.0, 1.0, 2.0]]),
            np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]),
        ],
    }

    clusters = cluster_clients(
        weights, distance_threshold=2.0, metric="euclidean", linkage="complete"
    )
    assert clusters == {2: 1, 1: 1, 4: 0, 3: 0}
