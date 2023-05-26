import numpy as np
import pandas as pd
from flwr.server.history import History
from sklearn.datasets import make_moons

from inflorescence.dataset.adult import AdultDataset
from inflorescence.experiment import (
    dataframe_from_history,
    parse_group_metric_name,
    parse_group_metrics,
)
from inflorescence.metrics import (
    add_binary_fairness_metrics,
    aggregate_binary_cm,
    between_group_theil_index,
    between_group_theil_index_cm,
    consistency_score,
    disparate_impact_ratio,
    equal_odds_ratio,
    false_negatives,
    false_positive_rate,
    false_positives,
    fpr_cm,
    group_binary_confusion_matrix,
    multiclass_confusion_matrix,
    ppp_cm,
    ppv_cm,
    sum_counts,
    theil_index,
    theil_index_cm,
    tpr_cm,
    true_negatives,
    true_positive_rate,
    true_positives,
    weighted_average,
)
from inflorescence.strategy import IFCA


def test_disparate_impact_ratio():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0])
    yhat = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0])
    a = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    result = [v for _, v in disparate_impact_ratio(yhat, y, a).items()]
    assert np.array_equal(result, np.array([2 / 3, 2 / 3, 2], dtype=float))


def test_disparate_impact_ratio_binary_case():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0])
    yhat = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0])
    a = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
    result = [v for _, v in disparate_impact_ratio(yhat, y, a).items()]
    assert np.array_equal(result, np.array([(1 / 3) / (1 / 2), (1 / 2) / (1 / 3)]))


def test_equal_odds_ratio():
    y___ = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0])
    yhat = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0])
    a___ = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2])

    result = [v for _, v in equal_odds_ratio(yhat, y___, a___).items()]
    assert np.array_equal(result, np.array([0.5, 1.25, 1.25], dtype=float))


def test_equal_odds_ratio_float():
    y___ = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0])
    yhat = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    a___ = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2])

    result = [v for _, v in equal_odds_ratio(yhat, y___, a___).items()]
    assert np.array_equal(result, np.array([0.5, 1.25, 1.25], dtype=float))


def test_consistency_score():
    """Test that the consistency_score function works."""
    x, y = make_moons()
    cons = consistency_score(x, y)
    assert np.isclose(cons, 1.0)


def test_theil_index():
    y___ = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0])
    yhat = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0])
    ti = theil_index(yhat, y___)
    tn = true_negatives(yhat, y___)
    tp = true_positives(yhat, y___)
    fn = false_negatives(yhat, y___)
    fp = false_positives(yhat, y___)
    ticm = theil_index_cm(tn, fp, fn, tp)
    assert np.isclose(ti, ticm)


def test_between_group_theil_index():
    y___ = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0])
    yhat = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0])
    a___ = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    groups = np.unique(a___)
    ti = between_group_theil_index(yhat, y___, a___)
    tn = np.array([true_negatives(yhat[a___ == g], y___[a___ == g]) for g in groups])
    tp = np.array([true_positives(yhat[a___ == g], y___[a___ == g]) for g in groups])
    fn = np.array([false_negatives(yhat[a___ == g], y___[a___ == g]) for g in groups])
    fp = np.array([false_positives(yhat[a___ == g], y___[a___ == g]) for g in groups])
    ticm = between_group_theil_index_cm(tn, fp, fn, tp)
    assert np.isclose(ti, ticm)


def test_weighted_average():
    """Test that weighted_average works."""
    metrics = [(10, {"accuracy": 0.5}), (20, {"accuracy": 0.8})]
    result = weighted_average(metrics)
    assert result["accuracy"] == 0.7


def test_weighted_average():
    """Test that weighted_average averages over clients who reported a given metric."""
    metrics = [
        (10, {"accuracy": 0.5}),
        (20, {"accuracy": 0.8}),
        (5, {"precision": 0.7}),
        (0, {"recall": 0.2}),
    ]
    result = weighted_average(metrics)
    assert result["accuracy"] == 0.7
    assert result["precision"] == 0.7
    assert np.isnan(result["recall"])


def test_multiclass_confusion_matrix():
    y___ = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    yhat = [2, 0, 0, 0, 1, 1, 1, 2, 0]
    res = multiclass_confusion_matrix(yhat=yhat, y=y___)
    assert res == {
        "cm_00_00": 2,
        "cm_00_01": 0,
        "cm_00_02": 1,
        "cm_01_00": 1,
        "cm_01_01": 2,
        "cm_01_02": 0,
        "cm_02_00": 1,
        "cm_02_01": 1,
        "cm_02_02": 1,
    }


def test_sum_counts():
    c1 = (
        9,
        {
            "cm_00_00": 2,
            "cm_00_01": 0,
            "cm_00_02": 1,
            "cm_01_00": 1,
            "cm_01_01": 2,
            "cm_01_02": 0,
            "cm_02_00": 1,
            "cm_02_01": 1,
            "cm_02_02": 1,
        },
    )
    c2 = (
        9,
        {
            "cm_00_00": 2,
            "cm_00_01": 0,
            "cm_00_02": 1,
            "cm_01_00": 1,
            "cm_01_01": 2,
            "cm_01_02": 0,
            "cm_02_00": 1,
            "cm_02_01": 1,
            "cm_02_02": 1,
        },
    )
    assert sum_counts([c1, c2]) == {
        "cm_00_00": 4,
        "cm_00_01": 0,
        "cm_00_02": 2,
        "cm_01_00": 2,
        "cm_01_01": 4,
        "cm_01_02": 0,
        "cm_02_00": 2,
        "cm_02_01": 2,
        "cm_02_02": 2,
    }


def test_aggregate_cm():
    """Test that aggregate_cm works."""
    c1 = (
        9,
        {
            "cm_00_00": 2,
            "cm_00_01": 0,
            "cm_00_02": 1,
            "cm_01_00": 1,
            "cm_01_01": 2,
            "cm_01_02": 0,
            "cm_02_00": 1,
            "cm_02_01": 1,
            "cm_02_02": 1,
            "val_tn": 1,
            "accuracy": 0.5,
            "cluster": 0,
        },
    )
    c2 = (
        9,
        {
            "cm_00_00": 2,
            "cm_00_01": 0,
            "cm_00_02": 1,
            "cm_01_00": 1,
            "cm_01_01": 2,
            "cm_01_02": 0,
            "cm_02_00": 1,
            "cm_02_01": 1,
            "cm_02_02": 1,
            "val_tn": 1,
            "accuracy": 0.9,
            "cluster": 1,
        },
    )
    assert aggregate_binary_cm([c1, c2]) == {
        "cluster_00": 1,
        "cm_00_00": 4,
        "cm_00_01": 0,
        "cm_00_02": 2,
        "cm_01_00": 2,
        "cm_01_01": 4,
        "cm_01_02": 0,
        "cm_02_00": 2,
        "cm_02_01": 2,
        "cm_02_02": 2,
        "val_tn": 2,
        "accuracy": 0.7,
        "cluster_01": 1,
    }


def test_aggregate_cm_groups():
    m_0 = {
        "valid_cm_00_00_group_00": 0.0,
        "valid_cm_00_00_group_01": 1.0,
        "valid_cm_00_00_group_02": 31.0,
        "valid_cm_00_00_group_03": 2.0,
    }
    m_1 = {
        "valid_cm_00_00_group_00": 342.0,
        "valid_cm_00_00_group_01": 333.0,
        "valid_cm_00_00_group_02": 1.0,
    }

    agg = aggregate_binary_cm([(34, m_0), (676, m_1)])

    assert agg == {
        "valid_cm_00_00_group_00": 342.0,
        "valid_cm_00_00_group_01": 334.0,
        "valid_cm_00_00_group_02": 32.0,
        "valid_cm_00_00_group_03": 2.0,
    }


def test_group_binary_confusion_matrix():
    """Test that we can calculate confusion matrix per group."""
    y___ = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0])
    yhat = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0])
    a___ = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    result = group_binary_confusion_matrix(yhat, y___, a___)
    assert result == {
        "tn_group_00": 2,
        "fp_group_00": 1,
        "fn_group_00": 1,
        "tp_group_00": 1,
        "tn_group_01": 1,
        "fp_group_01": 1,
        "fn_group_01": 0,
        "tp_group_01": 1,
        "tn_group_02": 1,
        "fp_group_02": 1,
        "fn_group_02": 0,
        "tp_group_02": 1,
    }


def test_parse_group_metric_name():
    """test parsing group metric names"""
    assert parse_group_metric_name("val_acc_group_00") == ("val_acc", 0, None)
    assert parse_group_metric_name("val_tn_group_03") == ("val_tn", 3, None)
    assert parse_group_metric_name("val_tn_cluster_01") == ("val_tn", None, 1)
    assert parse_group_metric_name("val_tn_group_03_cluster_01") == ("val_tn", 3, 1)
    assert parse_group_metric_name("val_tn_group_3_cluster_1") == ("val_tn", 3, 1)
    assert parse_group_metric_name("val_tn") == ("val_tn", None, None)


def test_parse_group_metrics():
    """test parsing group and cluster metrics into records."""
    metrics = {
        "val_acc_group_00": 0.5,
        "val_acc_cluster_1": 0.6,
        "val_acc": 0.7,
        "val_acc_group_0_cluster_1": 0.8,
    }
    expected = [
        {"metric": "val_acc", "value": 0.5, "group": 0},
        {"metric": "val_acc", "value": 0.6, "cluster": 1},
        {"metric": "val_acc", "value": 0.7},
        {"metric": "val_acc", "value": 0.8, "group": 0, "cluster": 1},
    ]
    result = parse_group_metrics(metrics)
    assert result == expected


def test_metrics_frame():
    metrics = {
        "loss_cluster_00": [(1, 59.447126388549805), (2, 68.02286009355025)],
        "val_tn_cluster_00": [(1, 2049.0), (2, 3578.0)],
        "val_fp_cluster_00": [(1, 64.0), (2, 135.0)],
        "val_fn_cluster_00": [(1, 474.0), (2, 987.0)],
        "val_tp_cluster_00": [(1, 131.0), (2, 283.0)],
        "val_tn_group_00_cluster_00": [(1, 1320.0), (2, 2096.0)],
        "val_fp_group_00_cluster_00": [(1, 40.0), (2, 85.0)],
        "val_fn_group_00_cluster_00": [(1, 208.0), (2, 445.0)],
        "val_tp_group_00_cluster_00": [(1, 48.0), (2, 117.0)],
        "val_tn_group_01_cluster_00": [(1, 729.0), (2, 1482.0)],
        "val_fp_group_01_cluster_00": [(1, 24.0), (2, 50.0)],
        "val_fn_group_01_cluster_00": [(1, 266.0), (2, 542.0)],
        "val_tp_group_01_cluster_00": [(1, 83.0), (2, 166.0)],
        "loss_cluster_01": [(1, 64.18461952209472), (2, 64.47491455078125)],
        "val_tn_cluster_01": [(1, 2961.0), (2, 1485.0)],
        "val_fp_cluster_01": [(1, 212.0), (2, 88.0)],
        "val_fn_cluster_01": [(1, 938.0), (2, 484.0)],
        "val_tp_cluster_01": [(1, 419.0), (2, 208.0)],
        "val_tn_group_00_cluster_01": [(1, 1597.0), (2, 849.0)],
        "val_fp_group_00_cluster_01": [(1, 124.0), (2, 51.0)],
        "val_fn_group_00_cluster_01": [(1, 489.0), (2, 275.0)],
        "val_tp_group_00_cluster_01": [(1, 208.0), (2, 116.0)],
        "val_tn_group_01_cluster_01": [(1, 1364.0), (2, 636.0)],
        "val_fp_group_01_cluster_01": [(1, 88.0), (2, 37.0)],
        "val_fn_group_01_cluster_01": [(1, 449.0), (2, 209.0)],
        "val_tp_group_01_cluster_01": [(1, 211.0), (2, 92.0)],
        "loss_cluster_02": [(1, 62.32779598236084), (2, 62.016512870788574)],
        "val_tn_cluster_02": [(1, 1550.0), (2, 1550.0)],
        "val_fp_cluster_02": [(1, 0.0), (2, 0.0)],
        "val_fn_cluster_02": [(1, 262.0), (2, 262.0)],
        "val_tp_cluster_02": [(1, 0.0), (2, 0.0)],
        "val_tn_group_00_cluster_02": [(1, 1297.0), (2, 1297.0)],
        "val_fp_group_00_cluster_02": [(1, 0.0), (2, 0.0)],
        "val_fn_group_00_cluster_02": [(1, 157.0), (2, 157.0)],
        "val_tp_group_00_cluster_02": [(1, 0.0), (2, 0.0)],
        "val_tn_group_01_cluster_02": [(1, 253.0), (2, 253.0)],
        "val_fp_group_01_cluster_02": [(1, 0.0), (2, 0.0)],
        "val_fn_group_01_cluster_02": [(1, 105.0), (2, 105.0)],
        "val_tp_group_01_cluster_02": [(1, 0.0), (2, 0.0)],
        "cluster_02": [(1, 4), (2, 4)],
        "val_tn": [(1, 6560.0), (2, 6613.0)],
        "val_fp": [(1, 276.0), (2, 223.0)],
        "val_fn": [(1, 1674.0), (2, 1733.0)],
        "val_tp": [(1, 550.0), (2, 491.0)],
        "val_tn_group_00": [(1, 4214.0), (2, 4242.0)],
        "val_fp_group_00": [(1, 164.0), (2, 136.0)],
        "val_fn_group_00": [(1, 854.0), (2, 877.0)],
        "val_tp_group_00": [(1, 256.0), (2, 233.0)],
        "val_tn_group_01": [(1, 2346.0), (2, 2371.0)],
        "val_fp_group_01": [(1, 112.0), (2, 87.0)],
        "val_fn_group_01": [(1, 820.0), (2, 856.0)],
        "val_tp_group_01": [(1, 294.0), (2, 258.0)],
        "cluster_00": [(1, 6), (2, 11)],
        "cluster_01": [(1, 10), (2, 5)],
    }
    hist = History()
    hist.metrics_distributed = metrics
    idx = pd.CategoricalIndex(
        ["Female", "Male"], categories=["Female", "Male"], ordered=True, dtype="category"
    )
    df = dataframe_from_history(
        hist, dataset=AdultDataset(download=False), strategy=IFCA(2), group_levels=idx, trial=1
    )
    assert df.group.value_counts()["Male"] == 32
    assert df.group.value_counts()["Female"] == 32
    assert df.dataset.value_counts()["Adult"] == 108
    assert df.strategy.value_counts()["IFCA"] == 108


def test_dataframe_from_history_none():
    """Test that dataframe_from_history returns an empty df if history is none"""
    df = dataframe_from_history(None)
    assert df.shape[0] == 0
    assert df.shape[1] > 0
