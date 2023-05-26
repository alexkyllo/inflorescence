"""Fairness metric functions."""
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from flwr.common.typing import Metrics, NDArray
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors

from inflorescence.dataset import Dataset


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Multiply each metric of each client by number of examples and divide by total sample size.
    Sample size for a metric will only include clients who reported that metric.
    """
    values = {}  # key: metric name, value: weighted sum of metric values across clients
    num_examples = {}  # key: metric name, value: number of examples for that metric
    for n, met in metrics:
        for k, v in met.items():
            values.setdefault(k, 0)
            num_examples.setdefault(k, 0)
            values[k] += n * v
            num_examples[k] += n
    return {k: v / num_examples[k] if num_examples[k] >= 1 else np.nan for k, v in values.items()}


def sum_counts(metrics: List[Tuple[int, Metrics]]):
    """Just sum the metrics, useful for counts."""
    values = {}
    for _, metric in metrics:
        for k, v in metric.items():
            values.setdefault(k, 0)
            values[k] += v
    return values


def aggregate_binary_cm(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Sum the confusion matrix counts but weighted average any other metrics."""
    values = {}
    num_examples = {}
    regex = "(tn|tp|fn|fp|cm)"
    regex2 = "(true_negatives|true_positives|false_negatives|false_positives)"
    for n, met in metrics:
        if "cluster" in met:
            # tally up the count of clients per cluster
            cluster_num = met["cluster"]
            key = f"cluster_{cluster_num:02}"
            values[key] = values.get(key, 0) + 1
        for k, v in met.items():
            if not k.startswith("cluster"):  # Don't aggregate cluster number
                values.setdefault(k, 0)
                num_examples.setdefault(k, 0)
                if re.search(regex, k) or re.search(regex2, k):
                    values[k] += v
                else:
                    values[k] += v * n
                num_examples[k] += n
    for k, v in values.items():
        if not re.search(regex, k) and not re.search(regex2, k) and not k.startswith("cluster"):
            values[k] = v / num_examples[k] if num_examples[k] >= 1 else np.nan

    return values


def ppp_cm(tn: int, fp: int, fn: int, tp: int):
    """Calculate probability of positive prediction from a confusion matrix."""
    return (tp + fp) / (tn + fp + fn + tp)


def tpr_cm(tn: int, fp: int, fn: int, tp: int):
    """Calculate true positive rate from a confusion matrix."""
    return tp / (tp + fn)


def fpr_cm(tn: int, fp: int, fn: int, tp: int):
    """Calculate false positive rate from a confusion matrix."""
    return fp / (fp + tn)


def ppv_cm(tn: int, fp: int, fn: int, tp: int):
    """Calculate positive predictive value from a confusion matrix."""
    return tp / (tp + fp)


def true_negatives(yhat: NDArray, y: NDArray):
    return ((1 - y) * (1 - yhat)).sum()


def true_positives(yhat: NDArray, y: NDArray):
    return (y * yhat).sum()


def false_positives(yhat: NDArray, y: NDArray):
    return ((1 - y) * yhat).sum()


def false_negatives(yhat: NDArray, y: NDArray):
    return (y * (1 - yhat)).sum()


def acc_cm(tn: int, fp: int, fn: int, tp: int):
    """Calculate binary classifier accuracy from a confusion matrix."""
    return (tn + tp) / (tn + fp + fn + tp)


def disparate_impact_ratio(yhat: NDArray, y: NDArray, a: NDArray):
    """Disparate impact ratio metric.
    Ratio of probability of positive prediction by group status.
    """
    y_hat = yhat >= 0.5
    groups = np.unique(a)
    num_groups = len(groups)
    if num_groups < 2:
        return {}
    result = np.ones((num_groups))
    for i, group in enumerate(groups):
        ingroup = y_hat[a == group]
        outgroup = y_hat[a != group]
        ingroup_ppp = ingroup.sum() / len(ingroup)
        outgroup_ppp = outgroup.sum() / len(outgroup)
        result[i] = ingroup_ppp / outgroup_ppp
    return {f"disparate_impact_ratio_{i:02}": v for i, v in enumerate(result)}


def equal_odds_ratio(yhat: NDArray, y: NDArray, a: NDArray):
    """
    Lesser of ratios of true positive and false positive rates by protected group status
    """
    groups = np.unique(a)
    num_groups = len(groups)
    if num_groups < 2:
        return {}
    result = np.ones((num_groups))
    for i, group in enumerate(groups):
        tpr_a = true_positive_rate(yhat[a == group], y[a == group])
        fpr_a = false_positive_rate(yhat[a == group], y[a == group])
        tpr_nota = true_positive_rate(yhat[a != group], y[a != group])
        fpr_nota = false_positive_rate(yhat[a != group], y[a != group])
        result[i] = min(np.divide(tpr_a, tpr_nota), np.divide(fpr_a, fpr_nota))
    return {f"equal_odds_ratio_{i:02}": v for i, v in enumerate(result)}


def binary_confusion_matrix(yhat: NDArray, y: NDArray):
    """Calculate the confusion matrix."""
    tp = (y * yhat).sum()
    tn = ((1 - y) * (1 - yhat)).sum()
    fp = ((1 - y) * yhat).sum()
    fn = (y * (1 - yhat)).sum()
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def group_binary_confusion_matrix(yhat: NDArray, y: NDArray, a: NDArray):
    """Calculate the confusion matrix."""
    groups = np.unique(a)
    result = {}
    for i, group in enumerate(groups):
        group_cm = binary_confusion_matrix(yhat[a == group], y[a == group])
        result.update({f"{k}_group_{i:02}": v for k, v in group_cm.items()})
    return result


def multiclass_confusion_matrix(yhat: NDArray, y: NDArray):
    cm = confusion_matrix(y, yhat)
    res = {}
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            res[f"cm_{i:02}_{j:02}"] = cm[i][j]
    return res


def true_positive_rate(yhat: NDArray, y: NDArray):
    """TPR."""
    tp = (y * yhat).sum()
    fn = (y * (1 - yhat)).sum()
    return tp / (tp + fn)


def false_positive_rate(yhat: NDArray, y: NDArray):
    """FPR."""
    tn = ((1 - y) * (1 - yhat)).sum()
    fp = ((1 - y) * yhat).sum()
    return fp / (fp + tn)


def positive_predictive_value(yhat: NDArray, y: NDArray):
    """PPV."""
    tp = (y * yhat).sum()
    fp = ((1 - y) * yhat).sum()
    return tp / (tp + fp)


def positive_predictive_parity_ratio(yhat: NDArray, y: NDArray, a: NDArray):
    """Positive predictive parity ratio.
    Ratio of positive predictive value by protected group status
    """
    groups = np.unique(a)
    num_groups = len(groups)
    if num_groups < 2:
        return {}
    result = np.ones((num_groups))
    for i, group in enumerate(groups):
        ppv_a = positive_predictive_value(yhat[a == group], y[a == group])
        ppv_nota = positive_predictive_value(yhat[a != group], y[a != group])
        result[i] = ppv_a / ppv_nota
    return {f"positive_predictive_parity_ratio_{i:02}": v for i, v in enumerate(result)}


def theil_index(yhat: NDArray, y: NDArray, a: Optional[NDArray] = None):
    """Theil Index."""
    b = 1 + yhat - y
    ub = b.mean()
    return (np.log((b / ub) ** b) / ub).mean()


def theil_index_cm(tn: int, fp: int, fn: int, tp: int):
    """Theil Index from a confusion matrix."""
    bsum = tn + tp + (2 * fp)
    n = tn + fp + fn + tp
    ub = bsum / n
    btn = np.log((1 / ub)) / ub
    bfp = np.log((2 / ub) ** 2) / ub
    btp = btn
    return ((btn * tn) + (bfp * fp) + (btp * tp)) / n


def between_group_theil_index(yhat: NDArray, y: NDArray, a: NDArray):
    """Between group Theil index.
    Based on https://aif360.readthedocs.io/en/latest/modules/generated/aif360.metrics.ClassificationMetric.html#aif360.metrics.ClassificationMetric.between_group_generalized_entropy_index
    """
    groups = np.unique(a)
    b = np.zeros(len(groups), dtype=np.float64)
    for group in groups:
        y_pred = yhat[a == group]
        y_true = y[a == group]
        b[group] = np.mean(1 + y_pred - y_true)
    return np.mean(np.log((b / np.mean(b)) ** b) / np.mean(b))


def between_group_theil_index_cm(tn: NDArray, fp: NDArray, fn: NDArray, tp: NDArray):
    """Between group Theil index based on confusion matrix."""
    groups = np.arange(tn.shape[0])
    b = np.zeros(len(groups), dtype=np.float64)
    for g in groups:
        bsum = tn[g] + tp[g] + (2 * fp[g])
        n = tn[g] + fp[g] + fn[g] + tp[g]
        b[g] = bsum / n
    return np.mean(np.log((b / np.mean(b)) ** b) / np.mean(b))


def consistency_score(X: NDArray, yhat: NDArray, k: int = 5):
    """Consistency score. Measures that predictions are similar to nearest neighbors."""
    knn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(X)
    idx = knn.kneighbors(X, return_distance=False)
    return 1 - abs(yhat - yhat[idx].mean(axis=1)).mean()


def add_binary_fairness_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add fairness metrics to a data frame"""
    grouping = ["round"]
    if "group" in df.columns:
        grouping.append("group")
    if "cluster" in df.columns:
        grouping.append("cluster")
    if "dataset" in df.columns:
        grouping.append("dataset")
    if "strategy" in df.columns:
        grouping.append("strategy")
    if "trial" in df.columns:
        grouping.append("trial")
    if "concentration" in df.columns:
        grouping.append("concentration")
    dfp = df.pivot(index=grouping, columns="metric", values="value").reset_index()
    dfp = dfp.assign(
        val_ppv=ppv_cm(
            dfp["valid_cm_tn"], dfp["valid_cm_fp"], dfp["valid_cm_fn"], dfp["valid_cm_tp"]
        ),
        val_ppp=ppp_cm(
            dfp["valid_cm_tn"], dfp["valid_cm_fp"], dfp["valid_cm_fn"], dfp["valid_cm_tp"]
        ),
        val_tpr=tpr_cm(
            dfp["valid_cm_tn"], dfp["valid_cm_fp"], dfp["valid_cm_fn"], dfp["valid_cm_tp"]
        ),
        val_fpr=fpr_cm(
            dfp["valid_cm_tn"], dfp["valid_cm_fp"], dfp["valid_cm_fn"], dfp["valid_cm_tp"]
        ),
        val_theil=theil_index_cm(
            dfp["valid_cm_tn"], dfp["valid_cm_fp"], dfp["valid_cm_fn"], dfp["valid_cm_tp"]
        ),
    )
    if "group" in df.columns:
        if dfp.group.dtype.name == "category":
            groups = dfp.group.cat.categories
        else:
            groups = sorted(dfp.group.dropna().unique())
        for g in groups:
            # Calculate PPP ratio
            ingroup_ppv = dfp[dfp.group.eq(g)]["val_ppv"]
            outgroup = dfp[~dfp.group.eq(g) & ~dfp.group.isna()]
            outgroup_ppv = ppv_cm(
                outgroup["valid_cm_tn"],
                outgroup["valid_cm_fp"],
                outgroup["valid_cm_fn"],
                outgroup["valid_cm_tp"],
            )
            dfp.loc[dfp.group.eq(g), "ppp_ratio"] = ingroup_ppv.values / outgroup_ppv.values
            # Calculate Disparate Impact Ratio
            ingroup_ppp = dfp[dfp.group.eq(g)]["val_ppp"]
            outgroup_ppp = ppp_cm(
                outgroup["valid_cm_tn"],
                outgroup["valid_cm_fp"],
                outgroup["valid_cm_fn"],
                outgroup["valid_cm_tp"],
            )
            dfp.loc[dfp.group.eq(g), "disparate_impact_ratio"] = (
                ingroup_ppp.values / outgroup_ppp.values
            )
            # Calculate Equal Odds Ratio
            ingroup_tpr = dfp[dfp.group.eq(g)]["val_tpr"]
            ingroup_fpr = dfp[dfp.group.eq(g)]["val_fpr"]
            outgroup_tpr = tpr_cm(
                outgroup["valid_cm_tn"],
                outgroup["valid_cm_fp"],
                outgroup["valid_cm_fn"],
                outgroup["valid_cm_tp"],
            )
            outgroup_fpr = fpr_cm(
                outgroup["valid_cm_tn"],
                outgroup["valid_cm_fp"],
                outgroup["valid_cm_fn"],
                outgroup["valid_cm_tp"],
            )
            dfp.loc[dfp.group.eq(g), "disparate_impact_ratio"] = np.minimum(
                ingroup_tpr.values / outgroup_tpr.values,
                ingroup_fpr.values / outgroup_fpr.values,
            )
    return dfp
    # TODO: add fields for experiment parameters
