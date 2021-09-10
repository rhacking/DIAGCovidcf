import itertools
from typing import List, Dict, Callable, Union

import pandas as pd
import numpy as np
import scipy.stats


# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)


def bootstrap(a, n_repeat, n_resample, metric):
    values = []
    for i in range(n_repeat):
        indices = np.random.choice(len(a), size=n_resample)
        a_resampled = a[indices]
        values.append(metric(a_resampled))
    return values


def bootstrap_test(ground_truth, predictions_one, predictions_two, n_repeat, metric):
    def two_better_than_one(x):
        x = x.T
        try:
            m1 = metric(x[0], x[1])
            m2 = metric(x[0], x[2])
        except ValueError:
            return 0.5
        return m1 < m2

    boot = bootstrap(np.array(list(zip(ground_truth, predictions_one, predictions_two))), n_repeat,
                     len(ground_truth), two_better_than_one)
    p_value = np.mean(boot)
    return p_value


statistic_function = Callable[[np.array, np.array, np.array], float]


def mcnemar_test(ground_truth: np.ndarray, predictions_one: np.ndarray, predictions_two: np.ndarray):
    from statsmodels.stats.contingency_tables import mcnemar
    ct = np.array([
        [((ground_truth == predictions_one.round()) & (ground_truth == predictions_two.round())).sum(),
         ((ground_truth == predictions_one.round()) & (ground_truth != predictions_two.round())).sum()],
        [((ground_truth != predictions_one.round()) & (ground_truth == predictions_two.round())).sum(),
         ((ground_truth != predictions_one.round()) & (ground_truth != predictions_two.round())).sum()]
    ])

    return mcnemar(ct, exact=True).pvalue


def compare_models(ground_truth, model_names: List[str], model_preds: List[np.ndarray],
                   statistic: Union[statistic_function, Dict[str, statistic_function]], bonferroni_factor=None, test_all=False, decimals=3):
    results = []
    iterator = itertools.permutations(range(len(model_preds)), 2) if test_all else itertools.combinations(range(len(model_preds)), 2)
    for a, b in iterator:
        if isinstance(statistic, dict):
            stats = {}
            for stat_name, stat_func in statistic.items():
                stats[stat_name+'_pvalue'] = float(np.round(stat_func(ground_truth, model_preds[a], model_preds[b]), decimals))
            row = {'model_a': model_names[a], 'model_b': model_names[b]}
            row.update(stats)
            results.append(row)
        else:
            p_value = float(np.round(statistic(ground_truth, model_preds[a], model_preds[b]), decimals))
            results.append({'model_a': model_names[a], 'model_b': model_names[b], 'pvalue': p_value})

    results = pd.DataFrame(results)
    if bonferroni_factor is None:
        results.loc[:, results.columns.str.contains('pvalue')] *= results.columns.str.contains('pvalue').sum() * len(results)
    else:
        results.loc[:, results.columns.str.contains('pvalue')] *= bonferroni_factor
    results.loc[:, results.columns.str.contains('pvalue')] = np.clip(results.loc[:, results.columns.str.contains('pvalue')], 0, 1)

    return results
