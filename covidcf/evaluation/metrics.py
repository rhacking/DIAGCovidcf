from typing import List, Dict

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer, get_scorer
from sklearn.metrics._scorer import _ProbaScorer, _ThresholdScorer, _BaseScorer


def thresh_accuracy(y, y_pred):
    y_thresh = y >= 3
    y_pred_thresh = y_pred >= 3
    return accuracy_score(y_thresh, y_pred_thresh)


def corads_roc_auc(y: np.ndarray, y_pred: np.ndarray, threshold: int = 3):
    y_thresh = y >= threshold
    y_pred_thresh = y_pred / 4
    return roc_auc_score(y_thresh, y_pred_thresh)


custom_metric_callables = {
    'thresh_accuracy': thresh_accuracy,
    'corads_roc_auc': corads_roc_auc
}

custom_metrics = {k: make_scorer(v) for k, v in custom_metric_callables.items()}


def get_metric(metric_name: str):
    return custom_metrics[metric_name] if metric_name in custom_metrics else get_scorer(metric_name)


def get_metrics(metric_names: List[str]):
    return {metric: get_metric(metric) for metric in metric_names}


def get_metric_callable(metric_name: str):
    scorer: _BaseScorer = custom_metrics[metric_name] if metric_name in custom_metrics else get_scorer(metric_name)

    def callable_no_thresh(y_true, y_pred, *args, **kwargs):
        return metric_name, scorer._score_func(y_true, np.round(y_pred)), scorer._sign == 1

    def callable_thresh(y_true, y_pred):
        return metric_name, scorer._score_func(y_true, y_pred), scorer._sign == 1

    return callable_thresh if isinstance(scorer, _ProbaScorer) or isinstance(scorer,
                                                                             _ThresholdScorer) else callable_no_thresh


def get_metric_callables(metric_names: List[str]):
    return [get_metric_callable(metric) for metric in metric_names]


def get_metric_callables_from_dict(metric_names: Dict[str, str]):
    return {m_name: (custom_metrics[metric] if metric in custom_metrics else get_scorer(metric))._score_func for m_name, metric in metric_names.items()}