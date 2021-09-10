import collections
import json
import os
import re
from operator import gt, lt
from typing import Dict, Any, Optional, Callable, List

import click
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import wandb
import yaml
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets._base import _pkl_filepath
from sklearn.model_selection import cross_validate
from sklearn.model_selection._validation import _score
from wandb.integration.lightgbm import wandb_callback

from .models import BestThresholdClassifier
from ..util import early_stopping
from ..config.dataset import RandomSplitEvaluationConfig, DatasetTarget
from ..config import Config, ConfigSchema
from ..config.pipeline import EstimatorType
from .base import load_data, generate_pipeline, prepreprocess_data, generate_cross_validator
from .metrics import get_metrics, get_metric_callables
from ..data.base import split_data_single, get_train_df_split

Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple(
    "WorkerInitData", ("num", "run_name", "config", "X_train", "X_test",
                       "y_train", "y_test", "metrics", "pipeline")
)
WorkerDoneData = collections.namedtuple("WorkerDoneData", ("val_accuracy",))


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def train(sweep_q, worker_q):
    reset_wandb_env()
    worker_data: WorkerInitData = worker_q.get()
    run_name = "{}-{}".format(worker_data.run_name, worker_data.num)
    config = worker_data.config
    run = wandb.init(
        group=worker_data.run_name,
        name=run_name,
        config=config,
    )

    pipeline = worker_data.pipeline
    pipeline.fit(worker_data.X_train, worker_data.y_train)

    scores = _score(pipeline, worker_data.X_test, worker_data.y_test, worker_data.metrics)
    scores = {f'val_{k}': v for k, v in scores.items()}

    run.log(scores)
    wandb.join()
    sweep_q.put(scores)


def parameter_set_to_params(parameter_set: Dict, prefix: Optional[str] = None) -> Dict[str, Any]:
    """Turn parameter set nested dict into flat parameter dict for scikit-learn

    :param parameter_set: The nested parameter dict
    :param prefix: The prefix for all parameters in ``parameter_set``
    :return: A flattened (non-nested) parameter dict for scikit-learn
    """
    prefix = prefix if prefix is not None else ""
    params = {}
    for k, v in parameter_set.items():
        if isinstance(v, dict):
            params.update(parameter_set_to_params(v, prefix + k + '__'))
        else:
            params[prefix + k] = v
    return params


def evaluate(config: Config, config_dict: Dict, name: Optional[str] = None, save_path: Optional[str] = None,
             use_wandb=False, save_pipeline: bool = False, adjust_thresholds: bool = False):
    print('Loading data...')
    config.pipeline.should_stop_early = False
    df = load_data(config.dataset.name)

    df, X, y = prepreprocess_data(df, config)

    cv_test = generate_cross_validator(df, config.dataset.test.cv)
    df_train = get_train_df_split(df, X, y, cv_test)
    if config.pipeline.should_stop_early:
        if config.dataset.early_stopping is None:
            # noinspection PyArgumentList
            cv_validate = generate_cross_validator(df_train, RandomSplitEvaluationConfig(test_size=0.1))
        else:
            cv_validate = generate_cross_validator(df_train, config.dataset.early_stopping.cv)

    params = parameter_set_to_params(config.parameters)
    pipeline = generate_pipeline(X, config)
    pipeline.set_params(**params)

    metrics = get_metrics(config.dataset.metrics)
    metric_callables = get_metric_callables(config.dataset.metrics)

    if use_wandb:
        run = wandb.init(project='covid-cf', entity='rhacking', config=config_dict, name=name)

    X_train, y_train, X_test, y_test = split_data_single(X, y, cv_test)
    if config.pipeline.should_stop_early:
        X_train, y_train, X_val, y_val = split_data_single(X_train, y_train, cv_validate)

    if config.pipeline.estimator_type == EstimatorType.LGBM:
        if config.pipeline.n_bagging is not None and config.pipeline.n_bagging > 1:
            # results = cross_validate(pipeline, X, y, cv=cv_test, scoring=metrics,
            #                          error_score=0.0,
            #                          return_estimator=True)
            pipeline.fit(X_train, y_train)
            results = {'estimator': [pipeline]}

            if use_wandb:
                run.log({k: v for k, v in results.items() if k != 'estimator'})
        elif not config.pipeline.should_stop_early:
            try:
                # results = cross_validate(pipeline, X, y, cv=cv_test, scoring=metrics,
                #                          error_score='raise',
                #                          return_estimator=True, fit_params=dict(
                #         estimator__callbacks=([wandb_callback()] if use_wandb else []),
                #         estimator__eval_set=[(X_train, y_train), (X_test, y_test)],
                #         estimator__eval_names=['train', 'test'],
                #         estimator__eval_metric=['logloss'] + metric_callables))
                pipeline.fit(X_train, y_train, estimator__callbacks=([wandb_callback()] if use_wandb else []),
                             estimator__eval_set=((X_train, y_train), (X_test, y_test)),
                             estimator__eval_names=['train', 'test'],
                             estimator__eval_metric=['logloss'] + metric_callables)
                results = {'estimator': [pipeline]}
                if use_wandb:
                    run.log({'fit_time': np.mean(results['fit_time']), 'score_time': np.mean(results['score_time'])})
            except (ValueError, TypeError):
                # results = cross_validate(pipeline, X, y, cv=cv_test, scoring=metrics,
                #                          error_score='raise',
                #                          return_estimator=True)
                pipeline.fit(X_train, y_train)
                results = {'estimator': [pipeline]}
        else:
            try:
                # FIXME: Training on validation
                # FIXME: Preprocessing of train/val/test
                # results = cross_validate(pipeline, X, y, cv=cv_test, scoring=metrics,
                #                          error_score='raise',
                #                          return_estimator=True, fit_params=dict(
                #         estimator__callbacks=([wandb_callback()] if use_wandb else []) + [
                #             early_stopping(12, 'val', first_metric_only=True)],
                #         estimator__eval_set=[(X_train, y_train), (X_val, y_val), (X_test, y_test)],
                #         estimator__eval_names=['train', 'val', 'test'],
                #         estimator__eval_metric=['logloss'] + metric_callables))
                pipeline.fit(X_train, y_train, estimator__callbacks=([wandb_callback()] if use_wandb else []) + [
                            early_stopping(12, 'val', first_metric_only=True)],
                        estimator__eval_set=[(X_train, y_train), (X_val, y_val), (X_test, y_test)],
                        estimator__eval_names=['train', 'val', 'test'],
                        estimator__eval_metric=['logloss'] + metric_callables)
                results = {'estimator': [pipeline]}
                if use_wandb:
                    run.log({'fit_time': np.mean(results['fit_time']), 'score_time': np.mean(results['score_time'])})
            except (ValueError, TypeError):
                # results = cross_validate(pipeline, X, y, cv=cv_test, scoring=metrics,
                #                          error_score='raise',
                #                          return_estimator=True)
                pipeline.fit(X_train, y_train)
                results = {'estimator': [pipeline]}
    else:
        try:
            # results = cross_validate(pipeline, X, y, cv=cv_test, scoring=metrics, error_score='raise',
            #                          return_estimator=True)
            pipeline.fit(X_train, y_train)
            results = {'estimator': [pipeline]}
        except:
            try:
                # results = cross_validate(pipeline, X, y, cv=cv_test, scoring=['accuracy'], error_score=0.0,
                #                          return_estimator=True)
                pipeline.fit(X_train, y_train)
                results = {'estimator': [pipeline]}
            except:
                raise
        if use_wandb:
            run.log({k: v for k, v in results.items() if k != 'estimator'})

    def generate_plots():
        # TODO: Improve this section. Support non-tree estimators
        if (isinstance(pipeline.steps[-1][1], LGBMClassifier) or
            isinstance(pipeline.steps[-1][1], LGBMRegressor)) and not (
                config.pipeline.n_bagging is not None and config.pipeline.n_bagging > 1):
            try:
                import shap
                explainer = shap.TreeExplainer(results['estimator'][0].steps[-1][1])
                shap_values = explainer(X)
                shap.plots.beeswarm(shap_values[:, :, 1], max_display=20, show=False)
                plt.tight_layout()
                fig = plt.gcf()
                w, h = fig.get_size_inches()
                fig.set_size_inches(w * 3, h)
                wandb.log({"shap_beeswarm": wandb.Image(fig)})

                plt.figure()
                shap.plots.bar(shap_values[:, :, 1], show=False)
                plt.tight_layout()
                fig = plt.gcf()
                w, h = fig.get_size_inches()
                fig.set_size_inches(w * 3, h)
                wandb.log({"shap_bar": wandb.Image(fig)})
            except:
                pass

        if config.dataset.target == DatasetTarget.MISSING:
            wandb.sklearn.plot_summary_metrics(pipeline, X_train, X_test, y_train, y_test)
            return
        # wandb.sklearn.plot_classifier(pipeline, X_train, X_test, y_train, y_test,
        #                               results['estimator'][0].predict(X_test),
        #                               results['estimator'][0].predict_proba(X_test),
        #                               [f'feat_{i}' for i in range(y_test.shape[1])])

        y_pred = results['estimator'][0].predict_proba(X_test)
        if y_test.dtype.name == 'category':
            wandb.log({"conf_mat": wandb.plot.confusion_matrix(
                y_pred,
                y_true=list(y_test.cat.codes.values), class_names=list(y_test.cat.categories))})
        else:
            wandb.log({"conf_mat": wandb.plot.confusion_matrix(
                y_pred,
                y_true=y_test.values)})

        wandb.log({"pr": wandb.plot.pr_curve(y_test,
                                             y_pred)})

        wandb.log({"roc": wandb.plot.roc_curve(y_test,
                                               y_pred)})

        sns.heatmap(X.corr())
        fig = plt.gcf()
        wandb.log({"data_corr": wandb.Image(fig)})

        fig = px.imshow(X.corr().values,
                        x=X.columns.values,
                        y=X.columns.values
                        )
        fig.update_xaxes(side="top")

        wandb.log({"data_corr": fig})

        sns.heatmap(X.isna())
        fig = plt.gcf()
        wandb.log({"missingness": wandb.Image(fig)})

    if use_wandb:
        generate_plots()

        if config.dataset.target != DatasetTarget.MISSING:
            results_table = wandb.Table(columns=['true', 'predicted', 'predicted_proba'],
                                        data=np.array([y_train, results['estimator'][0].predict(X_train),
                                                       results['estimator'][0].predict_proba(X_train)[:, 1]]).T)
            run.log({'results_train': results_table})
            # results_table = wandb.Table(columns=['true', 'predicted', 'predicted_proba'],
            #                             data=np.array([y_val, results['estimator'][0].predict(X_val),
            #                                            results['estimator'][0].predict_proba(X_val)[:, 1]]).T)
            # run.log({'results_val': results_table})
            results_table = wandb.Table(columns=['true', 'predicted', 'predicted_proba'],
                                        data=np.array([y_test, results['estimator'][0].predict(X_test),
                                                       results['estimator'][0].predict_proba(X_test)[:, 1]]).T)
            run.log({'results_test': results_table})

        # TODO: Log estimator artifact ig
        wandb.join()

    if save_path is not None:
        if adjust_thresholds and y.ndim == 1:
            data_to_save = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'preds': BestThresholdClassifier(results['estimator'][0]).fit(X_train, y_train).predict(X_test),
                'preds_proba': results['estimator'][0].predict_proba(X_test)[:,
                               1] if config.dataset.target != DatasetTarget.MISSING else results['estimator'][
                    0].predict_proba(X_test),
                # 'X_val': X_val,
                # 'y_val': y_val,
                # 'preds_val': BestThresholdClassifier(results['estimator'][0]).fit(X_train, y_train).predict(X_val),
                # 'preds_val_proba': results['estimator'][0].predict_proba(X_val)[:,
                #                1] if config.dataset.target != DatasetTarget.MISSING else results['estimator'][
                #     0].predict_proba(X_val),
            }
        else:
            data_to_save = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'preds': results['estimator'][0].predict(X_test),
                'preds_proba': results['estimator'][0].predict_proba(X_test)[:,
                               1] if config.dataset.target != DatasetTarget.MISSING else results['estimator'][
                    0].predict_proba(X_test),
                # 'X_val': X_val,
                # 'y_val': y_val,
                # 'preds_val': results['estimator'][0].predict(X_val),
                # 'preds_val_proba': results['estimator'][0].predict_proba(X_val)[:,
                #                1] if config.dataset.target != DatasetTarget.MISSING else results['estimator'][
                #     0].predict_proba(X_val)
            }
        if save_pipeline:
            data_to_save.update({'pipeline': results['estimator'][0]})
        joblib.dump(data_to_save, save_path)

    if use_wandb:
        return run.id


@click.command(name='evaluate')
@click.argument('config-path', type=click.Path(exists=True, dir_okay=False))
@click.option('--modified-config', type=click.Path(exists=True, dir_okay=False))
def evaluate_command(config_path: str, modified_config: Optional[str]):
    with open(config_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)
    if modified_config is not None:
        with open(modified_config, 'r') as f:
            config_mod_dict = json.load(f)
            print(config_mod_dict)
            for k, v in config_mod_dict.items():
                keys = k.split('.')
                current = config_dict
                for key in keys[:-1]:
                    current = current[key]
                current[keys[-1]] = v
    print(config_dict)
    config = ConfigSchema.load(config_dict)
    evaluate(config, config_dict)


@click.command(name='evaluate-hyperopt')
@click.argument('pkl-path', type=click.Path(exists=True, dir_okay=False))
def hyperopt_evaluate_command(pkl_path: str):
    study = joblib.load(pkl_path)
    path_to_file, filename = os.path.split(pkl_path)
    config_path = path_to_file + '/' + filename.split('__')[1].split('.')[0] + '.yaml'
    with open(config_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    for k, v in study.best_params.items():
        keys = k.split('__')
        current = config_dict['parameters']
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = v
    del config_dict['n_trials']
    del config_dict['dataset']['validation']

    config = ConfigSchema.load(config_dict)
    evaluate(config, config_dict)
