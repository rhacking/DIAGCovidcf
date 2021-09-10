import copy
import os
import re
import tempfile
from typing import Optional, Any, Dict, List

import click
import joblib
import numpy as np
import wandb
import yaml
from sklearn.pipeline import Pipeline

from covidcf.config import ConfigSchema, Config
from covidcf.config.pipeline import EstimatorType
from covidcf.evaluation.base import generate_pipeline, generate_cross_validator, prepreprocess_data, load_data
from covidcf.evaluation.evaluation import parameter_set_to_params
from covidcf.evaluation.metrics import get_metric_callables


def _flatten_dict(d: Dict[str, Any], result: Optional[Dict[str, Any]] = None, exceptions: List[str] = None):
    result = {} if result is None else result
    exceptions = [] if exceptions is None else exceptions
    for k, v in d.items():
        if isinstance(v, dict) and k not in exceptions:
            _flatten_dict(v, result=result, exceptions=exceptions)
        else:
            result[k] = v
    return result


def hyperopt(base_config: Config, sweep_config, count: int, resume_sweep_id: Optional[str]):
    # TODO: Add better logging ig
    print('Loading data...')
    raw_df = load_data(base_config.dataset)

    metric_callables = get_metric_callables(base_config.dataset.test.metrics)

    def train():
        with wandb.init() as run:
            config_dict = ConfigSchema.dump(copy.deepcopy(base_config))

            # artifact = wandb.Artifact('base_config', type='config')
            # artifact.add_file(config_path)
            # wandb.log_artifact(artifact)
            #
            # artifact = wandb.Artifact('sweep_config', type='config')
            # artifact.add_file(sweep_config_path)
            # wandb.log_artifact(artifact)

            # TODO: Do this every iteration?
            df = prepreprocess_data(raw_df, base_config)

            # TODO: Abstract this?
            y = df.Target[base_config.dataset.target.value]
            df = df[~y.isna()]
            y = y[~y.isna()]
            X = df.Input

            # TODO: Abstract this away
            for col in X.select_dtypes(include=['category']):
                X[col] = X[col].cat.codes
                X.loc[X[col] < 0, col] = np.nan

            X.columns = [re.sub(r'[^0-9a-zA-Z_-]+ ', '', '_'.join(col).strip()) for col in X.columns.values]

            cv_test = generate_cross_validator(df, base_config.dataset.test.cv)
            cv_validate = generate_cross_validator(df, base_config.dataset.validation.cv)

            train_index, test_index = list(cv_test.split(X, y))[0]
            X_not_test, X_test = X.iloc[train_index], X.iloc[test_index]
            y_not_test, y_test = y.iloc[train_index], y.iloc[test_index]

            artifact = wandb.Artifact(base_config.dataset.name, type='dataset')
            artifact.add(wandb.Table(dataframe=X_not_test), name='X_not_test')
            artifact.add(wandb.Table(dataframe=y_not_test.to_frame()), name='y_not_test')
            artifact.add(wandb.Table(dataframe=X_test), name='X_test')
            artifact.add(wandb.Table(dataframe=y_test.to_frame()), name='y_test')
            wandb.log_artifact(artifact)

            wandb_config = _flatten_dict(config_dict, exceptions=['validation', 'test'])
            columns = []
            for k, v in wandb.config.items():
                keys = k.split('.')
                if k.startswith('use_'):
                    # FIXME: Make this deal with changes in column names properly
                    if v and k.replace('use_', '') in X.columns:
                        columns.append(k.replace('use_', ''))
                elif len(keys) > 1:
                    current = config_dict
                    for key in keys[:-1]:
                        current = current[key]
                    current[keys[-1]] = v
                    wandb_config[keys[-1]] = v
                else:
                    config_dict['parameters']['estimator'][k] = v
                    wandb_config[k] = v

            config = ConfigSchema.load(config_dict)
            params = parameter_set_to_params(config.parameters)
            pipeline = generate_pipeline(X[columns], config)
            pipeline.set_params(**params)

            metric_hist = {}
            for train_ind, test_ind in cv_validate.split(X_not_test, y_not_test):
                X_train, y_train = X_not_test.iloc[train_ind], y_not_test.iloc[train_ind]
                X_val, y_val = X_not_test.iloc[test_ind], y_not_test.iloc[test_ind]

                # FIXME: Why weird naming?
                X_train = X_train[columns]
                X_val = X_val[columns]
                X_test_reduced = X_test[columns]

                if len(pipeline.steps) > 1:
                    preprocess = Pipeline(pipeline.steps[:-1])
                    X_train = preprocess.fit_transform(X_train, y_train)
                    X_val = preprocess.transform(X_val)
                    X_test_reduced = preprocess.transform(X_test_reduced)

                estimator = Pipeline(pipeline.steps[-1:])

                # TODO: Early stopping?
                if config.pipeline.estimator_type == EstimatorType.LGBM:
                    estimator.fit(X_train, y_train,
                                  estimator__eval_set=[(X_train, y_train), (X_val, y_val), (X_test_reduced, y_test)],
                                  estimator__eval_names=['train', 'val', 'test'],
                                  estimator__eval_metric=['logloss'] + metric_callables,
                                  estimator__verbose=-1)

                    results = estimator.steps[-1][1].evals_result_
                    for part, m in results.items():
                        if part not in metric_hist:
                            metric_hist[part] = {}
                        for k, v in results[part].items():
                            metric_hist[part][k] = metric_hist[part][k] + [v] if k in metric_hist[part] else [v]
                elif config.pipeline.estimator_type == EstimatorType.LR:
                    estimator.fit(X_train, y_train)
                    if len(metric_hist) == 0:
                        metric_hist = {'train': {}, 'val': {}, 'test': {}}
                    # TODO: Difficult to read, clean up
                    for k, v in {name: [metr(y_train.cat.codes, estimator.predict_proba(X_train)[:, 1])[1]] for
                                 name, metr in zip(base_config.dataset.test.metrics, metric_callables)}.items():
                        metric_hist['train'][k] = metric_hist['train'][k] + [v] if k in metric_hist['train'] else [v]
                    for k, v in {name: [metr(y_val.cat.codes, estimator.predict_proba(X_val)[:, 1])[1]] for name, metr
                                 in zip(base_config.dataset.test.metrics, metric_callables)}.items():
                        metric_hist['val'][k] = metric_hist['val'][k] + [v] if k in metric_hist['val'] else [v]
                    for k, v in {name: [metr(y_test.cat.codes, estimator.predict_proba(X_test_reduced)[:, 1])[1]] for
                                 name, metr in zip(base_config.dataset.test.metrics, metric_callables)}.items():
                        metric_hist['test'][k] = metric_hist['test'][k] + [v] if k in metric_hist['test'] else [v]

            metric_mean_hists = {}
            max_len = 0
            for part, m in metric_hist.items():
                for k, v in m.items():
                    hist = np.mean(v, axis=0)
                    metric_mean_hists[f'cv_{part}_{k}'] = hist
                    max_len = max(max_len, len(hist))

            X_not_test_reduced = X_not_test[columns]
            X_test_reduced = X_test[columns]

            if len(pipeline.steps) > 1:
                preprocess = Pipeline(pipeline.steps[:-1])
                X_not_test_reduced = preprocess.fit_transform(X_not_test_reduced, y_not_test)
                X_test_reduced = preprocess.transform(X_test_reduced)

            estimator = Pipeline(pipeline.steps[-1:])

            # TODO: Early stopping?
            if config.pipeline.estimator_type == EstimatorType.LGBM:
                estimator.fit(X_not_test_reduced, y_not_test,
                              estimator__eval_set=[(X_not_test_reduced, y_not_test), (X_test_reduced, y_test)],
                              estimator__eval_names=['train', 'test'],
                              estimator__eval_metric=['logloss'] + metric_callables,
                              estimator__verbose=-1)

                results = estimator.steps[-1][1].evals_result_
                for part, m in results.items():
                    if part not in metric_hist:
                        metric_hist[part] = {}
                    for k, v in results[part].items():
                        metric_hist[part][k] = metric_hist[part][k] + [v] if k in metric_hist[part] else [v]

                results = pipeline.steps[-1][1].evals_result_
                for part, m in results.items():
                    for k, v in results[part].items():
                        max_len = max(max_len, len(v))
                        metric_mean_hists[f'{part}_{k}'] = v
            elif config.pipeline.estimator_type == EstimatorType.LR:
                estimator.fit(X_not_test_reduced, y_not_test)
                if len(metric_hist) == 0:
                    metric_hist = {'train': {}, 'val': {}, 'test': {}}
                for k, v in {name: [metr(y_not_test.cat.codes, estimator.predict_proba(X_not_test_reduced)[:, 1])[1]]
                             for name, metr in
                             zip(base_config.dataset.test.metrics, metric_callables)}.items():
                    metric_hist['train'][k] = metric_hist['train'][k] + [v] if k in metric_hist['train'] else [v]
                for k, v in {name: [metr(y_test.cat.codes, estimator.predict_proba(X_test_reduced)[:, 1])[1]] for
                             name, metr in zip(base_config.dataset.test.metrics, metric_callables)}.items():
                    metric_hist['test'][k] = metric_hist['test'][k] + [v] if k in metric_hist['test'] else [v]

                for part, m in metric_hist.items():
                    for k, v in m.items():
                        hist = np.mean(v, axis=0)
                        metric_mean_hists[f'{part}_{k}'] = hist
                        max_len = max(max_len, len(hist))

            with tempfile.TemporaryDirectory() as temp_dir:
                joblib.dump(estimator, os.path.join(temp_dir, 'estimator.joblib'))
                artifact = wandb.Artifact('estimator', type='model', metadata=config_dict)
                artifact.add_dir(temp_dir)
                wandb.log_artifact(artifact)

            for i in range(max_len):
                wandb.log({k: v[i] for k, v in metric_mean_hists.items() if i < len(v)})


    # FIXME: Add base config here!!!!!!
    for col in raw_df.Input.columns:
        real_col = re.sub(r'[^0-9a-zA-Z_-]+ ', '', '_'.join(col).strip())
        sweep_config['parameters'][f'use_{real_col}'] = dict(
            distribution='categorical',
            values=[False, True]
        )
    sweep_id = wandb.sweep(sweep_config, project='covid-cf') if resume_sweep_id is None else resume_sweep_id

    wandb.agent(sweep_id, project='covid-cf', function=train, count=count)


@click.command(name='hyperopt')
@click.argument('config-path', type=click.Path(exists=True, dir_okay=False))
@click.argument('sweep-config-path', type=click.Path(exists=True, dir_okay=False))
@click.option('--count', default=10, type=int, help='The number of hyperparameter optimization iterations to run')
@click.option('--name', required=False, type=str, help='The name to use for the sweep, overrides the one specified in '
                                                       'the sweep config file')
@click.option('--resume-sweep-id', required=False, type=str, help='If specified, resumes the sweep with this id. '
                                                                  'Otherwise, a new sweep is created. ')
def hyperopt_command(config_path: str, sweep_config_path: str, count: int, name: Optional[str], resume_sweep_id: Optional[str]):
    """Performs hyperparameter optimization based on the base config at CONFIG_PATH and sweep config at
    SWEEP_CONFIG_PATH"""

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.load(f, Loader=yaml.SafeLoader)
    config = ConfigSchema.load(config)
    if name is not None:
        sweep_config['name'] = name
    hyperopt(config, sweep_config, count, resume_sweep_id)
