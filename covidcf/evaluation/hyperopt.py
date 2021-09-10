import os
import re
import warnings
from typing import Dict, Any, Optional
import pandas as pd

import click
import joblib
import numpy as np
import yaml
from optuna import Trial, create_study
from optuna.samplers import TPESampler
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import cross_validate

from ..config import HyperOptConfig, HyperOptConfigSchema
from ..config.distributions import CategoricalDistribution, RandIntDistribution, UniformDistribution, \
    LogUniformDistribution, ParameterSet
from ..data.base import aggregate_features, DATA_RESULTS_MODELS_OPTUNA_PATH, ensure_data_dirs, DATA_RESULTS_PATH, \
    split_data_single
from .base import load_data, generate_cross_validator, generate_pipeline, prepreprocess_data
from .metrics import get_metric, get_metrics


def hyperopt(config: HyperOptConfig, path: str):
    ensure_data_dirs()

    print('Loading data...')
    df = load_data(config.dataset.name)

    df, X, y = prepreprocess_data(df, config)

    cv_test = generate_cross_validator(df, config.dataset.test.cv)
    X_train, y_train, X_test, y_test = split_data_single(X, y, cv_test)
    cv_validate = generate_cross_validator(df, config.dataset.validation.cv)
    validation_metric = get_metric(config.dataset.validation.metric)

    # TODO: Check categorical for each dataset
    # assert len(df.select_dtypes(include='category').columns) > 0

    def objective(trial: Trial):
        params = parameter_set_to_params(config.parameters, trial)

        to_drop = []
        if config.optimize_features:
            for col in X.columns:
                use_col = trial.suggest_categorical(f'use_col_{col}', [False, True])
                if not use_col:
                    to_drop.append(col)

        X_reduced = X_train.drop(columns=to_drop)

        if len(X_reduced.columns) == 0:
            return 0.0

        pipeline = generate_pipeline(X_reduced, config)
        pipeline.set_params(**params)

        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore', category=FitFailedWarning)
        results = cross_validate(pipeline, X_reduced, y_train, cv=cv_validate, scoring=validation_metric, error_score=0.0)

        return results['test_score'].mean()

    path_to_file, filename = os.path.split(path)
    _, file_dir = os.path.split(path_to_file)
    study_name = f'{file_dir}__{filename.split(".")[0]}'
    print(study_name)

    # FIXME: Maximize is not necessarily correct
    study = create_study(direction='maximize', study_name=study_name,
                         sampler=TPESampler(multivariate=True))
    default = parameter_set_to_params(config.parameters, None)
    # FIXME: Check if defaults exist
    for i in range(1):
        study.enqueue_trial(default)

    try:
        study.optimize(objective, n_trials=config.n_trials)
    except BaseException as e:
        joblib.dump(study, os.path.join(path_to_file, f'{filename.split(".")[0]}.pkl'))
        raise e

    joblib.dump(study, os.path.join(path_to_file, f'{filename.split(".")[0]}.pkl'))


def sample_value(k, v, trial):
    if isinstance(v, str) or isinstance(v, bool) or isinstance(v, int) or isinstance(v, float):
        return v

    if trial is None:
        return v.default

    if isinstance(v, CategoricalDistribution):
        return trial.suggest_categorical(k, v.values)
    elif isinstance(v, RandIntDistribution):
        return trial.suggest_int(k, v.vmin, v.vmax)
    elif isinstance(v, UniformDistribution):
        return trial.suggest_uniform(k, v.vmin, v.vmax)
    elif isinstance(v, LogUniformDistribution):
        return trial.suggest_loguniform(k, v.vmin, v.vmax)

    return v


def parameter_set_to_params(parameter_set: ParameterSet, trial: Optional[Trial], prefix: Optional[str] = None) -> Dict[str, Any]:
    prefix = prefix if prefix is not None else ""
    params = {}
    for k, v in parameter_set.items():
        if isinstance(v, dict):
            params.update(parameter_set_to_params(v, trial, prefix+k+'__'))
        else:
            params[prefix+k] = sample_value(prefix+k, v, trial)
    return params


@click.command(name='hyperopt')
@click.argument('config-path', type=click.Path(exists=True, dir_okay=False))
def hyperopt_command(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    config = HyperOptConfigSchema.load(config)
    hyperopt(config, config_path)
