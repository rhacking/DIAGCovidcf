from __future__ import annotations

import ast
import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
# TODO: Use path objects rather than strings?
from sklearn.model_selection import BaseCrossValidator, StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold, \
    RepeatedStratifiedKFold, RepeatedKFold

DATA_PATH = 'data'
DATA_RAW_PATH = os.path.join(DATA_PATH, 'raw')
DATA_PROCESSED_PATH = os.path.join(DATA_PATH, 'processed')
DATA_RESULTS_PATH = os.path.join(DATA_PATH, 'results')
DATA_RESULTS_MODELS_PATH = os.path.join(DATA_RESULTS_PATH, 'models')
DATA_RESULTS_MODELS_OPTUNA_PATH = os.path.join(DATA_RESULTS_MODELS_PATH, 'optuna')


def ensure_data_dirs():
    from pathlib import Path
    Path(DATA_RAW_PATH).mkdir(parents=True, exist_ok=True)
    Path(DATA_PROCESSED_PATH).mkdir(parents=True, exist_ok=True)
    Path(DATA_RESULTS_PATH).mkdir(parents=True, exist_ok=True)
    Path(DATA_RESULTS_MODELS_PATH).mkdir(parents=True, exist_ok=True)
    Path(DATA_RESULTS_MODELS_OPTUNA_PATH).mkdir(parents=True, exist_ok=True)


def load_visual_features(path):
    visual_features = pd.read_csv(path)
    visual_features.features = visual_features.features.str.replace(r'[^\[] +', ', ', regex=True).apply(
        ast.literal_eval)
    vf = visual_features.features.apply(pd.Series)
    vf.columns = vf.columns.map('vis_feature_{}'.format)
    visual_features = pd.concat([visual_features[['scan_id']], vf], axis=1)
    visual_features['patientprimarymrn'] = pd.to_numeric(visual_features.scan_id.str.extract(r'(\d+)_st\d+').iloc[:, 0])
    visual_features['study'] = visual_features.scan_id.str.extract(r'\d+_(st\d+)').iloc[:, 0]
    visual_features.drop(columns=['scan_id'], inplace=True)
    visual_features.set_index(['patientprimarymrn', 'study'], inplace=True)
    return visual_features


def make_sklearn_friendly(df: pd.DataFrame):
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].cat.codes
        df[col] = df[col].astype('category')


def make_lightgbm_friendly(df: pd.DataFrame):
    df.columns = [re.sub(r'[^0-9a-zA-Z_-]+ ', '', '_'.join(col).strip()) for col in df.columns.values]


def split_data_single(X: pd.DataFrame, y: pd.Series, cv: BaseCrossValidator) -> Tuple[
        pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    try:
        train_index, test_index = list(cv.split(X, y))[0]
    except ValueError:
        # TODO: Do something nicer here?
        if isinstance(cv, StratifiedShuffleSplit):
            cv = ShuffleSplit(n_splits=cv.n_splits, random_state=cv.random_state, test_size=cv.test_size,
                              train_size=cv.train_size)
        elif isinstance(cv, StratifiedKFold):
            cv = KFold(n_splits=cv.n_splits, shuffle=cv.shuffle,
                       random_state=cv.random_state)
        elif isinstance(cv, RepeatedStratifiedKFold):
            cv = RepeatedKFold(n_splits=cv.cv.n_splits, n_repeats=cv.n_repeats, random_state=cv.random_state)
        else:
            raise
        train_index, test_index = list(cv.split(X, y))[0]
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    return X_train, y_train, X_test, y_test


def get_train_df_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, cv: BaseCrossValidator) -> pd.DataFrame:
    try:
        train_index, test_index = list(cv.split(X, y))[0]
    except ValueError:
        # TODO: Do something nicer here?
        if isinstance(cv, StratifiedShuffleSplit):
            cv = ShuffleSplit(n_splits=cv.n_splits, random_state=cv.random_state, test_size=cv.test_size, train_size=cv.train_size)
        elif isinstance(cv, StratifiedKFold):
            cv = KFold(n_splits=cv.n_splits, shuffle=cv.shuffle,
                 random_state=cv.random_state)
        elif isinstance(cv, RepeatedStratifiedKFold):
            cv = RepeatedKFold(n_splits=cv.cv.n_splits, n_repeats=cv.n_repeats, random_state=cv.random_state)
        else:
            raise
        train_index, test_index = list(cv.split(X, y))[0]
    return df.iloc[train_index]


def aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    from pandas.api.types import is_numeric_dtype

    df_merged = df.copy()
    df_merged.columns = df.columns.set_levels(df.columns.levels[2].str.replace(r'\(.+?\)', '', regex=True).str.strip(),
                                              level=2, verify_integrity=False)

    def aggregate(x: pd.Series):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        if is_numeric_dtype(x.dtypes[0]):
            return x.mean(axis=1)
        result = x.apply(lambda y: next(z for z in y.values if not pd.isna(z)) if not y.isna().all() else np.nan,
                         axis=1)
        return result

    df = df_merged.groupby(df_merged.columns, axis=1).agg(aggregate)
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    for col in df[['Input']].columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
        except ValueError:
            df[col] = df[col].astype('category')
    return df
