import os
import re
from typing import Union

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import MissingIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, PredefinedSplit, ShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from covidcf.config import HyperOptConfig, Config
from covidcf.config.dataset import CV, DatasetConfig, KFoldEvaluationConfig, RepeatedKFoldEvaluationConfig, \
    PartEvaluationConfig, RandomSplitEvaluationConfig, DatasetTarget
from covidcf.config.pipeline import ImputationType, FeatureSelectionType, EstimatorType, SpecialModel
from covidcf.data.base import DATA_PROCESSED_PATH, aggregate_features
from covidcf.evaluation.models import ShapZeroingClassifier, TestTimeImputingClassifier, ColumnImputer, \
    MultiOutputClassifier


def prepreprocess_data(df: pd.DataFrame, config: Union[Config, HyperOptConfig]):
    if config.dataset.should_aggregate:
        # FIXME: Only aggregate inputs
        df = aggregate_features(df)

    _use_features(df, df.Input.Visual.columns, config.dataset.use_visual_features)
    _use_features(df, df.Input.Clinical.columns, config.dataset.use_clinical_features)

    if config.dataset.target != DatasetTarget.MISSING:
        y = df.Target[config.dataset.target.value]
        df = df[~y.isna()]
        y = y[~y.isna()]
        X = df.Input

        if config.dataset.feature_value_fraction_required > 0.0:
            has_sufficient_values = (1 - X.isna().mean() >= config.dataset.feature_value_fraction_required)
            X = X.loc[:, has_sufficient_values]

        for col in X.select_dtypes(include=['category']):
            X[col] = X[col].cat.codes
            X.loc[X[col] < 0, col] = np.nan
            X[col] = X[col].astype('category')

        if y.apply(type).eq(str).any():
            y = y.astype('category').cat.codes
    else:
        X = df.Input
        for col in X.select_dtypes(include=['category']):
            X[col] = X[col].cat.codes
            X.loc[X[col] < 0, col] = np.nan
            X[col] = X[col].astype('category')
        y = pd.DataFrame(np.isnan(np.array(X)))

    X.columns = [re.sub(r'[^0-9a-zA-Z_-]+', '', '_'.join(col).strip()) for col in X.columns.values]
    X.columns = pd.io.parsers.ParserBase({'names': X.columns})._maybe_dedup_names(X.columns)

    if isinstance(y, pd.DataFrame):
        for col in y.columns:
            if len(y[col].unique()) == 1:
                y.drop(col, inplace=True, axis=1)

    return df, X, y


def _use_features(df: pd.DataFrame, columns: pd.Index, use_features: Union[bool, list]):
    if use_features is False:
        df.drop(columns=columns, level=2, inplace=True)
    elif isinstance(use_features, list):
        df.drop(columns=list(set(columns) - set(use_features)), level=2,
                inplace=True)


def generate_pipeline(X: pd.DataFrame, config: Union[Config, HyperOptConfig]):
    # FIXME: Bagging/preprocessing throws away data
    if config.pipeline.estimator_type == EstimatorType.LGBM:
        estimator = LGBMClassifier(n_jobs=-1) if config.dataset.target.is_classification else LGBMRegressor(
            n_jobs=-1)
    elif config.pipeline.estimator_type == EstimatorType.LR:
        # FIXME: Add scaling!!!!!
        estimator = LogisticRegression()
    elif config.pipeline.estimator_type == EstimatorType.RF:
        estimator = RandomForestClassifier() if config.dataset.target.is_classification else RandomForestRegressor()

    elif config.pipeline.estimator_type == EstimatorType.MULTI_LGBM:
        estimator = MultiOutputClassifier(LGBMClassifier(n_jobs=-1))
    else:
        raise ValueError(f'Cannot handle estimator type: {config.pipeline.estimator_type}')

    if config.pipeline.special == SpecialModel.REP_IMPUTE:
        # if config.pipeline.n_bagging is not None and config.pipeline.n_bagging > 1:
        #     estimator = TestTimeImputingClassifier(BaggingClassifier(LGBMClassifier(n_jobs=-1), n_estimators=config.pipeline.n_bagging), ColumnImputer(), n_repeat=100)
        # else:
        estimator = TestTimeImputingClassifier(estimator, ColumnImputer(), n_repeat=100)
    elif config.pipeline.special == SpecialModel.SHAP_ZERO:
        estimator = ShapZeroingClassifier(estimator)


    if (config.pipeline.imputation.numeric == ImputationType.NONE and
            config.pipeline.imputation.categorical == ImputationType.NONE and
            config.pipeline.feature_selection.fs_type == FeatureSelectionType.NONE):

        if config.pipeline.missing_indicators:
            pipeline = Pipeline([
                ('imputation', MissingIndicator(features='all')),
                ('estimator', estimator)
            ])
        else:
            pipeline = Pipeline([
                ('estimator', estimator)
            ])

        if config.pipeline.n_bagging is not None and config.pipeline.n_bagging > 1:# and config.pipeline.estimator_type != EstimatorType.REP_IMPUTE:
            if config.dataset.target.is_classification:
                return BaggingClassifier(pipeline, n_estimators=config.pipeline.n_bagging)
            else:
                return BaggingRegressor(pipeline, n_estimators=config.pipeline.n_bagging)

    if config.pipeline.missing_indicators:
        pipeline = Pipeline([
            ('imputation', MissingIndicator(features='all')),
            ('feature_selection', config.pipeline.feature_selection.generate_selector()),
            ('estimator', estimator)
        ])
    else:
        categorical_columns = list(X.select_dtypes(include=['category']).columns)
        numerical_columns = list(X.select_dtypes(exclude=['category']).columns)

        cateogrical_preprocessor = config.pipeline.imputation.categorical.generate_imputer()
        if config.pipeline.estimator_type == EstimatorType.LR:
            numerical_preprocessor = Pipeline([
                ('imputer', config.pipeline.imputation.numeric.generate_imputer()),
                ('scaler', StandardScaler())
            ])
        else:
            numerical_preprocessor = config.pipeline.imputation.numeric.generate_imputer()

        if config.pipeline.n_bagging is not None and config.pipeline.n_bagging > 1:
            imputation = numerical_preprocessor
        else:
            imputation = ColumnTransformer([
                ('categorical', cateogrical_preprocessor, categorical_columns),
                ('numerical', numerical_preprocessor, numerical_columns)
            ])

        pipeline = Pipeline([
            ('imputation', imputation),
            ('feature_selection', config.pipeline.feature_selection.generate_selector()),
            ('estimator', estimator)
        ])

    if config.pipeline.n_bagging is not None and config.pipeline.n_bagging > 1:
        if config.dataset.target.is_classification:
            pipeline = BaggingClassifier(pipeline, n_estimators=config.pipeline.n_bagging)
        else:
            pipeline = BaggingRegressor(pipeline, n_estimators=config.pipeline.n_bagging)

    return pipeline


def load_data(dataset_name: str) -> pd.DataFrame:
    return pd.read_pickle(os.path.join(DATA_PROCESSED_PATH, dataset_name + '.pkl'))


def generate_cross_validator(df: pd.DataFrame, config_cv: CV):
    if isinstance(config_cv, KFoldEvaluationConfig):
        return StratifiedKFold(n_splits=config_cv.n_folds, shuffle=True)
    elif isinstance(config_cv, RepeatedKFoldEvaluationConfig):
        return RepeatedStratifiedKFold(n_splits=config_cv.n_folds, n_repeats=config_cv.n_repeats)
    elif isinstance(config_cv, PartEvaluationConfig):
        part_name = config_cv.part_name
        if isinstance(config_cv.part, str):
            return PredefinedSplit((df.Meta[part_name] == config_cv.part) - 1)
        else:
            return PredefinedSplit((df.Meta[part_name].isin(config_cv.part)) - 1)
    elif isinstance(config_cv, RandomSplitEvaluationConfig):
        if config_cv.stratified:
            return StratifiedShuffleSplit(n_splits=1, test_size=config_cv.test_size, random_state=config_cv.seed)
        else:
            return ShuffleSplit(n_splits=1, test_size=config_cv.test_size, random_state=config_cv.seed)

def get_cat_codes(x: pd.Series):
    if x.dtype.name == 'category':
        return x.cat.codes
    else:
        return x.values

