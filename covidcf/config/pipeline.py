from dataclasses import field
from enum import Enum
from typing import Optional

from sklearn.experimental import enable_iterative_imputer  # noqa

from marshmallow import fields
from marshmallow_dataclass import dataclass, NewType


class FeatureSelectionType(Enum):
    SFM = 'sfm'
    RFE = 'rfe'
    NONE = 'none'


class EstimatorType(Enum):
    LGBM = 'lgbm'
    MULTI_LGBM = 'multi_lgbm'
    LR = 'lr'
    RF = 'rf'
    # SHAP_ZERO = 'shap_zero'
    # REP_IMPUTE = 'rep_impute'


class ImputationType(Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
    CONSTANT = 'constant'
    MICE = 'mice'
    KNN3 = 'knn3'
    KNN5 = 'knn5'
    KNN10 = 'knn10'
    NONE = 'none'

    def generate_imputer(self):
        from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
        if self == ImputationType.MEAN:
            return SimpleImputer(strategy='mean')
        elif self == ImputationType.MEDIAN:
            return SimpleImputer(strategy='median')
        elif self == ImputationType.CONSTANT:
            return SimpleImputer(strategy='constant', fill_value=-1)
        elif self == ImputationType.MICE:
            return IterativeImputer(max_iter=10, n_nearest_features=40)
        elif self == ImputationType.KNN3:
            return KNNImputer(n_neighbors=3)
        elif self == ImputationType.KNN5:
            return KNNImputer(n_neighbors=5)
        elif self == ImputationType.KNN10:
            return KNNImputer(n_neighbors=10)
        elif self == ImputationType.NONE:
            return 'passthrough'
        raise ValueError(f'Invalid config: {self.value}')


@dataclass
class ImputationConfig:
    categorical: ImputationType
    numeric: ImputationType


InfDecimal = NewType('InfDecimal', float, fields.Decimal, allow_nan=True)


@dataclass
class FeatureSelectionConfig:
    fs_type: FeatureSelectionType
    num_features: Optional[int]
    threshold: Optional[InfDecimal]

    def generate_selector(self):
        from lightgbm import LGBMClassifier
        if self.fs_type == FeatureSelectionType.RFE:
            from sklearn.feature_selection import RFE
            return RFE(LGBMClassifier(n_jobs=-1), n_features_to_select=self.num_features)
        elif self.fs_type == FeatureSelectionType.SFM:
            from sklearn.feature_selection import SelectFromModel
            return SelectFromModel(LGBMClassifier(n_jobs=-1), max_features=self.num_features, threshold=self.threshold)
        elif self.fs_type == FeatureSelectionType.NONE:
            return 'passthrough'
        raise ValueError(f'Invalid config: {self.fs_type}')


class SpecialModel(Enum):
    SHAP_ZERO = 'shap_zero'
    REP_IMPUTE = 'rep_impute'


@dataclass
class PipelineConfig:
    imputation: ImputationConfig
    feature_selection: FeatureSelectionConfig
    estimator_type: EstimatorType
    special: Optional[SpecialModel]
    n_bagging: Optional[int]
    should_stop_early: bool = field(default=True)
    missing_indicators: bool = field(default=False)
