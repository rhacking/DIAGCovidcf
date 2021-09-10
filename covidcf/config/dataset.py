from dataclasses import field
from enum import Enum
from typing import List, Union, Any, Optional

import sklearn.metrics
from marshmallow import validate, fields
from marshmallow_dataclass import dataclass, NewType

Metric = NewType('Metric', str, fields.String, required=True,
                 validate=validate.OneOf(['thresh_accuracy', 'corads_roc_auc'] + list(sklearn.metrics.SCORERS.keys())))


@dataclass
class KFoldEvaluationConfig:
    n_folds: int = field(metadata={'validate': validate.Range(min=2)})


@dataclass
class RepeatedKFoldEvaluationConfig:
    n_folds: int = field(metadata={'validate': validate.Range(min=2)})
    n_repeats: int = field(metadata={'validate': validate.Range(min=1)})


@dataclass
class PartEvaluationConfig:
    part: Union[str, List[str]]
    part_name: str = field(default='part')


@dataclass
class RandomSplitEvaluationConfig:
    test_size: float = field(metadata=dict(validate=validate.Range(min=0, max=1)))
    seed: Optional[int] = field(default=None)
    stratified: bool = field(default=True)


CV = Union[KFoldEvaluationConfig, RepeatedKFoldEvaluationConfig, PartEvaluationConfig, RandomSplitEvaluationConfig]
# CVSingle = Union[PartEvaluationConfig, RandomSplitEvaluationConfig]


@dataclass
class ValidationConfig:
    cv: CV
    metric: Metric


@dataclass
class TestConfig:
    cv: CV


class DatasetTarget(Enum):
    PCR = 'pcr'
    CORADS = 'corads'
    DIAGNOSIS = 'diagnosis'
    PCR_CORADS = "pcr_corads"
    MISSING = 'missing'

    @property
    def is_classification(self):
        return self == DatasetTarget.PCR or self == DatasetTarget.DIAGNOSIS or self == DatasetTarget.PCR_CORADS or self == DatasetTarget.MISSING


@dataclass
class EarlyStoppingConfig:
    metric: Metric
    cv: CV


@dataclass
class DatasetConfig:
    name: str
    target: DatasetTarget
    test: TestConfig
    metrics: List[Metric]

    should_aggregate: bool = field(default=False)
    # TODO: Add proper type?
    use_clinical_features: Any = field(default=True)
    use_visual_features: Any = field(default=True)
    feature_value_fraction_required: float = field(default=0.0, metadata={"validate": validate.Range(min=0, max=1)})
    early_stopping: Optional[EarlyStoppingConfig] = field(default=None)


@dataclass
class DatasetHyperoptConfig:
    name: str
    target: DatasetTarget
    validation: ValidationConfig
    test: TestConfig
    metrics: List[Metric]

    should_aggregate: bool = field(default=False)
    # TODO: Add proper type?
    use_clinical_features: Any = field(default=True)
    use_visual_features: Any = field(default=True)
    feature_value_fraction_required: float = field(default=0.0, metadata={"validate": validate.Range(min=0, max=1)})
    early_stopping: Optional[EarlyStoppingConfig] = field(default=None)
