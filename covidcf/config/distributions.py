from typing import Dict, Union, List, Optional

from marshmallow import fields, validate
from marshmallow_dataclass import NewType, dataclass

Categorical = NewType('Categorical', str, fields.String, required=True, validate=validate.Equal('CATEGORICAL'))


@dataclass
class CategoricalDistribution:
    distribution: Categorical
    values: Union[List[Optional[str]], List[Optional[bool]], List[Optional[int]], List[Optional[float]]]
    default: Optional[Union[str, bool, int, float]]


RandInt = NewType('RandInt', str, fields.String, required=True, validate=validate.Equal('RANDINT'))


@dataclass
class RandIntDistribution:
    distribution: RandInt
    vmin: int
    vmax: int
    default: Optional[int]


Uniform = NewType('Uniform', str, fields.String, required=True, validate=validate.Equal('UNIFORM'))


@dataclass
class UniformDistribution:
    distribution: Uniform
    vmin: float
    vmax: float
    default: Optional[float]


LogUniform = NewType('LogUniform', str, fields.String, required=True, validate=validate.Equal('LOG_UNIFORM'))


@dataclass
class LogUniformDistribution:
    distribution: LogUniform
    vmin: float
    vmax: float
    default: Optional[float]


# TODO: See if nested type can be supported with marshamllow?
ParameterSetTop = Dict[str, Union[str, bool, int, float, CategoricalDistribution, RandIntDistribution,
                                  UniformDistribution, LogUniformDistribution]]
ParameterSetMiddle = Dict[
    str, Union[str, bool, int, float, CategoricalDistribution, RandIntDistribution,
               UniformDistribution, LogUniformDistribution, ParameterSetTop]]
ParameterSetMiddle2 = Dict[
    str, Union[str, bool, int, float, CategoricalDistribution, RandIntDistribution,
               UniformDistribution, LogUniformDistribution, ParameterSetMiddle]]
ParameterSetBottom = Dict[
    str, Union[str, bool, int, float, CategoricalDistribution, RandIntDistribution,
               UniformDistribution, LogUniformDistribution, ParameterSetMiddle2]]
ParameterSet = Union[ParameterSetBottom, ParameterSetMiddle, ParameterSetMiddle2, ParameterSetTop]
