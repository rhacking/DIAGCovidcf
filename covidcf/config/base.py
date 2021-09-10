from dataclasses import field
from typing import Dict, Any, Optional

import yaml
from marshmallow_dataclass import dataclass

from .dataset import DatasetConfig, DatasetHyperoptConfig
from .distributions import ParameterSet
from .pipeline import PipelineConfig


@dataclass
class Config:
    dataset: DatasetConfig
    pipeline: PipelineConfig
    parameters: Dict[str, Any]
    meta: Dict[str, Any]


@dataclass
class HyperOptConfig:
    dataset: DatasetHyperoptConfig
    pipeline: PipelineConfig
    parameters: ParameterSet
    n_trials: int
    meta: Dict[str, Any]

    optimize_features: bool = field(default=False)


ConfigSchema = Config.Schema()
HyperOptConfigSchema = HyperOptConfig.Schema()


def load_config(path: str) -> Config:
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    config = ConfigSchema.load(config)
    return config


def load_hyperopt_config(path: str) -> HyperOptConfig:
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    config = HyperOptConfigSchema.load(config)
    return config
