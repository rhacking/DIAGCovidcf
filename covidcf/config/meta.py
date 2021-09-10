from dataclasses import field
from typing import Dict, Any, Optional, List

import yaml
from marshmallow_dataclass import dataclass

from covidcf.config.dataset import Metric


@dataclass
class MetaConfig:
    @dataclass
    class OverviewConfig:
        x: str
        row: Optional[str]
        metrics: Dict[str, Metric]

    overview: OverviewConfig
    plot_shap: bool
    adjust_thresholds: bool


MetaSchema = MetaConfig.Schema()


def load_meta(path: str) -> MetaConfig:
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    config = MetaSchema.load(config)
    return config
