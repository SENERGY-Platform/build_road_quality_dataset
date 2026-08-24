from dataclasses import dataclass
from typing import Optional

from src.model_building.config.model_config import ANNModelConfig, LinearModelConfig, XGBoostModelConfig


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one model-building experiment run."""
    experiment_name: str
    test_cases: list[str]
    case_b_all_osm_data: bool
    case_c_all_osm_data: bool
    cross_validation_k: int

    ds_version: str
    features: list[str]
    models: list[str] # 'ANN', Linear, XGBoost ..
    test_set_percentage: float # all models get the same test set - if a model uses validation it is later split of the training set

    ann_model_config: Optional[ANNModelConfig] = None
    linear_model_config: Optional[LinearModelConfig] = None
    xgb_model_config: Optional[XGBoostModelConfig] = None

    label_column: str = 'label'

    def __post_init__(self):
        """Validate that each requested model has a matching config object."""
        if self.cross_validation_k <= 0:
            raise ValueError("cross_validation_k must be greater than 0.")
        if 'ANN' in self.models and not self.ann_model_config:
            raise ValueError("Please provide a configuration for the ANN model.")
        if 'Linear' in self.models and not self.linear_model_config:
            raise ValueError("Please provide a configuration for the linear model.")
        if 'XGBoost' in self.models and not self.xgb_model_config:
            raise ValueError("Please provide a configuration for the XGBoost model.")
