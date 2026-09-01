from dataclasses import dataclass
from typing import Optional

from src.model_building.config.model_config import ANNModelConfig, LinearModelConfig, XGBoostModelConfig


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one model-building experiment run."""
    experiment_name: str
    test_case: str
    all_osm_data: bool | None
    cross_validation_k: int

    ds_version: str
    feature_set_name: str
    features: list[str]
    models: list[str]  # 'ANN', Linear, XGBoost ..
    test_set_percentage: float  # all models get the same test set - if a model uses validation it is later split of the training set

    ann_model_config: Optional[ANNModelConfig] = None
    linear_model_config: Optional[LinearModelConfig] = None
    xgb_model_config: Optional[XGBoostModelConfig] = None

    label_column: str = 'label'

    def __post_init__(self):
        """Validate that each requested model has a matching config object."""
        if self.test_case not in {'A', 'B', 'C'}:
            raise ValueError("test_case must be one of 'A', 'B', or 'C'.")
        if self.test_case in {'B', 'C'} and self.all_osm_data is None:
            raise ValueError("all_osm_data must be True or False for test cases B and C.")
        if self.cross_validation_k <= 0:
            raise ValueError("cross_validation_k must be greater than 0.")
        if 'ANN' in self.models and not self.ann_model_config:
            raise ValueError("Please provide a configuration for the ANN model.")
        if 'Linear' in self.models and not self.linear_model_config:
            raise ValueError("Please provide a configuration for the linear model.")
        if 'XGBoost' in self.models and not self.xgb_model_config:
            raise ValueError("Please provide a configuration for the XGBoost model.")

    def get_case_id(self, manual_ds_id: str, osm_ds_id: str) -> str:
        return f'{self.get_case_group()}__{manual_ds_id}__{osm_ds_id}'

    def get_case_group(self) -> str:
        if self.test_case == 'A':
            return 'case_a__osm_no_points'
        elif self.test_case == 'B':
            return 'case_b__osm_all_points' if self.all_osm_data else 'case_b__osm_limited_points'
        elif self.test_case == 'C':
            return 'case_c__osm_all_points' if self.all_osm_data else 'case_c__osm_limited_points'
        raise ValueError('Case type not recognized.')
