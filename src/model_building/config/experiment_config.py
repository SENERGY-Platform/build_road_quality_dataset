from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from src.model_building.config.model_config import ANNModelConfig, LinearModelConfig, XGBoostModelConfig

ModelConfig = ANNModelConfig | LinearModelConfig | XGBoostModelConfig


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one model-building experiment run."""
    experiment_name: str
    case_type: str
    all_osm_data: bool | None
    cross_validation_k: int

    ds_version: str
    feature_set_name: str
    features: list[str]
    model: str  # 'ANN', Linear, XGBoost ..
    test_set_percentage: float  # all models get the same test set - if a model uses validation it is later split of the training set

    ann_model_config: Optional[ANNModelConfig] = None
    linear_model_config: Optional[LinearModelConfig] = None
    xgb_model_config: Optional[XGBoostModelConfig] = None

    label_column: str = 'label'

    def __post_init__(self):
        """Validate that each requested model has a matching config object."""
        if self.case_type not in {'A', 'B', 'C'}:
            raise ValueError("test_case must be one of 'A', 'B', or 'C'.")
        if self.case_type in {'B', 'C'} and self.all_osm_data is None:
            raise ValueError("all_osm_data must be True or False for test cases B and C.")
        if self.cross_validation_k <= 0:
            raise ValueError("cross_validation_k must be greater than 0.")
        if self.model not in {'ANN', 'Linear', 'XGBoost'}:
            raise ValueError("model must be one of 'ANN', 'Linear', or 'XGBoost'.")

    def get_case_id(self, manual_ds_id: str, osm_ds_id: str) -> str:
        """Return the full dataset-case identifier for one manual/OSM pairing."""
        return f'{self.get_ds_case_group()}__{manual_ds_id}__{osm_ds_id}'

    def get_ds_case_group(self) -> str:
        """Return the dataset-case group name used for grouping experiment runs."""
        if self.case_type == 'A':
            return 'case_a__osm_no_points'
        elif self.case_type == 'B':
            return 'case_b__osm_all_points' if self.all_osm_data else 'case_b__osm_limited_points'
        elif self.case_type == 'C':
            return 'case_c__osm_all_points' if self.all_osm_data else 'case_c__osm_limited_points'
        raise ValueError('Case type not recognized.')

    def set_model_config(self, config: ANNModelConfig | XGBoostModelConfig | LinearModelConfig) -> ExperimentConfig:
        """Return a copy with the selected model's hyperparameter config set."""
        if isinstance(config, ANNModelConfig) and self.model == 'ANN':
            return replace(self, ann_model_config=config)

        elif isinstance(config, XGBoostModelConfig) and self.model == 'XGBoost':
            return replace(self, xgb_model_config=config)

        elif isinstance(config, LinearModelConfig) and self.model == 'Linear':
            return replace(self, linear_model_config=config)

        raise ValueError(f"Config type {type(config).__name__} does not match selected model {self.model}.")

    def get_model_params(self) -> ModelConfig:
        """Return the hyperparameter config for the selected model."""
        if self.model == "ANN" and self.ann_model_config is not None:
            return self.ann_model_config
        if self.model == "Linear" and self.linear_model_config is not None:
            return self.linear_model_config
        if self.model == "XGBoost" and self.xgb_model_config is not None:
            return self.xgb_model_config

        raise ValueError(f"Missing model config for selected model {self.model}.")
