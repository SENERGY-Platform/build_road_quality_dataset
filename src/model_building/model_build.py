from dataclasses import asdict

from xgboost import XGBClassifier

from experiment_config import ExperimentConfig
from model_data import ModelData
from models_ann import TwoPhaseANNModel
from sklearn.linear_model import Ridge
import xgboost as xgb


def setup_model(model_type: str, experiment_config: ExperimentConfig) -> TwoPhaseANNModel | Ridge | XGBClassifier:
    """Instantiate the requested model type from the experiment configuration."""
    if model_type == 'ANN':
        return TwoPhaseANNModel(**asdict(experiment_config.ann_model_config))

    if model_type == 'Linear':
        return Ridge(**asdict(experiment_config.linear_model_config))

    if model_type == 'XGBoost':
        return xgb.XGBClassifier(**asdict(experiment_config.xgb_model_config))

    else:
        raise ValueError(f"Unknown model: {model_type}")

