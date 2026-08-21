from dataclasses import asdict, replace

from xgboost import XGBRegressor

from src.model_building.config.experiment_config import ExperimentConfig
from src.model_building.models.models_ann import TwoPhaseANNModel
from src.model_building.data.model_data import ModelData
from src.model_building.data.data_split import split_model_data_for_validation
from src.model_building.features.features import label_discrete_from_continuous
from sklearn.linear_model import Ridge



def setup_model(model_type: str, experiment_config: ExperimentConfig) -> TwoPhaseANNModel | Ridge | XGBRegressor:
    """Instantiate the requested model type from the experiment configuration."""
    if model_type == 'ANN':
        ann_config = asdict(experiment_config.ann_model_config)
        # val_set_percentage is used for splitting, not for constructing the network.
        ann_config.pop('val_set_percentage')
        return TwoPhaseANNModel(**ann_config)

    if model_type == 'Linear':
        return Ridge(**asdict(experiment_config.linear_model_config))

    if model_type == 'XGBoost':
        return XGBRegressor(**asdict(experiment_config.xgb_model_config))

    else:
        raise ValueError(f"Unknown model: {model_type}")


def update_model_data_for_validation(
    model: TwoPhaseANNModel | Ridge | XGBRegressor,
    model_data: ModelData,
    experiment_config: ExperimentConfig,
) -> ModelData:
    """Return model-specific data, including validation data if the model needs it."""
    if type(model) == TwoPhaseANNModel:
        return split_model_data_for_validation(model_data, experiment_config.ann_model_config)

    if type(model) in [Ridge, XGBRegressor]:
        return replace(model_data)

    raise ValueError(f"Unknown model: {type(model)}")


def train_model(model: TwoPhaseANNModel | Ridge | XGBRegressor,
                model_data: ModelData) -> TwoPhaseANNModel | Ridge | XGBRegressor:
    """Fit a configured model on the prepared training data."""
    if type(model) == TwoPhaseANNModel:
        return model.fit(model_data)

    elif type(model) in [Ridge, XGBRegressor]:
        return model.fit(model_data.get_train_x(), model_data.get_train_y())

    else: raise ValueError(f"Unknown model: {type(model)}")
