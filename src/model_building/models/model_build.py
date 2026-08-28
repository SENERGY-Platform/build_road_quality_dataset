from dataclasses import asdict, replace
import time

import pandas as pd
from xgboost import XGBRegressor

from src.model_building.config.experiment_config import ExperimentConfig
from src.model_building.models.models_ann import TwoPhaseANNModel
from src.model_building.data.model_data import ModelData
from src.model_building.data.data_split import split_model_data_for_validation
from src.model_building.features.features import label_discrete_from_continuous
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score, mean_absolute_error
from src.model_building.models.metrics import ModelPerformance

Model = TwoPhaseANNModel | Pipeline | XGBRegressor


def setup_model(model_type: str, experiment_config: ExperimentConfig) -> Model:
    """Instantiate the requested model type from the experiment configuration."""
    if model_type == 'ANN':
        ann_config = asdict(experiment_config.ann_model_config)
        # val_set_percentage is used for splitting, not for constructing the network.
        ann_config.pop('val_set_percentage')
        return TwoPhaseANNModel(**ann_config)

    if model_type == 'Linear':
        return make_pipeline(
            StandardScaler(),
            Ridge(**asdict(experiment_config.linear_model_config)),
        )

    if model_type == 'XGBoost':
        return XGBRegressor(**asdict(experiment_config.xgb_model_config))

    else:
        raise ValueError(f"Unknown model: {model_type}")


def update_model_data_for_validation(
    model: Model,
    model_data: ModelData,
    experiment_config: ExperimentConfig,
) -> ModelData:
    """Return model-specific data, including validation data if the model needs it."""
    if isinstance(model, TwoPhaseANNModel):
        return split_model_data_for_validation(model_data, experiment_config.ann_model_config)

    if isinstance(model, (Pipeline, XGBRegressor)):
        return replace(model_data)

    raise ValueError(f"Unknown model: {type(model)}")


def train_model(model: Model, model_data: ModelData) -> Model:
    """Fit a configured model on the prepared training data."""
    if isinstance(model, TwoPhaseANNModel):
        return model.fit(model_data)

    elif isinstance(model, (Pipeline, XGBRegressor)):
        model.fit(model_data.get_train_x(), model_data.get_train_y())
        return model

    else: raise ValueError(f"Unknown model: {type(model)}")


def evaluate_model(
    model: Model,
    model_data: ModelData,
) -> tuple[Model, ModelPerformance]:
    """Train a model, score it on the held-out manual test set, and return both.

    Training time is measured around `train_model`. Inference time is measured
    around `model.predict(model_data.test_x)` and reported as milliseconds per
    test sample, together with the number of samples timed.

    MAE is calculated on the numeric label scale. F1 metrics are calculated
    after converting both true labels and predictions to the discrete road
    quality classes: 0=good, 1=medium, and 2=bad.
    """
    train_start = time.perf_counter()
    trained_model = train_model(model, model_data)
    train_time_s = time.perf_counter() - train_start

    inference_start = time.perf_counter()
    raw_predictions = model.predict(model_data.test_x)
    inference_time_s = time.perf_counter() - inference_start

    inference_n_samples = len(model_data.test_x.index)
    inference_time_ms_per_sample = (inference_time_s * 1000) / inference_n_samples

    predictions_cont = pd.Series(raw_predictions,index=model_data.test_y.index,)
    score_mae = mean_absolute_error(model_data.test_y, predictions_cont)

    test_y_discrete = label_discrete_from_continuous(model_data.test_y)
    predictions_discrete = label_discrete_from_continuous(predictions_cont)
    f1_good, f1_medium, f1_bad = f1_score(
        test_y_discrete,
        predictions_discrete,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    f1_macro = f1_score(
        test_y_discrete,
        predictions_discrete,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
    performance = ModelPerformance(
        mae=score_mae,
        f1_macro=f1_macro,
        f1_good=f1_good,
        f1_medium=f1_medium,
        f1_bad=f1_bad,
        train_time_s=train_time_s,
        inference_time_ms_per_sample=inference_time_ms_per_sample,
        inference_n_samples=inference_n_samples,
    )

    return trained_model, performance
