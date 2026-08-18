from dataclasses import dataclass


@dataclass(frozen=True)
class ANNModelConfig:
    """Hyperparameters for the custom two-phase ANN model."""

    val_set_percentage: float
    layer_num_first_round: int
    layer_num_second_round: int


@dataclass(frozen=True)
class LinearModelConfig:
    """Hyperparameters for the linear baseline model."""

    pass


@dataclass(frozen=True)
class XGBoostModelConfig:
    """Hyperparameters for the XGBoost classifier."""

    pass
