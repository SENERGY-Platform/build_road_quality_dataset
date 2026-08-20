from dataclasses import dataclass


@dataclass(frozen=True)
class ANNModelConfig:
    """Hyperparameters for the custom two-phase ANN model."""

    val_set_percentage: float
    layer_num_first_round: int
    layer_num_second_round: int
    pretrain_max_epochs: int = 100
    finetune_max_epochs: int = 200
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.0001


@dataclass(frozen=True)
class LinearModelConfig:
    """Hyperparameters for the linear baseline model."""

    pass


@dataclass(frozen=True)
class XGBoostModelConfig:
    """Hyperparameters for the XGBoost classifier."""

    pass
