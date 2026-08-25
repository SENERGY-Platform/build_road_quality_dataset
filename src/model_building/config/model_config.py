from dataclasses import dataclass


@dataclass(frozen=True)
class ANNModelConfig:
    """Hyperparameters for the custom two-phase ANN model."""

    val_set_percentage: float
    layer_num_first_round: int
    layer_num_second_round: int
    pretrain_learning_rate: float = 0.001
    finetune_learning_rate: float = 0.001
    batch_size: int = 64
    dropout: float = 0.0
    weight_decay: float = 0.0
    pretrain_max_epochs: int = 100
    finetune_max_epochs: int = 200
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.0001


@dataclass(frozen=True)
class LinearModelConfig:
    """Hyperparameters for the linear baseline model."""
    alpha: float


@dataclass(frozen=True)
class XGBoostModelConfig:
    """Hyperparameters for the XGBoost regressor."""

    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 6
    min_child_weight: float = 1.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_lambda: float = 1.0

    objective: str = "reg:squarederror"
    tree_method = "hist"
    random_state: int = 42
