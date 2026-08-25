from dataclasses import dataclass

from src.model_building.models.metrics import ModelPerformance

@dataclass(frozen=True)
class RidgeOptimisationResult:
    """Final cross-validation result for one alpha and one dataset test case."""

    alpha: float
    testcase_id: str
    performance: ModelPerformance


@dataclass(frozen=True)
class XGBoostOptimisationResult:
    """Final cross-validation result for one XGBoost parameter set and one dataset test case."""

    parameter_set_id: int
    parameters: dict[str, int | float | str]
    testcase_id: str
    performance: ModelPerformance
