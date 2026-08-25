from src.model_building.models.metrics import ModelPerformance
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class RidgeOptimisationResult:
    """Final cross-validation result for one alpha and one dataset test case."""

    alpha: float
    testcase_id: str
    performance: ModelPerformance