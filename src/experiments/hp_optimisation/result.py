from dataclasses import dataclass

from src.model_building.models.metrics import ModelPerformance, ModelPerformanceStd

ParameterValue = int | float | str
ParameterSet = dict[str, ParameterValue]


@dataclass(frozen=True)
class OptimisationResult:
    """Final cross-validation result for one model parameter set and one dataset test case."""

    model_name: str
    parameter_set_id: int
    parameters: ParameterSet
    testcase_id: str
    performance: ModelPerformance
    performance_std: ModelPerformanceStd
