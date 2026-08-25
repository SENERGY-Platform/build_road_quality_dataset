
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ModelPerformance:
    mae: float
    f1_macro: float
    f1_good: float
    f1_medium: float
    f1_bad: float

@dataclass(frozen=True)
class CrossValidationPerformance:
    _mae_scores: list[float] = field(default_factory=list)
    _f1_macro_scores: list[float] = field(default_factory=list)
    _f1_good_scores: list[float] = field(default_factory=list)
    _f1_medium_scores: list[float] = field(default_factory=list)
    _f1_bad_scores: list[float] = field(default_factory=list)

    def get_final_performance(self):
        return ModelPerformance(
            mae=float(np.mean(self._mae_scores)),
            f1_macro=float(np.mean(self._f1_macro_scores)),
            f1_good=float(np.mean(self._f1_good_scores)),
            f1_medium=float(np.mean(self._f1_medium_scores)),
            f1_bad=float(np.mean(self._f1_bad_scores))
        )

    def add_performance(self, performance: ModelPerformance):
        self._mae_scores.append(performance.mae)
        self._f1_macro_scores.append(performance.f1_macro)
        self._f1_good_scores.append(performance.f1_good)
        self._f1_medium_scores.append(performance.f1_medium)
        self._f1_bad_scores.append(performance.f1_bad)