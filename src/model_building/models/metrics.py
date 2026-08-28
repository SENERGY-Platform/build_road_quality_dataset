
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class ModelPerformance:
    """Model quality and runtime metrics for one train/evaluation run."""

    mae: float
    f1_macro: float
    f1_good: float
    f1_medium: float
    f1_bad: float
    train_time_s: float
    inference_time_ms_per_sample: float
    inference_n_samples: int

@dataclass(frozen=True)
class CrossValidationPerformance:
    """Collect and aggregate model performance across cross-validation splits."""

    _mae_scores: list[float] = field(default_factory=list)
    _f1_macro_scores: list[float] = field(default_factory=list)
    _f1_good_scores: list[float] = field(default_factory=list)
    _f1_medium_scores: list[float] = field(default_factory=list)
    _f1_bad_scores: list[float] = field(default_factory=list)
    _train_time_s_scores: list[float] = field(default_factory=list)
    _inference_time_ms_per_sample_scores: list[float] = field(default_factory=list)
    _inference_n_samples: list[int] = field(default_factory=list)

    def get_final_performance(self):
        """Return averaged score and runtime metrics across all collected splits."""
        inference_n_samples = sum(self._inference_n_samples)
        inference_time_ms_per_sample = float(
            np.average(
                self._inference_time_ms_per_sample_scores,
                weights=self._inference_n_samples,
            )
        )

        return ModelPerformance(
            mae=float(np.mean(self._mae_scores)),
            f1_macro=float(np.mean(self._f1_macro_scores)),
            f1_good=float(np.mean(self._f1_good_scores)),
            f1_medium=float(np.mean(self._f1_medium_scores)),
            f1_bad=float(np.mean(self._f1_bad_scores)),
            train_time_s=float(np.mean(self._train_time_s_scores)),
            inference_time_ms_per_sample=inference_time_ms_per_sample,
            inference_n_samples=inference_n_samples,
        )

    def add_performance(self, performance: ModelPerformance):
        """Add metrics from one cross-validation split."""
        self._mae_scores.append(performance.mae)
        self._f1_macro_scores.append(performance.f1_macro)
        self._f1_good_scores.append(performance.f1_good)
        self._f1_medium_scores.append(performance.f1_medium)
        self._f1_bad_scores.append(performance.f1_bad)
        self._train_time_s_scores.append(performance.train_time_s)
        self._inference_time_ms_per_sample_scores.append(performance.inference_time_ms_per_sample)
        self._inference_n_samples.append(performance.inference_n_samples)
