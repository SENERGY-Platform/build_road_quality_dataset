
from dataclasses import dataclass, field
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.model_building.data.model_data import ModelData
from src.model_building.models.models_ann import TwoPhaseANNModel

Model = Ridge | Pipeline | TwoPhaseANNModel | XGBRegressor


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
class ModelPerformanceStd:
    """Standard deviation of model quality and runtime metrics across CV splits."""

    mae: float
    f1_macro: float
    f1_good: float
    f1_medium: float
    f1_bad: float
    train_time_s: float
    inference_time_ms_per_sample: float
    inference_n_samples: float


@dataclass
class CrossValidationPerformance:
    """Collect and aggregate model performance across cross-validation splits."""

    median_model: Model | None = field(default=None)
    median_data_set: ModelData | None = field(default=None)
    median_performance: ModelPerformance | None = field(default=None)

    _mae_scores: list[float] = field(default_factory=list)
    _f1_macro_scores: list[float] = field(default_factory=list)
    _f1_good_scores: list[float] = field(default_factory=list)
    _f1_medium_scores: list[float] = field(default_factory=list)
    _f1_bad_scores: list[float] = field(default_factory=list)
    _train_time_s_scores: list[float] = field(default_factory=list)
    _inference_time_ms_per_sample_scores: list[float] = field(default_factory=list)
    _inference_n_samples: list[int] = field(default_factory=list)

    _models: list[Model] = field(default_factory=list)
    _used_data_sets: list[ModelData] = field(default_factory=list)

    def get_final_performance(self) -> tuple[ModelPerformance, ModelPerformanceStd]:
        """Return mean and standard deviation metrics across all collected splits."""
        inference_n_samples = sum(self._inference_n_samples)
        inference_time_ms_per_sample = float(
            np.average(
                self._inference_time_ms_per_sample_scores,
                weights=self._inference_n_samples,
            )
        )

        performance = ModelPerformance(
            mae=float(np.mean(self._mae_scores)),
            f1_macro=float(np.mean(self._f1_macro_scores)),
            f1_good=float(np.mean(self._f1_good_scores)),
            f1_medium=float(np.mean(self._f1_medium_scores)),
            f1_bad=float(np.mean(self._f1_bad_scores)),
            train_time_s=float(np.mean(self._train_time_s_scores)),
            inference_time_ms_per_sample=inference_time_ms_per_sample,
            inference_n_samples=inference_n_samples,
        )

        performance_std = ModelPerformanceStd(
            mae=float(np.std(self._mae_scores)),
            f1_macro=float(np.std(self._f1_macro_scores)),
            f1_good=float(np.std(self._f1_good_scores)),
            f1_medium=float(np.std(self._f1_medium_scores)),
            f1_bad=float(np.std(self._f1_bad_scores)),
            train_time_s=float(np.std(self._train_time_s_scores)),
            inference_time_ms_per_sample=float(np.std(self._inference_time_ms_per_sample_scores)),
            inference_n_samples=float(np.std(self._inference_n_samples)),
        )

        return performance, performance_std

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

    def add_model(self, model):
        """Store the fitted model for one cross-validation split."""
        self._models.append(model)

    def add_model_data(self, model_data):
        """Store the model data used for one cross-validation split."""
        self._used_data_sets.append(model_data)

    def calculate_median_model(self) -> "CrossValidationPerformance":
        """Cache the median fold's model, data, and metrics.

        Runs are ordered from worst to best by macro F1 first and MAE second.
        Macro F1 is better when higher; MAE is better when lower. For an even
        number of runs, the upper median is selected.
        """
        performance_count = len(self._mae_scores)
        if performance_count == 0:
            raise ValueError("Cannot calculate a median model without performance scores.")

        performance_lengths = {
            "mae": len(self._mae_scores),
            "f1_macro": len(self._f1_macro_scores),
            "f1_good": len(self._f1_good_scores),
            "f1_medium": len(self._f1_medium_scores),
            "f1_bad": len(self._f1_bad_scores),
            "train_time_s": len(self._train_time_s_scores),
            "inference_time_ms_per_sample": len(self._inference_time_ms_per_sample_scores),
            "inference_n_samples": len(self._inference_n_samples),
            "models": len(self._models),
            "used_data_sets": len(self._used_data_sets),
        }
        if len(set(performance_lengths.values())) != 1:
            raise ValueError(f"Cannot align cross-validation runs: {performance_lengths}")

        ranked_indices = sorted(
            range(performance_count),
            key=lambda index: (
                self._f1_macro_scores[index],
                -self._mae_scores[index],
            ),
        )
        median_index = ranked_indices[len(ranked_indices) // 2]
        median_performance = ModelPerformance(
            mae=self._mae_scores[median_index],
            f1_macro=self._f1_macro_scores[median_index],
            f1_good=self._f1_good_scores[median_index],
            f1_medium=self._f1_medium_scores[median_index],
            f1_bad=self._f1_bad_scores[median_index],
            train_time_s=self._train_time_s_scores[median_index],
            inference_time_ms_per_sample=self._inference_time_ms_per_sample_scores[median_index],
            inference_n_samples=self._inference_n_samples[median_index],
        )

        self.median_model =  self._models[median_index]
        self.median_data_set = self._used_data_sets[median_index]
        self.median_performance = median_performance

        self._models.clear()
        self._used_data_sets.clear()

        return self
