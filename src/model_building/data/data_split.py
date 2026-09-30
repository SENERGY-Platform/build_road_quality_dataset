from __future__ import annotations
from dataclasses import replace
from typing import Hashable
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

from src.model_building.config.experiment_config import ExperimentConfig
from src.model_building.config.model_config import ANNModelConfig
from src.model_building.data.data_test_cases import DataTestCase
from src.model_building.data.model_data import ModelData
from src.model_building.features.features import label_category_from_continuous

# Columns that identify the sensor reading behind a row. Manual labels-first rows
# repeat a reading once per matched label, and manual and OSM datasets are built
# from the same raw readings, so splits must keep each reading on one side.
READING_KEY_COLS = ["timestamp", "longitude", "latitude"]


def _split_a_case(test_case: DataTestCase, exp_config: ExperimentConfig, random_state) -> ModelData:
    """Create train/test data for case A, using only manual labels for training."""
    # all manual
    if test_case.case_id.startswith('case_a') and (test_case.manual_ds is None or test_case.manual_ds_id is None):
        raise ValueError('Manual dataset is required for splitting model data for combination case A.')

    x_train, y_train, x_test, y_test, groups_train, _ = _split_manual_dataset(test_case, exp_config, random_state)

    return ModelData(
        test_case_id=test_case.case_id,
        random_state=random_state,
        manual_train_x=x_train,
        manual_train_y=y_train,
        osm_train_x=pd.DataFrame(),  # empty
        osm_train_y=pd.Series(),  # empty
        test_x=x_test,
        test_y=y_test,
        manual_train_groups=groups_train,
    )


def _split_b_case(test_case: DataTestCase, exp_config: ExperimentConfig, random_state) -> ModelData:
    """Create train/test data for case B, combining manual and sampled OSM training data."""
    # manual and osm
    if test_case.case_id.startswith('case_b') and (test_case.osm_ds is None or test_case.osm_ds_id is None or
                                                   test_case.manual_ds_id is None or test_case.manual_ds is None):
        raise ValueError('Both manual and osm datasets are required for splitting model data for combination case B.')

    manual_x_train, manual_y_train, x_test, y_test, manual_groups_train, groups_test = _split_manual_dataset(
        test_case, exp_config, random_state)
    osm_x_train, osm_y_train, osm_groups_train = _split_osm_by_manual_label_distribution(
        test_case, exp_config, manual_y_train, groups_test, random_state)

    return ModelData(
        test_case_id=test_case.case_id,
        random_state=random_state,
        manual_train_x=manual_x_train,
        manual_train_y=manual_y_train,
        osm_train_x=osm_x_train,
        osm_train_y=osm_y_train,
        test_x=x_test,
        test_y=y_test,
        manual_train_groups=manual_groups_train,
        osm_train_groups=osm_groups_train,
    )


def _split_c_case(test_case: DataTestCase, exp_config: ExperimentConfig, random_state) -> ModelData:
    """Create train/test data for case C, training on sampled OSM data and testing on manual data."""
    # osm only
    if (test_case.case_id.startswith('case_c') and (test_case.osm_ds is None or test_case.osm_ds_id is None or
                                                    test_case.manual_ds_id is None or test_case.manual_ds is None)):
        raise ValueError('Both manual and osm datasets are required for splitting model data for combination case C.')

    _, manual_y_train, x_test, y_test, _, groups_test = _split_manual_dataset(test_case, exp_config, random_state)
    osm_x_train, osm_y_train, osm_groups_train = _split_osm_by_manual_label_distribution(
        test_case, exp_config, manual_y_train, groups_test, random_state)

    return ModelData(
        test_case_id=test_case.case_id,
        random_state=random_state,
        manual_train_x=pd.DataFrame(),
        manual_train_y=pd.Series(),
        osm_train_x=osm_x_train,
        osm_train_y=osm_y_train,
        test_x=x_test,
        test_y=y_test,
        osm_train_groups=osm_groups_train,
    )


def get_reading_ids(data_set: pd.DataFrame) -> pd.Series:
    """Return an id per row identifying the sensor reading it was built from.

    Rows built from the same reading share timestamp and coordinates, also across
    manual and OSM datasets, so equal ids mean identical sensor features.
    """
    return pd.util.hash_pandas_object(data_set[READING_KEY_COLS], index=False)


def _draw_group_ids(y: pd.Series, groups: pd.Series, test_size: float, random_state: int) -> pd.Index:
    """Draw a `test_size` share of groups, stratified by each group's most common label category."""
    group_classes = label_category_from_continuous(y).groupby(groups.to_numpy()).agg(lambda s: s.mode().iloc[0])
    _, drawn_group_ids = train_test_split(group_classes.index, stratify=group_classes,
                                          random_state=random_state, test_size=test_size)
    return pd.Index(drawn_group_ids)


def _split_by_group_ids(
        x: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        drawn_group_ids: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Separate the rows of drawn groups from the remaining rows, keeping each group on one side.

    Returns `(x_rest, x_drawn, y_rest, y_drawn, groups_rest, groups_drawn)`; empty
    inputs give empty outputs.
    """
    is_drawn = groups.isin(drawn_group_ids)
    return x[~is_drawn], x[is_drawn], y[~is_drawn], y[is_drawn], groups[~is_drawn], groups[is_drawn]


def _split_manual_dataset(test_case: DataTestCase, exp_config: ExperimentConfig, random_state):
    """Split a manual dataset into stratified train and test sets grouped by sensor reading.

    `test_set_percentage` is the share of readings drawn into the test set.
    """
    features_df, label_s = _split_features_label(test_case.manual_ds, exp_config.label_column, exp_config.features)
    groups = get_reading_ids(test_case.manual_ds)
    test_group_ids = _draw_group_ids(label_s, groups, exp_config.test_set_percentage, random_state)
    x_train, x_test, y_train, y_test, groups_train, groups_test = _split_by_group_ids(
        features_df, label_s, groups, test_group_ids)
    return x_train, y_train, x_test, y_test, groups_train, groups_test


def _split_features_label(data_set: pd.DataFrame, label_column: str, features: list[str]) -> tuple[
    pd.DataFrame, pd.Series]:
    """Return the selected feature columns and label column from a dataset."""
    return data_set[features], data_set[label_column]


def _get_series_distribution(label_s: pd.Series[str]) -> dict[Hashable, float]:
    """Return normalized label frequencies for a label series."""
    return label_s.value_counts(normalize=True).to_dict()


def _calc_highest_possible_n_by_distribution(df: pd.DataFrame, label_col: str, requested_n: int,
                                             target_distribution: dict[Hashable, float]) -> float:
    """Calculate the largest sample size that can satisfy the requested label distribution."""
    available_per_label = df[label_col].value_counts()
    max_feasible_n = min(
        int(available_per_label.get(label, 0) / frac) for label, frac in target_distribution.items() if frac > 0
    )
    return min(requested_n, max_feasible_n)


def _draw_stratified_osm_sample(
        df: pd.DataFrame,
        label_col: str,
        requested_sample_size: int,
        requested_distribution: dict,
        random_state: int | None = None,
) -> pd.DataFrame:
    """Draw a shuffled stratified sample that approximates a requested label distribution."""
    actual_n = _calc_highest_possible_n_by_distribution(df, label_col, requested_sample_size, requested_distribution)

    parts = []
    for label, frac in requested_distribution.items():
        n_label = round(actual_n * frac)
        part = df[df[label_col] == label].sample(
            n=n_label,
            random_state=random_state,
        )
        parts.append(part)

    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)


def _split_osm_by_manual_label_distribution(test_case: DataTestCase, exp_config: ExperimentConfig,
                                            manual_train_y: pd.Series, manual_test_groups: pd.Series,
                                            random_state: int = 42) \
        -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Sample OSM training data with the same categorical label distribution as manual training labels.

    OSM rows built from a sensor reading that is in the manual test set are excluded
    first, so the model never trains on the exact features it is tested on.
    """
    osm_df = test_case.osm_ds.copy()
    osm_df['reading_id'] = get_reading_ids(osm_df)
    osm_df = osm_df.loc[~osm_df['reading_id'].isin(manual_test_groups)].copy()
    osm_df['label_str'] = label_category_from_continuous(osm_df['label'])
    train_sample_num = _calc_osm_train_size(test_case, exp_config, len(manual_train_y.index))
    target_distribution = _get_series_distribution(label_category_from_continuous(manual_train_y))
    osm_train = _draw_stratified_osm_sample(
        df=osm_df,
        label_col='label_str',
        requested_sample_size=train_sample_num,
        requested_distribution=target_distribution,
        random_state=random_state,
    )
    features_df, label_s = _split_features_label(osm_train, exp_config.label_column, exp_config.features)
    return features_df, label_s, osm_train['reading_id']


def _calc_osm_train_size(test_case: DataTestCase, exp_config: ExperimentConfig, manual_train_len: int) -> int:
    """Return the number of OSM rows to use for the current case and experiment settings."""
    if exp_config.all_osm_data and (test_case.case_id.startswith('case_b') or test_case.case_id.startswith('case_c')):
        return len(test_case.osm_ds.index)
    else:
        return manual_train_len


def split_data_for_test_case(test_case: DataTestCase, experiment_config: ExperimentConfig,
                             random_state: int = 42) -> ModelData:
    """Dispatch a data test case to the corresponding train/test split strategy."""
    if test_case.case_id.startswith('case_a'):
        return _split_a_case(test_case, experiment_config, random_state)
    elif test_case.case_id.startswith('case_b'):
        return _split_b_case(test_case, experiment_config, random_state)
    elif test_case.case_id.startswith('case_c'):
        return _split_c_case(test_case, experiment_config, random_state)
    else:
        raise ValueError('Unknown test case type')


def _groups_or_row_ids(train_y: pd.Series, train_groups: pd.Series, source_name: str) -> pd.Series:
    """Return reading ids, or stable source-prefixed row ids when no groups exist."""
    if train_y.empty or not train_groups.empty:
        return train_groups
    return pd.Series(
        [f"{source_name}_{i}" for i in range(len(train_y.index))],
        index=train_y.index,
    )


def _get_validation_group_ids(
        manual_train_y: pd.Series,
        manual_train_groups: pd.Series,
        osm_train_y: pd.Series,
        osm_train_groups: pd.Series,
        model_config: ANNModelConfig,
        random_state: int,
) -> pd.Index:
    """Choose validation reading ids for all training sources, manual readings first.

    Validation readings are drawn from the manual training readings, stratified by
    manual labels, so manual validation mirrors the manual test split. OSM readings
    that are not in the manual training data are drawn separately, stratified by
    OSM labels, so OSM pretraining keeps a validation set of the same share. The
    returned ids apply to both sources, which keeps a shared reading on one side.
    """
    validation_group_id_parts = []
    if not manual_train_y.empty:
        validation_group_id_parts.append(_draw_group_ids(
            manual_train_y, manual_train_groups, model_config.val_set_percentage, random_state))
    if not osm_train_y.empty:
        is_osm_only = ~osm_train_groups.isin(manual_train_groups)
        if is_osm_only.any():
            validation_group_id_parts.append(_draw_group_ids(
                osm_train_y[is_osm_only], osm_train_groups[is_osm_only],
                model_config.val_set_percentage, random_state))
    if not validation_group_id_parts:
        return pd.Index([])
    return validation_group_id_parts[0].append(validation_group_id_parts[1:])


def split_model_data_for_validation(model_data: ModelData, model_config: ANNModelConfig, random_state=42) -> ModelData:
    """Returns a new model data object containing updated training and new validation data created by the split of the old training data.
    Assumes pre-regulated sample distributions and propagates these through the labels in the val split."""
    manual_groups = _groups_or_row_ids(model_data.manual_train_y, model_data.manual_train_groups, "manual")
    osm_groups = _groups_or_row_ids(model_data.osm_train_y, model_data.osm_train_groups, "osm")
    try:
        validation_group_ids = _get_validation_group_ids(
            model_data.manual_train_y, manual_groups, model_data.osm_train_y, osm_groups, model_config, random_state)
    except Exception as e:
        warnings.warn(f"Train validation split didn't work, returning empty validation set for now. {e.args[0]}")
        validation_group_ids = pd.Index([])

    manual_train_x, manual_val_x, manual_train_y, manual_val_y, manual_train_groups, _ = _split_by_group_ids(
        model_data.manual_train_x, model_data.manual_train_y, manual_groups, validation_group_ids)
    osm_train_x, osm_val_x, osm_train_y, osm_val_y, osm_train_groups, _ = _split_by_group_ids(
        model_data.osm_train_x, model_data.osm_train_y, osm_groups, validation_group_ids)
    return replace(
        model_data,
        manual_train_x=manual_train_x,
        manual_train_y=manual_train_y,
        osm_train_x=osm_train_x,
        osm_train_y=osm_train_y,
        manual_train_groups=manual_train_groups,
        osm_train_groups=osm_train_groups,
        manual_val_x=manual_val_x,
        manual_val_y=manual_val_y,
        osm_val_x=osm_val_x,
        osm_val_y=osm_val_y,
    )


def build_stratified_shuffle_split_datasets(test_case: DataTestCase, experiment_config: ExperimentConfig) -> list[
    ModelData]:
    """Build repeated stratified train/test splits, grouped by sensor reading, for one test case."""
    model_data_cases: list[ModelData] = []
    for k in range(experiment_config.cross_validation_k):
        model_data_cases.append(split_data_for_test_case(test_case, experiment_config, random_state=k))
    return model_data_cases
