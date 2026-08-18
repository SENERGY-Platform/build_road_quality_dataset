from __future__ import annotations
from typing import Hashable

import pandas as pd
from sklearn.model_selection import train_test_split

from data_test_cases import DataTestCase
from experiment_config import ExperimentConfig
from features import label_category_from_continuous
from model_data import ModelData


def _split_a_case(test_case: DataTestCase, exp_config: ExperimentConfig) -> ModelData:
    """Create train/test data for case A, using only manual labels for training."""
    # all manual
    if test_case.case_type == 'A' and (test_case.manual_ds is None or test_case.manual_ds_id is None):
        raise ValueError('Manual dataset is required for splitting model data for combination case A.')

    x_train, y_train, x_test, y_test = _split_manual_dataset(test_case, exp_config)

    return ModelData(
        manual_train_x=x_train,
        manual_train_y=y_train,
        osm_train_x=pd.DataFrame(), # empty
        osm_train_y=pd.Series(),    # empty
        test_x=x_test,
        test_y=y_test,
    )

def _split_b_case(test_case: DataTestCase, exp_config: ExperimentConfig)-> ModelData:
    """Create train/test data for case B, combining manual and sampled OSM training data."""
    # manual and osm
    if test_case.case_type == 'B' and (test_case.osm_ds is None or test_case.osm_ds_id is None or
                                       test_case.manual_ds_id is None or test_case.manual_ds is None):
        raise ValueError('Both manual and osm datasets are required for splitting model data for combination case B.')

    manual_x_train, manual_y_train, x_test, y_test = _split_manual_dataset(test_case, exp_config)
    osm_x_train, osm_y_train = _split_osm_by_manual_label_distribution(test_case, exp_config, manual_y_train)

    return ModelData(
        manual_train_x=manual_x_train,
        manual_train_y=manual_y_train,
        osm_train_x=osm_x_train,
        osm_train_y=osm_y_train,
        test_x=x_test,
        test_y=y_test,
    )

def _split_c_case(test_case: DataTestCase, exp_config: ExperimentConfig) -> ModelData:
    """Create train/test data for case C, training on sampled OSM data and testing on manual data."""
    # osm only
    if test_case.case_type == 'C' and (test_case.osm_ds is None or test_case.osm_ds_id is None or
                                       test_case.manual_ds_id is None or test_case.manual_ds is None):
        raise ValueError('Both manual and osm datasets are required for splitting model data for combination case C.')

    manual_x_train, manual_y_train, x_test, y_test = _split_manual_dataset(test_case, exp_config)

    osm_x_train, osm_y_train = _split_osm_by_manual_label_distribution(test_case, exp_config, manual_y_train)

    return ModelData(
        manual_train_x=pd.DataFrame(),
        manual_train_y=pd.Series(),
        osm_train_x=osm_x_train,
        osm_train_y=osm_y_train,
        test_x=x_test,
        test_y=y_test,
    )

def _split_manual_dataset(test_case: DataTestCase, exp_config: ExperimentConfig):
    """Split a manual dataset into stratified train and test feature/label sets."""
    features_df, label_s = _split_features_label(test_case.manual_ds, exp_config.label_column, exp_config.features)
    y_classes = label_category_from_continuous(label_s)
    x_train, x_test, y_train, y_test = train_test_split(features_df, label_s,
                                                        stratify=y_classes, random_state=42,
                                                        test_size=exp_config.test_set_percentage)
    return x_train, y_train, x_test, y_test


def _split_features_label(data_set: pd.DataFrame, label_column: str, features:list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Return the selected feature columns and label column from a dataset."""
    return data_set[features], data_set[label_column]

def _get_series_distribution(label_s: pd.Series[str])-> dict[Hashable, float]:
    """Return normalized label frequencies for a label series."""
    return label_s.value_counts(normalize=True).to_dict()

def _calc_highest_possible_n_by_distribution(df: pd.DataFrame, label_col:str, requested_n:int, target_distribution: dict[Hashable, float]) -> float:
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

def _split_osm_by_manual_label_distribution(test_case: DataTestCase, exp_config: ExperimentConfig, manual_train_y: pd.Series)-> tuple[pd.DataFrame, pd.Series]:
    """Sample OSM training data with the same categorical label distribution as manual training labels."""
    osm_df = test_case.osm_ds.copy()
    osm_df['label_str'] = label_category_from_continuous(osm_df['label'])
    train_sample_num = _calc_osm_train_size(test_case, exp_config, len(manual_train_y.index))
    target_distribution = _get_series_distribution(label_category_from_continuous(manual_train_y))
    osm_train = _draw_stratified_osm_sample(
        df=osm_df,
        label_col='label_str',
        requested_sample_size=train_sample_num,
        requested_distribution=target_distribution,
        random_state=42
    )
    features_df, label_s = _split_features_label(osm_train, exp_config.label_column, exp_config.features)
    return features_df, label_s

def _calc_osm_train_size(test_case: DataTestCase, exp_config: ExperimentConfig, manual_train_len:int) -> int:
    """Return the number of OSM rows to use for the current case and experiment settings."""
    if (test_case.case_type == 'B' and exp_config.case_b_all_osm_data) or (test_case.case_type == 'C' and exp_config.case_c_all_osm_data):
        return len(test_case.osm_ds.index)
    else:
        return manual_train_len

def split_data_for_test_case(test_case: DataTestCase, experiment_config: ExperimentConfig)-> ModelData:
    """Dispatch a data test case to the corresponding train/test split strategy."""
    if test_case.case_type == 'A':
        return _split_a_case(test_case, experiment_config)
    elif test_case.case_type == 'B':
        return _split_b_case(test_case, experiment_config)
    elif test_case.case_type == 'C':
        return _split_c_case(test_case, experiment_config)
    else:
        raise ValueError('Unknown test case type')
