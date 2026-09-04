from dataclasses import dataclass

import pandas as pd

from src.model_building.config.experiment_config import ExperimentConfig


@dataclass
class DataTestCaseManualParameters:
    """Parsed parameter values from a manual feature dataset identifier."""

    manual_radius: str | None
    manual_mapping_procedure: str | None
    manual_time_threshold: str | None
    manual_vehicle_type: str | None


@dataclass
class DataTestCaseOsmParameters:
    """Parsed parameter values from an OSM feature dataset identifier."""

    osm_smoothness_mapping: str | None
    osm_surface_mapping: str | None
    osm_combination_mapping: str | None


@dataclass
class DataTestCase:
    """One experiment dataset combination for a scenario A, B, or C run."""
    case_id: str  # {case_group}__{manual_ds_id}__{osm_ds_id}
    ds_version: str

    osm_ds: pd.DataFrame | None
    osm_ds_id: str | None
    osm_parameters: DataTestCaseOsmParameters

    manual_ds: pd.DataFrame | None
    manual_ds_id: str | None
    manual_parameters: DataTestCaseManualParameters


MANUAL_PARAMETERS_MAP = {
    'radius': 'manual_radius',
    'mappingprocedure': 'manual_mapping_procedure',
    'timethreshold': 'manual_time_threshold',
    'vehicletype': 'manual_vehicle_type',
}

OSM_PARAMETERS_MAP = {
    'sm': 'osm_smoothness_mapping',
    'surf': 'osm_surface_mapping',
    'c': 'osm_combination_mapping',
}


def get_test_dataset_configurations(feature_datasets: dict[str, dict[str, pd.DataFrame]],
                                    experiment_config: ExperimentConfig) -> list[DataTestCase]:
    """Build all requested A/B/C dataset combinations from loaded feature datasets."""
    if experiment_config.case_type == 'A':
        return _get_scenario_a_cases(feature_datasets, experiment_config)
    elif experiment_config.case_type == 'B':
        return _get_scenario_b_cases(feature_datasets, experiment_config)
    elif experiment_config.case_type == 'C':
        return _get_scenario_c_cases(feature_datasets, experiment_config)
    else:
        raise ValueError(f'Unknown test case: {experiment_config.case_type}')


def _get_scenario_a_cases(
        feature_datasets: dict[str, dict[str, pd.DataFrame]],
        experiment_config: ExperimentConfig,
) -> list[DataTestCase]:
    """Create case A configurations where manual datasets provide train and test data."""
    # manuals only
    chosen_scenario_cases = []
    for manual_ds_id, manual_ds_df in feature_datasets['manual'].items():
        osm_parameters = _parse_none_case_ds_parameters('osm')
        manual_parameters = _parse_ds_parameters(manual_ds_id)
        chosen_scenario_cases.append(
            DataTestCase(
                case_id=experiment_config.get_case_id(manual_ds_id=manual_ds_id, osm_ds_id='None'),
                ds_version=experiment_config.ds_version,
                osm_ds=None,
                osm_ds_id=None,
                osm_parameters=osm_parameters,
                manual_ds=manual_ds_df,
                manual_ds_id=manual_ds_id,
                manual_parameters=manual_parameters,
            )
        )
    return chosen_scenario_cases


def _get_scenario_b_cases(
        feature_datasets: dict[str, dict[str, pd.DataFrame]],
        experiment_config: ExperimentConfig,
) -> list[DataTestCase]:
    """Create case B configurations for every manual and OSM dataset pairing."""
    # both
    chosen_scenario_cases = []
    for manual_ds_id, manual_ds_df in feature_datasets['manual'].items():
        for osm_ds_id, osm_ds_df in feature_datasets['osm'].items():
            osm_parameters = _parse_ds_parameters(osm_ds_id)
            manual_parameters = _parse_ds_parameters(manual_ds_id)
            chosen_scenario_cases.append(
                DataTestCase(
                    case_id=experiment_config.get_case_id(manual_ds_id=manual_ds_id, osm_ds_id=osm_ds_id),
                    ds_version=experiment_config.ds_version,
                    osm_ds=osm_ds_df,
                    osm_ds_id=osm_ds_id,
                    osm_parameters=osm_parameters,
                    manual_ds=manual_ds_df,
                    manual_ds_id=manual_ds_id,
                    manual_parameters=manual_parameters,
                )
            )
    return chosen_scenario_cases


def _get_scenario_c_cases(
        feature_datasets: dict[str, dict[str, pd.DataFrame]],
        experiment_config: ExperimentConfig,
) -> list[DataTestCase]:
    """Create case C configurations where OSM trains and manual datasets provide tests."""
    # osm only
    chosen_scenario_cases = []
    for osm_ds_id, osm_ds_df in feature_datasets['osm'].items():
        for manual_ds_id, manual_ds_df in feature_datasets['manual'].items():
            manual_parameters = _parse_ds_parameters(manual_ds_id)
            osm_parameters = _parse_ds_parameters(osm_ds_id)
            chosen_scenario_cases.append(
                DataTestCase(
                    case_id=experiment_config.get_case_id(manual_ds_id=manual_ds_id, osm_ds_id=osm_ds_id),
                    ds_version=experiment_config.ds_version,
                    osm_ds=osm_ds_df,
                    osm_ds_id=osm_ds_id,
                    osm_parameters=osm_parameters,
                    manual_ds=manual_ds_df,
                    manual_ds_id=manual_ds_id,
                    manual_parameters=manual_parameters,
                )
            )
    return chosen_scenario_cases


def _parse_ds_parameters(ds_id: str) -> DataTestCaseManualParameters | DataTestCaseOsmParameters:
    """Parse manual or OSM dataset parameters from a normalized dataset identifier."""
    id_contents = ds_id.split('.')[0].split('_')
    mode = id_contents[0]
    parameters = id_contents[2:]
    p_mapping = MANUAL_PARAMETERS_MAP if mode == 'manual' else OSM_PARAMETERS_MAP
    parsed = {}
    for p_description in parameters:
        for p_name_short, p_name_long in p_mapping.items():
            if p_name_short in p_description:
                parsed[p_name_long] = p_description.split(p_name_short)[1]
                break
    parsed = parsed | {p: None for p in p_mapping.values() if p not in parsed}
    return DataTestCaseManualParameters(**parsed) if mode == 'manual' else DataTestCaseOsmParameters(**parsed)


def _parse_none_case_ds_parameters(mode: str) -> DataTestCaseManualParameters | DataTestCaseOsmParameters:
    """Return empty parameter values for a missing manual or OSM side of a test case."""
    if mode not in ['manual', 'osm']:
        raise ValueError(f'Invalid mode: {mode} for None-case parsing.')
    parameters = MANUAL_PARAMETERS_MAP if mode == 'manual' else OSM_PARAMETERS_MAP
    parsed = {p: None for p in parameters.values()}
    return DataTestCaseManualParameters(**parsed) if mode == 'manual' else DataTestCaseOsmParameters(**parsed)
