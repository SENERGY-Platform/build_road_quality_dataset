"""Build datasets from manually labeled road-quality points.

Run this module as a script to generate both labels-first and street-first
dataset variants from the configured CSV inputs.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.ds_building.manual_labels import labels_first
from src.ds_building.manual_labels import street_first
from src.ds_building.manual_labels import utils


@dataclass(frozen=False)
class ManualLabelsConfig:
    """Configuration for the manual-label dataset build."""

    labels_path: str
    street_path: str
    output_dir: str

    mapping_type: str = 'labels_first' # 'labels_first'/'street_first'
    mapping_procedure: str = "average" # "single", "average", "mostfrequent"
    vehicle_type: str = "Car"

    lon_threshold: float = 8e-05
    lat_threshold: float = 6e-05
    speed_threshold: float = 7
    time_threshold: int = 10
    radius: float = 2


LABELS_FIRST_MAPPING_PROCEDURES = ("single", "average")
STREET_FIRST_MAPPING_PROCEDURES = ("mostfrequent",)
DEFAULT_MAPPING_COMBINATIONS = (
    ("labels_first", "single"),
    ("labels_first", "average"),
    ("street_first", "mostfrequent"),
)


def build_labels_first_dataset(
    df_labels: pd.DataFrame,
    df_street: pd.DataFrame,
    config: ManualLabelsConfig,
) -> list[dict[str, Any]]:
    """Build a dataset by assigning nearby street measurements to manual labels."""
    first_sort_dict = labels_first.compute_first_sort_dict(
        df_labels,
        df_street,
        lon_threshold=config.lon_threshold,
        lat_threshold=config.lat_threshold,
        speed_threshold=config.speed_threshold,
        radius=config.radius,
    )
    vehicle_type_dict = labels_first.compute_vehicle_type_dict(
        first_sort_dict,
        time_threshold=config.time_threshold,
    )
    return labels_first.create_data_set(
        df_labels,
        vehicle_type_dict,
        mapping_procedure=config.mapping_procedure,
        vehicle_type=config.vehicle_type,
    )


def build_street_first_dataset(
    df_labels: pd.DataFrame,
    df_street: pd.DataFrame,
    config: ManualLabelsConfig,
) -> list[dict[str, Any]]:
    """Build a dataset by assigning nearby manual labels to street measurements."""
    vehicle_type_dict = street_first.compute_vehicle_type_dict(
        df_labels,
        df_street,
        lon_threshold=config.lon_threshold,
        lat_threshold=config.lat_threshold,
        speed_threshold=config.speed_threshold,
        radius=config.radius,
    )
    return street_first.create_data_set(
        df_street,
        vehicle_type_dict,
        mapping_procedure=config.mapping_procedure,
        vehicle_type=config.vehicle_type,
    )


def output_path_for_config(config: ManualLabelsConfig) -> str:
    """Return the base output path for the configured pipeline mode."""
    if config.mapping_type == "labels_first":
        subdir = "labels_first"
        filename = (
            f"manual_dataset_"
            f"radius{config.radius}_"
            f"mappingprocedure{config.mapping_procedure}_"
            f"timethreshold{config.time_threshold}_"
            f"vehicletype{config.vehicle_type}"
        )
    else:
        subdir = "street_first"
        filename = (
            f"manual_dataset_"
            f"radius{config.radius}_"
            f"mappingprocedure{config.mapping_procedure}_"
            f"vehicletype{config.vehicle_type}"
        )

    output_dir = os.path.join(config.output_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


def save_dataset(data_set: list[dict[str, Any]], file_path: str) -> None:
    """Save a dataset to a parquet file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pd.DataFrame(data_set).to_parquet(
        f"{file_path}.parquet",
        index=False,
    )


def validate_config(config: ManualLabelsConfig) -> None:
    """Validate mapping type and procedure compatibility."""
    if config.mapping_type == "labels_first":
        if config.mapping_procedure not in LABELS_FIRST_MAPPING_PROCEDURES:
            raise ValueError(
                f"Unsupported labels_first mapping procedure: {config.mapping_procedure}. "
                f"Use one of {LABELS_FIRST_MAPPING_PROCEDURES}."
            )
    elif config.mapping_type == "street_first":
        if config.mapping_procedure not in STREET_FIRST_MAPPING_PROCEDURES:
            raise ValueError(
                f"Unsupported street_first mapping procedure: {config.mapping_procedure}. "
                f"Use one of {STREET_FIRST_MAPPING_PROCEDURES}."
            )
    else:
        raise ValueError("mapping_type must be 'labels_first' or 'street_first'.")


def run_pipeline(config: ManualLabelsConfig) -> str:
    """Run the configured manual-label pipeline and return the output base path."""
    validate_config(config)

    df_labels = utils.load_data(config.labels_path)
    df_street = utils.load_data(config.street_path)

    if config.mapping_type == "labels_first":
        data_set = build_labels_first_dataset(df_labels, df_street, config)
    else:
        data_set = build_street_first_dataset(df_labels, df_street, config)

    file_path = output_path_for_config(config)
    save_dataset(data_set, file_path)
    return file_path


def run_default_pipelines(
    labels_path: str = "data/molewa/labels",
    street_path: str = "data/molewa/raw",
    output_dir: str = "data/molewa/datasets",
) -> None:
    """Run all default manual-label dataset variants."""
    config = ManualLabelsConfig(
        labels_path=labels_path,
        street_path=street_path,
        output_dir=output_dir,
    )
    for mapping_type, mapping_procedure in DEFAULT_MAPPING_COMBINATIONS:
        print(f"Starting pipeline for mapping_type: {mapping_type}, and procedure: {mapping_procedure}.")
        config.mapping_type = mapping_type
        config.mapping_procedure = mapping_procedure
        output_path = run_pipeline(config)
        print(f"Saved dataset to {output_path}.parquet")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build manual-label road-quality datasets.",
    )
    parser.add_argument("--labels-path", default="data/molewa/labels")
    parser.add_argument("--street-path", default="data/molewa/raw")
    parser.add_argument("--output-dir", default="data/molewa/datasets")
    return parser.parse_args()


def main() -> None:
    """Parse CLI arguments and run all default manual-label dataset variants."""
    args = _parse_args()
    run_default_pipelines(
        labels_path=args.labels_path,
        street_path=args.street_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
