import dataclasses
import os
import pathlib
from pathlib import Path

import pandas as pd

import features

REQUIRED_NUMBER_COLS = [
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "speed",
    "longitude",
    "latitude",
]

FEATURE_COLS = [
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "vibration_magnitude",
    "speed",
    "score_mild",
    "score_standard",
    "score_strict",
]

METADATA_COLS = [
    "label",
    "timestamp",
    "longitude",
    "latitude",
]


@dataclasses.dataclass(frozen=True)
class DataConfig:
    """Configuration for building or loading model feature datasets.

    Attributes:
        osm_ds_dir: Directory containing OpenStreetMap-labelled parquet datasets.
        manual_ds_dir: Directory containing manually labelled parquet datasets.
        feature_ds_dir: Optional output/input directory for saved feature datasets.
        skip_feature_build_if_exists: Reuse saved feature datasets when available.
    """

    osm_ds_dir: str
    manual_ds_dir: str
    feature_ds_dir: str | None
    skip_feature_build_if_exists: bool


def _build_ds_group(file_paths: list[Path]) -> dict[str, pd.DataFrame]:
    """Build prepared feature datasets for a collection of input parquet files."""
    ds_group = {}
    for file_path in file_paths:
        ds_id, df = _read_single_label_ds(file_path)
        df = _prep_ds(df)
        df = features.add_scores(df)
        df = _sort_columns(df)
        ds_group[ds_id] = df
    return ds_group


def _read_single_label_ds(file_path: Path) -> tuple[str, pd.DataFrame]:
    """Read one labelled parquet dataset and derive its dataset identifier."""
    raw_data = pd.read_parquet(file_path)
    ds_name = file_path.stem
    ds_id = "_".join(ds_name.split("_")[2:])
    return ds_id, raw_data


def _parse_label_num(label_num: str | float | int) -> float:
    """Convert text or numeric road-quality labels to numeric label values."""
    parse_dict = {
        "good": 0.0,
        "medium": 1.0,
        "bad": 2.0,
    }
    if isinstance(label_num, str):
        label_num = label_num.strip().lower()
    if label_num in parse_dict:
        return parse_dict[label_num]
    return float(label_num)


def _prep_ds(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize input columns, parse types, and return the modelling base table."""
    raw_df = raw_df.rename(columns={
        "lon": "longitude",
        "lat": "latitude",
    })

    missing_cols = [
        col
        for col in ["timestamp", "label"] + REQUIRED_NUMBER_COLS
        if col not in raw_df.columns
    ]
    if missing_cols:
        missing_col_list = ", ".join(missing_cols)
        raise ValueError(f"Missing required columns: {missing_col_list}")

    df = raw_df[["timestamp", "label"] + REQUIRED_NUMBER_COLS].copy()

    for col in REQUIRED_NUMBER_COLS:
        df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df["label"] = df["label"].apply(_parse_label_num)
    return df


def _sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature and metadata columns in the expected output order."""
    return df[FEATURE_COLS + METADATA_COLS]


def _get_parquet_file_paths(input_dir: str) -> list[Path]:
    """Find all parquet files below an input directory."""
    file_paths = []
    for file in pathlib.Path(input_dir).rglob("*.parquet"):
        file_paths.append(file)
    return file_paths


def _save_feature_ds(
    datasets: dict[str, pd.DataFrame],
    output_dir: str,
    ds_type: str,
) -> None:
    """Save prepared feature datasets into the configured dataset-type folder."""
    for ds_id, df in datasets.items():
        file_path = os.path.join(output_dir, f"{ds_type}/{ds_type}_feature_ds_{ds_id}.parquet")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_parquet(file_path)
    print(f"Saved {len(datasets)} feature_ds files to {output_dir}/{ds_type}")


def _read_feature_group(feature_type_ds_dir: str) -> dict[str, pd.DataFrame]:
    """Read saved feature datasets from one dataset-type folder."""
    feat_group_datasets = {}
    for feature_ds_path in pathlib.Path(feature_type_ds_dir).glob("*.parquet"):
        df = pd.read_parquet(feature_ds_path)
        ds_id = "_".join([p for p in feature_ds_path.name.split(".")[0].split("_")[3:]])
        feat_group_datasets[ds_id] = df
    return feat_group_datasets


def _load_features_from_existing_dir(feature_dir: str) -> dict[str, dict[str, pd.DataFrame]]:
    """Load saved OSM and manual feature datasets from an existing feature directory."""
    print(
        "Features exist already. Not building new features. "
        f"Reading features from {feature_dir}. If you do not want that, set "
        "config parameter: skip_feature_build_if_exists=False."
    )
    return {
        "osm": _read_feature_group(f"{feature_dir}/osm"),
        "manual": _read_feature_group(f"{feature_dir}/manual"),
    }


def _check_for_saved_feature_data(feature_ds_path: str | None) -> bool:
    """Return whether saved OSM and manual feature datasets are available."""
    if feature_ds_path is None:
        return False

    dirs = [
        f"{feature_ds_path}/osm",
        f"{feature_ds_path}/manual",
    ]
    for ds_dir in dirs:
        if not os.path.exists(ds_dir):
            return False
        if not list(pathlib.Path(ds_dir).glob("*.parquet")):
            return False
    return True


def load_feature_ds(config: DataConfig) -> dict[str, dict[str, pd.DataFrame]]:
    """Load existing feature datasets or build them from labelled parquet files."""
    if config.skip_feature_build_if_exists and _check_for_saved_feature_data(config.feature_ds_dir):
        return _load_features_from_existing_dir(config.feature_ds_dir)

    print(
        "Building new feature datasets from "
        f"{config.osm_ds_dir} and {config.manual_ds_dir}"
    )
    osm_feat_datasets = _build_ds_group(_get_parquet_file_paths(config.osm_ds_dir))
    manual_feat_datasets = _build_ds_group(_get_parquet_file_paths(config.manual_ds_dir))

    if config.feature_ds_dir:
        _save_feature_ds(osm_feat_datasets, config.feature_ds_dir, "osm")
        _save_feature_ds(manual_feat_datasets, config.feature_ds_dir, "manual")

    return {
        "osm": osm_feat_datasets,
        "manual": manual_feat_datasets,
    }
