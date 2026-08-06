"""Build OSM-derived label scenarios and join them to street measurements.

When run as a script, this module maps nearest-way OSM labels into every
configured numeric scenario, saves those scenario CSVs, and writes joined street
datasets for training.
"""

import os
from collections.abc import Iterator

import pandas as pd

from combination_mappings import calc_save_combination_scenarios
from smoothness_mappings import calc_smoothness_scenarios
from surface_mappings import calc_surface_scenarios


def clean_labeled_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unusable and duplicate label rows.

    Drops rows where both `smoothness` and `surface` are missing, then drops exact
    duplicate rows, printing a short before/after summary.

    Args:
        df: DataFrame containing at least `smoothness` and `surface` columns.

    Returns:
        Cleaned copy of the input DataFrame.
    """
    df = df.copy()
    n0 = len(df)

    drop_missing = df["smoothness"].isna() & df["surface"].isna()
    d1 = int(drop_missing.sum())
    df = df.loc[~drop_missing]

    n1 = len(df)
    df = df.drop_duplicates()
    d2 = n1 - len(df)

    print(f"Cleaned labeled locations: start={n0} | dropped_missing={d1} | dropped_dupes={d2} | end={len(df)}")
    return df

def sm_surf_variations(
    sm_score_columns:dict[str, pd.Series],
    surf_score_columns:dict[str, list[pd.Series]],
) -> Iterator[tuple[str, str, pd.Series, list[pd.Series]]]:
    """Yield all smoothness/surface scenario combinations with their score columns.

    Args:
        sm_score_columns: Mapping of smoothness scenario id -> smoothness score Series.
        surf_score_columns: Mapping of surface scenario id -> list of surface score Series.

    Yields:
        Tuples: (sm_scenario, surf_scenario, sm_score_series, surf_score_series_list).
    """
    for sm_scenario, sm_col in sm_score_columns.items():
        for surf_scenario, surf_col in surf_score_columns.items():
            yield sm_scenario, surf_scenario, sm_col, surf_col

def build_mapped_labels(labeled_location_file:str, ds_save_dir:str) -> None:
    """Compute and save all mapped label combination scenarios for a label parquet.

    Loads labeled locations from `labeled_location_file`, cleans them, computes all
    configured smoothness and surface score scenarios, then writes each configured
    combination scenario output under `ds_save_dir`.

    Args:
        labeled_location_file: Parquet path containing labeled locations.
        ds_save_dir: Output directory for scenario parquet files.

    Returns:
        None.
    """
    labeled_locations = pd.read_parquet(labeled_location_file)
    labeled_locations = clean_labeled_locations(labeled_locations)

    sm_score_columns = calc_smoothness_scenarios(labeled_locations)
    surf_score_columns = calc_surface_scenarios(labeled_locations)
    print(f'Smoothness score columns: {sm_score_columns}')
    print(f'Surface score columns: {surf_score_columns}')

    for sm_name, surf_name, sm_score_column, surf_score_column in sm_surf_variations(sm_score_columns, surf_score_columns):
        print(f'> CALCULATING COMBINATION SCENARIOS FOR: smoothness = {sm_name} | surface = {surf_name}')
        calc_save_combination_scenarios(labeled_locations, sm_name, surf_name, sm_score_column, surf_score_column, ds_save_dir)

# ----------------------------------------------------------------------------------------------------------------------

def join_labels_to_streets(read_dir_labels: str, read_dir_street: str, out_dir: str) -> None:
    """Join per-file label datasets onto a combined street dataset and save per label file.

    Concatenates all street CSV files in `read_dir_street` into one DataFrame, then
    for each label parquet in `read_dir_labels` performs a left join on (`lat`, `lon`).
    Rows with missing labels are dropped and the result is saved to `out_dir`.

    Args:
        read_dir_labels: Directory containing label parquet files (expects `lat`, `lon`, `label`).
        read_dir_street: Directory containing street CSV files.
        out_dir: Output directory for joined datasets.

    Returns:
        None.
    """
    os.makedirs(out_dir, exist_ok=True)

    street_files = [f for f in os.listdir(read_dir_street) if f.lower().endswith(".csv")]
    streets = pd.concat(
        [pd.read_csv(os.path.join(read_dir_street, f)) for f in sorted(street_files)],
        ignore_index=True,
    )

    label_files = [f for f in os.listdir(read_dir_labels) if f.lower().endswith(".parquet")]
    for f in sorted(label_files):
        print(f"joining {f} ...")
        label_cols = ["lat", "lon", "label", "smoothness", "surface", 'smoothness_score', 'surface_score']
        labels = pd.read_parquet(os.path.join(read_dir_labels, f))[label_cols].dropna(subset=["label"])
        out = streets.merge(labels, on=["lat", "lon"], how="left")
        out = out.dropna(subset=["label"])
        f = f.replace("labels", "dataset")
        out.to_parquet(os.path.join(out_dir, f), index=False)

labeled_location_file = f'data/open_street_map/label_steps/labeled_location_data/labeled_locations.parquet'
labels_save_dir = f'data/open_street_map/label_steps/mapped_labels'
build_mapped_labels(labeled_location_file, labels_save_dir)

street_read_dir = 'data/molewa/raw'
osm_save_dir = 'data/open_street_map/datasets'
join_labels_to_streets(labels_save_dir, street_read_dir, osm_save_dir)
