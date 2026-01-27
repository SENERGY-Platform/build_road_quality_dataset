import json
import os
from datetime import datetime
from typing import  Optional, Sequence, Set, Tuple
import pandas as pd

def read_locations(
    dir_path: str,
    n_rows_per_file: Optional[int] = None,
) -> Set[Tuple[float, float]]:
    """Return unique (lon, lat) tuples from all CSV files in `dir_path`.

    Reads each *.csv, expects `lon` and `lat` columns, drops NaNs and duplicates
    per file, and returns a set of float tuples.

    Args:
        dir_path: Directory containing CSV files.
        n_rows_per_file: Optional row limit per file for faster sampling.

    Returns:
        A set of (lon, lat) coordinate tuples.
    """
    points: Set[Tuple[float, float]] = set()

    if not os.path.isdir(dir_path):
        return points

    for fname in os.listdir(dir_path):
        if not fname.lower().endswith(".csv"):
            continue

        file_path = os.path.join(dir_path, fname)
        df = pd.read_csv(file_path, nrows=n_rows_per_file)

        if not {"lon", "lat"}.issubset(df.columns):
            continue

        coords = df[["lon", "lat"]].dropna()

        for lon, lat in coords.drop_duplicates().itertuples(index=False, name=None):
            points.add((float(lon), float(lat)))

    return points


def read_new_locations(
    dir_path: str,
    save_dir: str,
    n_rows_per_file: Optional[int] = None,
) -> Set[Tuple[float, float]]:
    """Return points present in `dir_path` but not yet present in `save_dir`.

    Loads locations from both directories (if `save_dir` exists) and returns the
    set difference.

    Args:
        dir_path: Directory containing new CSV files.
        save_dir: Directory containing previously processed CSV files.
        n_rows_per_file: Optional row limit per file for faster sampling.

    Returns:
        A set of (lon, lat) tuples that are new.
    """
    points = read_locations(dir_path, n_rows_per_file=n_rows_per_file)
    _log("INFO", f"Loaded {len(points)} locations from {dir_path}")

    if not os.path.isdir(save_dir):
        _log("WARN", f"Save dir {save_dir} does not exist yet. Treating all points as new.")
        return points

    existing = read_locations(save_dir, n_rows_per_file=None)
    new_points = points.difference(existing)
    _log("INFO", f"{len(new_points)} of these locations are new.")
    return new_points

def save_raw_data(save_dir, points_df, payload, batch, batch_idx, time):
    """Persist an Overpass batch payload and update a CSV of requested points.

    Writes the raw API response (elements + batch points) as a JSON file under
    `<save_dir>/payloads/` and appends the batch points to `points_df`, then saves
    the updated points CSV under `<save_dir>/requested_points/`.

    Args:
        save_dir: Base output directory.
        points_df: Existing DataFrame of requested points (lon/lat).
        payload: Overpass JSON response (expects an `elements` key).
        batch: Iterable of (lon, lat) tuples for this request batch.
        batch_idx: Batch index used for the JSON filename.
        time: Datetime used for timestamping filenames.

    Returns:
        Updated `points_df` with the current batch appended.
    """
    save_subdir = os.path.join(save_dir, "payloads")
    os.makedirs(save_subdir, exist_ok=True)
    raw_path = os.path.join(save_subdir, f"{time.isoformat()}_overpass_batch_{batch_idx:05d}.json")

    raw_data = {"points": batch, "elements": payload["elements"]}
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False)
    _log("INFO", f"Saved raw payload JSON: {raw_path}")

    batch_df = pd.DataFrame(list(batch), columns=["lon", "lat"])
    points_df = pd.concat([points_df, batch_df], ignore_index=True)

    # (optional) remove duplicates if batches can overlap
    # points_df = points_df.drop_duplicates(subset=["lon", "lat"], keep="first")

    save_subdir = os.path.join(save_dir, "requested_points")
    os.makedirs(save_subdir, exist_ok=True)
    df_path = os.path.join(save_subdir, f"requested_points_at_{time.isoformat()}.csv")
    points_df.to_csv(df_path, index=False)
    _log("INFO", f"Updated requested_points CSV ({len(points_df)} total points): {df_path}")

    return points_df


def _log(level: str, message: str) -> None:
    """Print a timestamped log line to stdout."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {message}")