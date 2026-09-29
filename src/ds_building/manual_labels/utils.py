"""Shared utility functions for loading sensor data and measuring distances."""

from pathlib import Path

import numpy as np
import pandas as pd
from geopy import distance

# Lower bounds for metres per degree on WGS84, so the coarse box never
# undercuts the geodesic radius check that follows it.
METRES_PER_DEGREE_LAT = 110_574.0
METRES_PER_DEGREE_LON_AT_EQUATOR = 111_319.0

def load_data(path: str) -> pd.DataFrame:
    """Load one CSV file or all CSV files in a directory.

    Args:
        path: Path to a CSV file or directory of CSV files.

    Returns:
        DataFrame with `timestamp` converted to pandas datetime values.
    """
    input_path = Path(path)
    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {path}")
        df = pd.concat([pd.read_csv(file_path) for file_path in csv_files], ignore_index=True)
    else:
        df = pd.read_csv(input_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"],format="ISO8601")
    return df


def compute_coarse_box(radius: float, lat: float) -> tuple[float, float]:
    """Return latitude/longitude half-widths in degrees covering `radius` metres.

    The box is used as a cheap prefilter before the exact geodesic distance
    check, so it must be at least as large as the radius in every direction.
    The longitude half-width grows with latitude.

    Args:
        radius: Matching radius in metres.
        lat: Latitude of the box centre in degrees.

    Returns:
        Tuple of `(lat_threshold, lon_threshold)` in degrees.
    """
    lat_threshold = radius / METRES_PER_DEGREE_LAT
    lon_threshold = radius / (METRES_PER_DEGREE_LON_AT_EQUATOR * np.cos(np.radians(lat)))
    return lat_threshold, lon_threshold


def filter_by_time_window(
    candidates: pd.DataFrame,
    reference_timestamp: pd.Timestamp,
    time_window_s: float,
) -> pd.DataFrame:
    """Keep candidate rows recorded within `time_window_s` seconds of a reference time.

    Labels and street measurements come from phones travelling in the same car, so
    a label is only matched to measurements taken around the same moment. This
    excludes other passes over the same spot and measurements from other drives.

    Args:
        candidates: DataFrame of candidate rows with a `timestamp` column.
        reference_timestamp: Timestamp of the label or street row being matched.
        time_window_s: Maximum absolute time difference in seconds.

    Returns:
        Filtered DataFrame of candidate rows.
    """
    time_diff = (candidates["timestamp"] - reference_timestamp).abs()
    return candidates.loc[time_diff <= pd.Timedelta(seconds=time_window_s)]


def compute_distance(lat_1: float, lon_1: float, lat_2: float, lon_2: float) -> distance.Distance:
    """Calculate the geodesic distance between two latitude/longitude points.

    Args:
        lat_1: Latitude of the first point.
        lon_1: Longitude of the first point.
        lat_2: Latitude of the second point.
        lon_2: Longitude of the second point.

    Returns:
        geopy distance object for the two coordinates.
    """
    return distance.distance((lat_1, lon_1),(lat_2, lon_2))
