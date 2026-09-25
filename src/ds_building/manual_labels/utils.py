"""Shared utility functions for loading sensor data and measuring distances."""

from pathlib import Path

import pandas as pd
from geopy import distance

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
