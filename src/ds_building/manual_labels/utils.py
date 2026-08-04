"""Shared utility functions for loading sensor data and measuring distances."""

import pandas as pd
from geopy import distance

def load_data(path: str) -> pd.DataFrame:
    """Load a CSV file and parse its `timestamp` column as datetimes.

    Args:
        path: Path to the CSV file to read.

    Returns:
        DataFrame with `timestamp` converted to pandas datetime values.
    """
    df = pd.read_csv(path)
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
