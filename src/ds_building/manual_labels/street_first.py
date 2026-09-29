"""Street-measurement-first matching for road-quality dataset construction.

This module starts from each street sensor measurement, finds manual labels
recorded nearby and around the same time, and assigns the most frequent of them
to the measurement.
"""

from tqdm import tqdm
import pandas as pd
from typing import Any

from src.ds_building.manual_labels import utils

def sort_vehicle_types(df_street: pd.DataFrame, vehicle_type: str) -> pd.DataFrame:
    """Return street measurement rows for one vehicle type.

    Args:
        df_street: Street measurement DataFrame with a `vehicleType` column.
        vehicle_type: Vehicle type string to keep.

    Returns:
        Filtered DataFrame containing only matching vehicle rows.
    """
    return df_street.loc[df_street["vehicleType"] == vehicle_type]

def compute_first_sort_dict(
    df_labels: pd.DataFrame,
    df_vehicle_street: pd.DataFrame,
    speed_threshold: float = 7,
    radius: float = 5,
    time_window_s: float = 60,
) -> dict[int, pd.DataFrame]:
    """Find nearby manual labels for each street measurement row.

    Applies vehicle-speed, a coarse coordinate box sized to `radius`,
    geodesic-radius, and time-window filters to identify manual labels that can
    be assigned to each street measurement. The geodesic distance to the street
    row is kept in a `distance` column.

    Args:
        df_labels: DataFrame containing manual labels with `lat`, `lon`, `label`,
            and `timestamp`.
        df_vehicle_street: Street measurement DataFrame already filtered to one
            vehicle type.
        speed_threshold: Minimum street-measurement speed to consider.
        radius: Maximum accepted point-to-point distance in metres.
        time_window_s: Maximum absolute time difference between street row and
            label, in seconds.

    Returns:
        Dict mapping street-row indices to nearby manual-label rows.
    """
    first_sort_dict = {}

    for i in tqdm(df_vehicle_street.index):
        if df_vehicle_street["speed"][i] > speed_threshold:
            lat_threshold, lon_threshold = utils.compute_coarse_box(radius, df_vehicle_street["lat"][i])
            candidates = df_labels[(abs(df_vehicle_street["lon"][i]-df_labels["lon"]) < lon_threshold) &
                                   (abs(df_vehicle_street["lat"][i]-df_labels["lat"]) < lat_threshold)]
            distances = [utils.compute_distance(row["lat"], row["lon"], df_vehicle_street["lat"][i], df_vehicle_street["lon"][i]).m
                         for _, row in candidates.iterrows()]
            candidates = candidates.assign(distance=distances)
            candidates = candidates.loc[candidates["distance"] <= radius]
            first_sort_dict[i] = utils.filter_by_time_window(candidates, df_vehicle_street["timestamp"][i], time_window_s)

    return first_sort_dict


def compute_vehicle_type_dict(
    df_labels: pd.DataFrame,
    df_street: pd.DataFrame,
    speed_threshold: float = 7,
    radius: float = 5,
    time_window_s: float = 60,
) -> dict[str, dict[int, pd.DataFrame]]:
    """Build nearby-label mappings for each supported vehicle type.

    Args:
        df_labels: DataFrame containing manual label points.
        df_street: DataFrame containing all street measurements.
        speed_threshold: Minimum street-measurement speed to consider.
        radius: Maximum accepted point-to-point distance in metres.
        time_window_s: Maximum absolute time difference between street row and
            label, in seconds.

    Returns:
        Nested dict keyed by vehicle type and then street-row index.
    """
    vehicle_type_dict = {}

    for vehicle_type in ["Car", "Bike", "E-Scooter"]:
        vehicle_type_dict[vehicle_type] = compute_first_sort_dict(df_labels, sort_vehicle_types(df_street, vehicle_type),
                                                                  speed_threshold=speed_threshold,
                                                                  radius=radius,
                                                                  time_window_s=time_window_s)

    print(vehicle_type_dict["Car"][list(vehicle_type_dict["Car"].keys())[0]].iloc[:30])
    return vehicle_type_dict

def most_frequent_label(labels: pd.DataFrame) -> Any:
    """Return the most common label, breaking ties by the nearest label.

    Args:
        labels: Nearby manual-label rows with `label` and `distance` columns.

    Returns:
        Label value with the highest occurrence count. If several labels are
        equally common, the one belonging to the nearest label point wins.
    """
    counts = labels["label"].value_counts()
    tied_labels = counts[counts == counts.max()].index
    candidates = labels.loc[labels["label"].isin(tied_labels)]
    return candidates.loc[candidates["distance"].idxmin(), "label"]


def create_data_set(
    df_street: pd.DataFrame,
    vehicle_type_dict: dict[str, dict[int, pd.DataFrame]],
    mapping_procedure: str,
    vehicle_type: str = "Car",
) -> list[dict[str, Any]]:
    """Create vibration/label examples using labels near each street point.

    Currently supports `mostfrequent`, which assigns the most common nearby manual
    label to each qualifying street measurement; ties go to the nearest label.

    Args:
        df_street: Street measurement DataFrame containing vibration columns.
        vehicle_type_dict: Nested vehicle type mapping returned by
            `compute_vehicle_type_dict`.
        mapping_procedure: Mapping strategy name; currently `mostfrequent`.
        vehicle_type: Vehicle type to extract from `vehicle_type_dict`.

    Returns:
        List of dicts with flat `vibration_x`, `vibration_y`, `vibration_z`,
        `speed`, `label`, `lon`, `lat`, and `timestamp` fields.
    """
    data_set = []
    for i, labels in vehicle_type_dict[vehicle_type].items():
        if not labels.empty:
            if mapping_procedure == "mostfrequent":
                street_row = df_street.loc[i]
                data_set.append({"vibration_x": street_row["vibration_x"],
                                "vibration_y": street_row["vibration_y"],
                                "vibration_z": street_row["vibration_z"],
                                "speed": street_row["speed"],
                                "label": most_frequent_label(labels),
                                "lon": street_row["lon"],
                                "lat": street_row["lat"],
                                "timestamp": street_row["timestamp"]
                                 }
                )
    print(data_set[:100])
    return data_set
