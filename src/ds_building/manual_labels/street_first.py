"""Street-measurement-first matching for road-quality dataset construction.

This module starts from each street sensor measurement, finds nearby manual
labels, and assigns the most frequent nearby label to the measurement.
"""

from tqdm import tqdm
import pandas as pd
from typing import Any

import utils

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
    lon_threshold: float = 3e-05,
    lat_threshold: float = 2e-05,
    speed_threshold: float = 7,
    radius: float = 2,
) -> dict[int, pd.DataFrame]:
    """Find nearby manual labels for each street measurement row.

    Applies vehicle-speed, coordinate-threshold, and geodesic-radius filters to
    identify manual labels that can be assigned to each street measurement.

    Args:
        df_labels: DataFrame containing manual labels with `lat`, `lon`, and `label`.
        df_vehicle_street: Street measurement DataFrame already filtered to one
            vehicle type.
        lon_threshold: Maximum absolute longitude difference for the coarse filter.
        lat_threshold: Maximum absolute latitude difference for the coarse filter.
        speed_threshold: Minimum street-measurement speed to consider.
        radius: Maximum accepted point-to-point distance in metres.

    Returns:
        Dict mapping street-row indices to nearby manual-label rows.
    """
    first_sort_dict = {}

    for i in tqdm(df_vehicle_street.index):
        if df_vehicle_street["speed"][i] > speed_threshold:
            first_sort_dict[i] = df_labels[(abs(df_vehicle_street["lon"][i]-df_labels["lon"]) < lon_threshold) &
                                           (abs(df_vehicle_street["lat"][i]-df_labels["lat"]) < lat_threshold)]
            indices_far = []
            for index, row in first_sort_dict[i].iterrows():
                if utils.compute_distance(row["lat"], row["lon"], df_vehicle_street["lat"][i], df_vehicle_street["lon"][i]).m > radius:
                    indices_far.append(index)
            first_sort_dict[i] = first_sort_dict[i].drop(indices_far)
            
    return first_sort_dict


def compute_vehicle_type_dict(
    df_labels: pd.DataFrame,
    df_street: pd.DataFrame,
    lon_threshold: float = 3e-05,
    lat_threshold: float = 2e-05,
    speed_threshold: float = 7,
    radius: float = 2,
) -> dict[str, dict[int, pd.DataFrame]]:
    """Build nearby-label mappings for each supported vehicle type.

    Args:
        df_labels: DataFrame containing manual label points.
        df_street: DataFrame containing all street measurements.
        lon_threshold: Maximum absolute longitude difference for the coarse filter.
        lat_threshold: Maximum absolute latitude difference for the coarse filter.
        speed_threshold: Minimum street-measurement speed to consider.
        radius: Maximum accepted point-to-point distance in metres.

    Returns:
        Nested dict keyed by vehicle type and then street-row index.
    """
    vehicle_type_dict = {}

    for vehicle_type in ["Car", "Bike", "E-Scooter"]:
        vehicle_type_dict[vehicle_type] = compute_first_sort_dict(df_labels, sort_vehicle_types(df_street, vehicle_type),
                                                                  lon_threshold=lon_threshold, lat_threshold=lat_threshold, 
                                                                  speed_threshold=speed_threshold,
                                                                  radius=radius)
        
    print(vehicle_type_dict["Car"][list(vehicle_type_dict["Car"].keys())[0]].iloc[:30])
    return vehicle_type_dict

def most_frequent(list: list[Any]) -> Any:
    """Return the most common value in a list-like sequence.

    Args:
        list: Sequence of values to count.

    Returns:
        Value with the highest occurrence count.
    """
    return max(set(list), key=list.count)


def create_data_set(
    df_street: pd.DataFrame,
    vehicle_type_dict: dict[str, dict[int, pd.DataFrame]],
    mapping_procedure: str,
    vehicle_type: str = "Car",
) -> list[dict[str, Any]]:
    """Create vibration/label examples using labels near each street point.

    Currently supports `mostfrequent`, which assigns the most common nearby manual
    label to each qualifying street measurement.

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
        if list(labels["label"]):
            if mapping_procedure == "mostfrequent":
                street_row = df_street.loc[i]
                data_set.append({"vibration_x": street_row["vibration_x"],
                                "vibration_y": street_row["vibration_y"],
                                "vibration_z": street_row["vibration_z"],
                                "speed": street_row["speed"],
                                "label": most_frequent(list(labels["label"])),
                                "lon": street_row["lon"],
                                "lat": street_row["lat"],
                                "timestamp": street_row["timestamp"]
                                 }
                )
    print(data_set[:100])
    return data_set