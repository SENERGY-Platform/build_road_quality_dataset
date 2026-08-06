"""Manual-label-first matching for road-quality dataset construction.

This module starts from each manually labeled road-quality point, finds nearby
street sensor measurements, filters them by vehicle type and recency, and creates
training examples with vibration values paired to the manual label.
"""

import pandas as pd
import numpy as np
from typing import Any

import utils

def compute_first_sort_dict(
    df_labels: pd.DataFrame,
    df_street: pd.DataFrame,
    lon_threshold: float = 8e-05,
    lat_threshold: float = 6e-05,
    speed_threshold: float = 7,
    radius: float = 2,
) -> dict[int, pd.DataFrame]:
    """Find nearby street rows for each manual label point.

    Applies a coarse longitude/latitude bounding-box filter, requires street rows
    to exceed `speed_threshold`, and then removes candidates farther than `radius`
    metres by geodesic distance.

    Args:
        df_labels: DataFrame containing manual label points with `lat` and `lon`.
        df_street: DataFrame containing street measurements with coordinates and speed.
        lon_threshold: Maximum absolute longitude difference for the coarse filter.
        lat_threshold: Maximum absolute latitude difference for the coarse filter.
        speed_threshold: Minimum speed required for a street row to be considered.
        radius: Maximum accepted point-to-point distance in metres.

    Returns:
        Dict mapping each label-row index to a DataFrame of nearby street rows.
    """
    first_sort_dict = {}
    for i in range(len(df_labels)):
        first_sort_dict[i] = df_street[(abs(df_street["lon"]-df_labels["lon"][i]) < lon_threshold) &
                                       (abs(df_street["lat"]-df_labels["lat"][i]) < lat_threshold) & 
                                       (df_street["speed"] > speed_threshold)]

        indices_far = []
        for index, row in first_sort_dict[i].iterrows():
            if utils.compute_distance(row["lat"], row["lon"], df_labels["lat"][i], df_labels["lon"][i]).m > radius:
                indices_far.append(index)
        first_sort_dict[i] = first_sort_dict[i].drop(indices_far)
    print("First Sort Dict created!")
    return first_sort_dict

def compute_vehicle_dict(first_sort_dict: dict[int, pd.DataFrame], vehicle_type: str) -> dict[int, pd.DataFrame]:
    """Filter each label's candidate street rows to a single vehicle type.

    Args:
        first_sort_dict: Mapping from label index to nearby street measurement rows.
        vehicle_type: Vehicle type string to keep, such as `Car`, `Bike`, or
            `E-Scooter`.

    Returns:
        Dict with the same keys as `first_sort_dict`, where each value contains
        only rows matching `vehicle_type`.
    """
    vehicle_dict = {}
    for i in first_sort_dict.keys():
        vehicle_dict[i] = first_sort_dict[i].loc[first_sort_dict[i]["vehicleType"] == vehicle_type]
    print(f"Vehicle Dict for vehicle type {vehicle_type} created!")
    return vehicle_dict

def compute_vehicle_type_dict(
    first_sort_dict: dict[int, pd.DataFrame],
    time_threshold: int = 10,
) -> dict[str, dict[int, pd.DataFrame]]:
    """Build candidate street-row mappings for all supported vehicle types.

    Candidate rows are grouped by vehicle type and filtered to keep only rows whose
    timestamp falls within `time_threshold` days of the newest candidate timestamp
    for the same label point.

    Args:
        first_sort_dict: Mapping from label index to nearby street measurement rows.
        time_threshold: Number of days before the newest candidate timestamp to keep.

    Returns:
        Nested dict keyed by vehicle type and then label-row index.
    """
    vehicle_type_dict = {}
    vehicle_type_dict_aux = {}

    vehicle_type_dict_aux["Car"] = compute_vehicle_dict(first_sort_dict, "Car")
    vehicle_type_dict_aux["Bike"] = compute_vehicle_dict(first_sort_dict, "Bike")
    vehicle_type_dict_aux["E-Scooter"] = compute_vehicle_dict(first_sort_dict, "E-Scooter")

    for vehicle_type in vehicle_type_dict_aux.keys():
        vehicle_type_dict[vehicle_type] = {}
        for i in vehicle_type_dict_aux[vehicle_type].keys():
            if list(vehicle_type_dict_aux[vehicle_type][i]["timestamp"]):
                vehicle_type_dict[vehicle_type][i] = vehicle_type_dict_aux[vehicle_type][i].loc[max(vehicle_type_dict_aux[vehicle_type][i]["timestamp"]) - 
                                                                                                    vehicle_type_dict_aux[vehicle_type][i]["timestamp"] < pd.Timedelta(time_threshold,"d")]
    print("Vehicle type dict created!") 

    return vehicle_type_dict


def create_data_set(
    df_labels: pd.DataFrame,
    vehicle_type_dict: dict[str, dict[int, pd.DataFrame]],
    mapping_procedure: str = "single",
    vehicle_type: str = "Car",
) -> list[dict[str, Any]]:
    """Create labeled street-measurement examples from manually labeled points.

    Args:
        df_labels: DataFrame of manual labels containing a `label` column.
        vehicle_type_dict: Nested vehicle type mapping returned by
            `compute_vehicle_type_dict`.
        mapping_procedure: `single` to create one example per street row, or
            `average` to average vibration, speed, location, and timestamp values
            for each label point.
        vehicle_type: Vehicle type to extract from `vehicle_type_dict`.

    Returns:
        List of dicts with flat `vibration_x`, `vibration_y`, `vibration_z`,
        `speed`, `label`, `lon`, `lat`, and `timestamp` fields.

    Raises:
        ValueError: If `mapping_procedure` is not `single` or `average`.
    """
    if mapping_procedure not in {"single", "average"}:
        raise ValueError(
            f"Unsupported labels_first mapping_procedure '{mapping_procedure}'. "
            "Use 'single' or 'average'."
        )

    data_set = []
    for i in vehicle_type_dict[vehicle_type].keys():
        street_rows = vehicle_type_dict[vehicle_type][i]
        if mapping_procedure == "single":
            for _, entry in street_rows.iterrows():
                data_set.append({"vibration_x": entry["vibration_x"],
                                 "vibration_y": entry["vibration_y"],
                                 "vibration_z": entry["vibration_z"],
                                 "speed": entry["speed"],
                                 "label": df_labels["label"].iloc[i],
                                 "lon": entry["lon"],
                                 "lat": entry["lat"],
                                 "timestamp": entry["timestamp"],})
        elif mapping_procedure == "average":
            data_set.append({"vibration_x": street_rows["vibration_x"].mean(),
                            "vibration_y": street_rows["vibration_y"].mean(),
                            "vibration_z": street_rows["vibration_z"].mean(),
                            "speed": street_rows["speed"].mean(),
                            "label": df_labels["label"].iloc[i],
                            "lon": street_rows["lon"].mean(),
                            "lat": street_rows["lat"].mean(),
                            "timestamp": street_rows["timestamp"].mean(),}
            )
            
    print("Data set created!")
    print(data_set[:300])       
    return data_set
