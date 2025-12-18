import pandas as pd
import numpy as np
from utils import utils

def compute_first_sort_dict(df_labels, df_street, lon_threshold=8e-05, lat_threshold=6e-05, speed_threshold=7, radius=2):
    first_sort_dict = {}
    for i in range(len(df_labels)):
        first_sort_dict[i] = df_street[(abs(df_street["lon"]-df_labels["lon"][i]) < lon_threshold) &
                                       (abs(df_street["lat"]-df_labels["lat"][i]) < lat_threshold) & 
                                       (df_street["speed"] > speed_threshold)]

        indices_far = []
        for index, row in first_sort_dict[i].iterrows():
            if utils.compute_distance(row["lat"], row["lon"], df_labels["lat"][i], df_labels["lon"][i]).m > radius:
                indices_far.append(index)
        first_sort_dict[i].drop(indices_far)
    print("First Sort Dict created!")
    return first_sort_dict

def compute_vehicle_dict(first_sort_dict, vehicle_type):
    vehicle_dict = {}
    for i in first_sort_dict.keys():
        vehicle_dict[i] = first_sort_dict[i].loc[first_sort_dict[i]["vehicleType"] == vehicle_type]
    print(f"Vehicle Dict for vehicle type {vehicle_type} created!")
    return vehicle_dict

def compute_vehicle_type_dict(first_sort_dict, time_threshold=10):
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


def create_data_set(df_labels, vehicle_type_dict, mapping_procedure="single", vehicle_type="Car"):
    data_set = []
    for i in vehicle_type_dict[vehicle_type].keys():
        if mapping_procedure == "single":
            for _, entry in vehicle_type_dict[vehicle_type][i].iterrows():
                data_set.append({"vibrations": (entry["vibration_x"], entry["vibration_y"], entry["vibration_z"]), 
                                 "label": df_labels["label"].iloc[i]})
        elif mapping_procedure == "average":
            data_set.append({"vibrations": (np.mean([entry["vibration_x"] for _, entry in vehicle_type_dict[vehicle_type][i].iterrows()]),
                                            np.mean([entry["vibration_y"] for _, entry in vehicle_type_dict[vehicle_type][i].iterrows()]),
                                            np.mean([entry["vibration_z"] for _, entry in vehicle_type_dict[vehicle_type][i].iterrows()])),
                            "label": df_labels["label"].iloc[i]})
            
    print("Data set created!")
    print(data_set[:300])       
    return data_set