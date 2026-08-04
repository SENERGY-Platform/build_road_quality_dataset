import csv
import os

import pandas as pd

from utils import utils
import labels_first, street_first
import pickle

LABELED_PATH = "data/street_data/labels/molewa_labels.csv"
UNLABELED_PATH = "data/street_data/raw/molewa_street - bearbeitet.csv"
OUTPUT_DIR = "data/street_data/datasets"

LABELS_FIRST = True

LON_THRESHOLD = 8e-05
LAT_THRESHOLD = 6e-05
SPEED_THRESHOLD = 7

TIME_THRESHOLD = 10

LABELS_FIRST_MAPPING_PROCEDURE = "average"
STREET_FIRST_MAPPING_PROCEDURE = "most_frequent"

VEHICLE_TYPE = "Car"

RADIUS = 2

df_labels = utils.load_data(LABELED_PATH)
df_street = utils.load_data(UNLABELED_PATH)

if LABELS_FIRST:
    first_sort_dict = labels_first.compute_first_sort_dict(df_labels, 
                                                           df_street, 
                                                           lon_threshold=LON_THRESHOLD,
                                                           lat_threshold=LAT_THRESHOLD,
                                                           speed_threshold=SPEED_THRESHOLD,
                                                           radius=RADIUS)
    vehicle_type_dict = labels_first.compute_vehicle_type_dict(first_sort_dict, time_threshold=TIME_THRESHOLD)
    data_set = labels_first.create_data_set(df_labels, 
                                            vehicle_type_dict, 
                                            mapping_procedure=LABELS_FIRST_MAPPING_PROCEDURE, 
                                            vehicle_type=VEHICLE_TYPE)
    os.makedirs(f"{OUTPUT_DIR}/labels_first", exist_ok=True)
    file_path = f"{OUTPUT_DIR}/labels_first/{RADIUS}radius_{LABELS_FIRST_MAPPING_PROCEDURE}mappingprocedure_{TIME_THRESHOLD}timethreshold_{VEHICLE_TYPE}vehicletype"
    with open(f"{file_path}.pickle", "wb") as f:
        pickle.dump(data_set, f)
    pd.DataFrame(data_set).to_csv(f"{file_path}.csv", index=False)
else:
    vehicle_type_dict = street_first.compute_vehicle_type_dict(df_labels, 
                                                               df_street, 
                                                               lon_threshold=LON_THRESHOLD, 
                                                               lat_threshold=LAT_THRESHOLD, 
                                                               speed_threshold=SPEED_THRESHOLD,
                                                               radius=RADIUS)
    data_set = street_first.create_data_set(df_street, vehicle_type_dict, mapping_procedure=STREET_FIRST_MAPPING_PROCEDURE, vehicle_type=VEHICLE_TYPE)
    os.makedirs(f"{OUTPUT_DIR}/street_first", exist_ok=True)
    file_path = f"{OUTPUT_DIR}/street_first/{RADIUS}radius_{STREET_FIRST_MAPPING_PROCEDURE}mappingprocedure_{VEHICLE_TYPE}vehicletype"
    with open(f"{file_path}.pickle", "wb") as f:
        pickle.dump(data_set, f)
    pd.DataFrame(data_set).to_csv(f"{file_path}.csv", index=False)
