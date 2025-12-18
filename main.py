from utils import utils
from labels_first import labels_first
from street_first import street_first

LABELED_PATH = "./data/molewa_labels.csv"
UNLABELED_PATH = "./data/molewa_street - bearbeitet.csv"

LABELS_FIRST = False

LON_THRESHOLD = 8e-05
LAT_THRESHOLD = 6e-05
SPEED_THRESHOLD = 7

TIME_THRESHOLD = 10

MAPPING_PROCEDURE = "average"

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
                                            mapping_procedure=MAPPING_PROCEDURE, 
                                            vehicle_type=VEHICLE_TYPE)
else:
    vehicle_type_dict = street_first.compute_vehicle_type_dict(df_labels, 
                                                               df_street, 
                                                               lon_threshold=LON_THRESHOLD, 
                                                               lat_threshold=LAT_THRESHOLD, 
                                                               speed_threshold=SPEED_THRESHOLD,
                                                               radius=RADIUS)
    data_set = street_first.create_data_set(df_street, vehicle_type_dict, vehicle_type=VEHICLE_TYPE)