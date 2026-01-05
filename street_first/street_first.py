from tqdm import tqdm
from utils import utils

def sort_vehicle_types(df_street, vehicle_type):
    return df_street.loc[df_street["vehicleType"] == vehicle_type]

def compute_first_sort_dict(df_labels, df_vehicle_street, lon_threshold=3e-05, lat_threshold=2e-05, speed_threshold=7, radius=2):
    first_sort_dict = {}

    for i in tqdm(df_vehicle_street.index):
        if df_vehicle_street["speed"][i] > speed_threshold:
            first_sort_dict[i] = df_labels[(abs(df_vehicle_street["lon"][i]-df_labels["lon"]) < lon_threshold) &
                                           (abs(df_vehicle_street["lat"][i]-df_labels["lat"]) < lat_threshold)]
            indices_far = []
            for index, row in first_sort_dict[i].iterrows():
                if utils.compute_distance(row["lat"], row["lon"], df_vehicle_street["lat"][i], df_vehicle_street["lon"][i]).m > radius:
                    indices_far.append(index)
            first_sort_dict[i].drop(indices_far)
            
    return first_sort_dict


def compute_vehicle_type_dict(df_labels, df_street, lon_threshold=3e-05, lat_threshold=2e-05, speed_threshold=7, radius=2):
    vehicle_type_dict = {}

    for vehicle_type in ["Car", "Bike", "E-Scooter"]:
        vehicle_type_dict[vehicle_type] = compute_first_sort_dict(df_labels, sort_vehicle_types(df_street, vehicle_type),
                                                                  lon_threshold=lon_threshold, lat_threshold=lat_threshold, 
                                                                  speed_threshold=speed_threshold,
                                                                  radius=radius)
        
    print(vehicle_type_dict["Car"][list(vehicle_type_dict["Car"].keys())[0]].iloc[:30])
    return vehicle_type_dict

def most_frequent(list):
    return max(set(list), key=list.count)


def create_data_set(df_street, vehicle_type_dict, mapping_procedure, vehicle_type="Car"):
    data_set = []
    for i, labels in vehicle_type_dict[vehicle_type].items():
        if list(labels["label"]):
            if mapping_procedure == "most_frequent":
                data_set.append({"vibrations": (df_street["vibration_x"].iloc[i], 
                                        df_street["vibration_y"].iloc[i],
                                        df_street["vibration_z"].iloc[i]),
                             "label": most_frequent(list(labels["label"]))})
    print(data_set[:100])
    return data_set

#TODO: Implement a reasonable condition on a maximal time gap between vibration-label pairs in the data set. (as is present in the labels first method)