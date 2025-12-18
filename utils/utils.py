import pandas as pd
from geopy import distance

def load_data(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"],format="ISO8601")
    return df


def compute_distance(lat_1, lon_1, lat_2, lon_2):
    return distance.distance((lat_1, lon_1),(lat_2, lon_2))