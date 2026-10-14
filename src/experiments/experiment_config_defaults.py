EXPERIMENT_NAME = "road_quality_evaluation_1"
LOG_TO_MLFLOW = True
RUN_ON_RAY = True

OSM_DS_DIR = "data/open_street_map/datasets"
MANUAL_DS_DIR = "data/molewa/datasets"
FEATURE_DS_DIR = "data/molewa/model_building/feature_ds"
SKIP_FEATURE_BUILD_IF_EXISTS = True

DS_VERSION = "v2.0"
FEATURE_SET_NAME = "speed_and_magnitude"
FEATURES = [ "speed", "vibration_magnitude"] # "vibration_x", "vibration_y", "vibration_z", "score_mild", "score_standard", "score_strict"

CROSS_VALIDATION_K = 10
TEST_SET_PERCENTAGE = 0.3

RIDGE_N_PARAMETER_SETS = 30
XGB_N_PARAMETER_SETS = 30
ANN_N_PARAMETER_SETS = 30
