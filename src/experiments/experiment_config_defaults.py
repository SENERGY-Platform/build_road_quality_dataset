EXPERIMENT_NAME = "road_quality_test"
LOG_TO_MLFLOW = True
RUN_ON_RAY = True

OSM_DS_DIR = "data/open_street_map/datasets"
MANUAL_DS_DIR = "data/molewa/datasets"
FEATURE_DS_DIR = "data/molewa/model_building/feature_ds"
SKIP_FEATURE_BUILD_IF_EXISTS = True

DS_VERSION = "v1.0"
FEATURE_SET_NAME = "all_features"
FEATURES = ["vibration_x", "vibration_y", "vibration_z", "speed",
            "vibration_magnitude", "score_mild", "score_standard", "score_strict"]

CROSS_VALIDATION_K = 10
TEST_SET_PERCENTAGE = 0.3

RIDGE_N_PARAMETER_SETS = 30
XGB_N_PARAMETER_SETS = 30
ANN_N_PARAMETER_SETS = 30
