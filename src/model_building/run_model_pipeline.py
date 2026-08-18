import data_loader
from data_test_cases import get_test_dataset_configurations
from experiment_config import ExperimentConfig
from model_config import ANNModelConfig, LinearModelConfig, XGBoostModelConfig

def main() -> None:
    """Configure and run feature loading, test-case creation, splitting, and model setup."""
    data_config = data_loader.DataConfig(
        osm_ds_dir="data/open_street_map/datasets",
        manual_ds_dir="data/molewa/datasets",
        feature_ds_dir="data/molewa/model_building/feature_ds",
        skip_feature_build_if_exists=True,
    )
    feature_datasets = data_loader.load_feature_ds(data_config)

    # testcases:
    #  A: manual only
    #  B: combination
    #  C: only osm (but manual as test set)
    experiment_config = ExperimentConfig(
        experiment_name="Testrun",
        test_cases=['A','B', 'C'],
        case_b_all_osm_data=False, # True uses all available data vs False equals osm train data to available manual data
        case_c_all_osm_data=False,
        ds_version="v1.0",
        features=["vibration_x", "vibration_y", "vibration_z", "speed"], # vibration_magnitude, score_mild, score_standard, score_strict
        models=['ANN','Linear','XGBoost'],
        test_set_percentage=0.3,
        ann_model_config=ANNModelConfig(
            val_set_percentage=0.2,
            layer_num_first_round=10,
            layer_num_second_round=3,
        ),
        linear_model_config=LinearModelConfig(),
        xgb_model_config=XGBoostModelConfig()
    )


    test_cases = get_test_dataset_configurations(feature_datasets, experiment_config.test_cases, experiment_config.ds_version)


if __name__ == "__main__":
    main()
