import data_loader
import ray_mlflow_classifier


def main() -> None:
    """Configure and run the model-building feature dataset pipeline."""
    config = data_loader.DataConfig(
        osm_ds_dir="data/open_street_map/datasets",
        manual_ds_dir="data/molewa/datasets",
        feature_ds_dir="data/molewa/model_building/feature_ds",
        skip_feature_build_if_exists=True,
    )
    feature_datasets = data_loader.load_feature_ds(config)
    dataset = next(iter(feature_datasets["manual"].values()))

    training_result = ray_mlflow_classifier.train(dataset)

    print("Training complete")
    print(training_result)

if __name__ == "__main__":
    main()
