from pathlib import Path

import pandas as pd

from src.model_building.data.data_augmentation import VibrationAugmenter
from src.model_building.features.features import label_category_from_continuous


BALANCE_COLUMN = "augmentation_label_category"
AUGMENTATION_SEEDS = {
    "average": 548613720,
    "mostfrequent": 1425025783,
    "single": 3185540658,
}

DATASET_DIRS = [
    (
        Path("data/molewa/datasets/labels_first"),
        Path("data/molewa/datasets_augmented/labels_first"),
    ),
    (
        Path("data/molewa/datasets/street_first"),
        Path("data/molewa/datasets_augmented/street_first"),
    ),
    (
        Path("data/molewa/model_building/feature_ds/manual"),
        Path("data/molewa/model_building/feature_ds_augmented/manual"),
    ),
]


def get_label_categories(labels):
    # Feature labels are numeric.
    if pd.api.types.is_numeric_dtype(labels):
        return label_category_from_continuous(labels)
    return labels.str.strip().str.lower()


def get_seed(file_path):
    mapping_procedure = file_path.stem.split("mappingprocedure")[1].split("_")[0]
    return AUGMENTATION_SEEDS[mapping_procedure]


def augment_file(source_path, output_dir):
    dataset = pd.read_parquet(source_path)
    dataset[BALANCE_COLUMN] = get_label_categories(dataset["label"])

    augmenter = VibrationAugmenter(random_state=get_seed(source_path))
    augmented = augmenter.augment_dataset(
        dataset,
        label_column=BALANCE_COLUMN,
    )
    augmented = augmented.drop(columns=BALANCE_COLUMN)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / source_path.name
    augmented.to_parquet(output_path, index=False)

    print(f"{source_path.name}: {len(dataset):,} -> {len(augmented):,} rows")
    return len(dataset), len(augmented)


def main():
    source_rows = 0
    augmented_rows = 0

    for source_dir, output_dir in DATASET_DIRS:
        for source_path in sorted(source_dir.glob("*.parquet")):
            source_count, augmented_count = augment_file(source_path, output_dir)
            source_rows += source_count
            augmented_rows += augmented_count

    print(f"Total: {source_rows:,} -> {augmented_rows:,} rows")


if __name__ == "__main__":
    main()
