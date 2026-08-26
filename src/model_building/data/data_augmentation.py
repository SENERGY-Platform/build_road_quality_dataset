# Label-aware augmentation for three-dimensional vibration measurements.

import numpy as np
import pandas as pd


VIBRATION_COLUMNS = ("vibration_x", "vibration_y", "vibration_z")
VIBRATION_MAGNITUDE_COLUMN = "vibration_magnitude"
SCORE_EXPONENTS = {
    "score_mild": 0.5,
    "score_standard": 1.0,
    "score_strict": 2.0,
}
GRAVITY = 9.813


class VibrationAugmenter:
    # Augment vibration samples and oversample underrepresented labels.
    #
    # The original rows are always retained. balance_strength interpolates
    # each class size towards the size of the largest class. For example, a
    # value of 0.5 closes half of the gap, while 1.0 balances all labels in
    # expectation. The actual number can vary slightly because fractional copy
    # counts are sampled randomly.

    METHODS = ("jitter", "rotation", "reflection")

    def __init__(
        self,
        jitter_std=0.05,
        max_rotation_degrees=15.0,
        balance_strength=0.5,
        method_probabilities=None,
        random_state=None,
    ):
        probabilities = method_probabilities or dict.fromkeys(self.METHODS, 1.0)
        weights = np.asarray([probabilities[name] for name in self.METHODS])

        self.jitter_std = jitter_std
        self.max_rotation_radians = np.deg2rad(max_rotation_degrees)
        self.balance_strength = balance_strength
        self.method_probabilities = weights / weights.sum()
        self.rng = np.random.default_rng(random_state)

    # Add independent Gaussian noise to x, y, and z.
    def jitter(self, values):
        vector = self.as_vector(values)
        return vector + self.rng.normal(0.0, self.jitter_std, size=3)

    # Rotate a vector around a random 3-D axis by a random angle.
    def rotate(self, values):
        vector = self.as_vector(values)
        axis = self.random_unit_vector()
        angle = self.rng.uniform(-self.max_rotation_radians, self.max_rotation_radians)
        # Rodrigues' rotation formula.
        return (
            vector * np.cos(angle)
            + np.cross(axis, vector) * np.sin(angle)
            + axis * np.dot(axis, vector) * (1 - np.cos(angle))
        )

    # Reflect a vector across a random plane through the origin.
    def reflect(self, values):
        vector = self.as_vector(values)
        normal = self.random_unit_vector()
        return vector - 2 * np.dot(vector, normal) * normal

    def augment_dataset(
        self,
        dataset,
        label_column="label",
        shuffle=True,
    ):
        # Return originals plus label-aware, randomly transformed copies.
        if dataset.empty or self.balance_strength == 0:
            return dataset.copy()

        class_counts = dataset[label_column].value_counts()
        largest_class = int(class_counts.max())
        augmented_rows = []

        for _, row in dataset.iterrows():
            label_count = int(class_counts.loc[row[label_column]])
            expected_copies = (
                self.balance_strength * (largest_class - label_count) / label_count
            )
            guaranteed_copies = int(np.floor(expected_copies))
            extra_copy = self.rng.random() < expected_copies - guaranteed_copies

            for _ in range(guaranteed_copies + int(extra_copy)):
                augmented_rows.append(self.augment_row(row))

        if augmented_rows:
            result = pd.concat(
                [dataset.copy(), pd.DataFrame(augmented_rows, columns=dataset.columns)],
                ignore_index=True,
            )
        else:
            result = dataset.copy()

        if shuffle:
            order = self.rng.permutation(len(result))
            result = result.iloc[order]
        return result

    def augment_row(self, row):
        augmented = row.copy()
        values = row.loc[list(VIBRATION_COLUMNS)].to_numpy()
        method_name = self.rng.choice(self.METHODS, p=self.method_probabilities)
        method = {
            "jitter": self.jitter,
            "rotation": self.rotate,
            "reflection": self.reflect,
        }[method_name]
        augmented.loc[list(VIBRATION_COLUMNS)] = method(values)
        self.update_scores(augmented)
        return augmented

    def as_vector(self, values):
        return np.asarray(values, dtype=float)

    def random_unit_vector(self):
        vector = self.rng.normal(size=3)
        return vector / np.linalg.norm(vector)

    # Keep existing derived vibration features consistent after augmentation.
    def update_scores(self, row):
        present_scores = set(SCORE_EXPONENTS).intersection(row.index)
        has_vibration_magnitude = VIBRATION_MAGNITUDE_COLUMN in row.index
        if not present_scores and not has_vibration_magnitude:
            return

        magnitude = max(
            0.0,
            float(np.linalg.norm(row.loc[list(VIBRATION_COLUMNS)].to_numpy()))
            - GRAVITY,
        )
        if has_vibration_magnitude:
            row[VIBRATION_MAGNITUDE_COLUMN] = magnitude

        if not present_scores:
            return

        speed = float(row["speed"])
        for score_column in present_scores:
            exponent = SCORE_EXPONENTS[score_column]
            row[score_column] = magnitude / speed**exponent if speed > 0 else np.nan
