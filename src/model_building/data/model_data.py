from dataclasses import dataclass, field

import pandas as pd

DEFAULT_SHUFFLE_RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelData:
    """Container for train, test, and optional validation splits used by models."""
    test_case_id: str
    random_state: int

    manual_train_x: pd.DataFrame
    manual_train_y: pd.Series
    osm_train_x: pd.DataFrame
    osm_train_y: pd.Series

    # only manual data anyway
    test_x: pd.DataFrame
    test_y: pd.Series

    # optional for ann
    manual_val_x: pd.DataFrame = field(default_factory=pd.DataFrame)
    manual_val_y: pd.Series = field(default_factory=pd.Series)
    osm_val_x: pd.DataFrame = field(default_factory=pd.DataFrame)
    osm_val_y: pd.Series = field(default_factory=pd.Series)

    @staticmethod
    def _combine_and_shuffle_xy(
            x_parts: list[pd.DataFrame],
            y_parts: list[pd.Series],
            random_state: int = DEFAULT_SHUFFLE_RANDOM_STATE,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Combine matching feature/label parts and optionally shuffle them together."""
        pairs = [
            (x_part, y_part)
            for x_part, y_part in zip(x_parts, y_parts)
            if not x_part.empty and not y_part.empty
        ]
        if not pairs:
            return pd.DataFrame(), pd.Series()

        x = pd.concat([pair[0] for pair in pairs], ignore_index=True)
        y = pd.concat([pair[1] for pair in pairs], ignore_index=True)

        if len(y) <= 1:
            return x, y

        order = y.sample(frac=1, random_state=random_state).index
        return x.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)

    def get_train_x(self) -> pd.DataFrame:
        """Return the combined manual and OSM training feature matrix."""
        train_x, _ = self._combine_and_shuffle_xy(
            [self.manual_train_x, self.osm_train_x],
            [self.manual_train_y, self.osm_train_y],
        )
        return train_x

    def get_train_y(self) -> pd.Series:
        """Return the combined manual and OSM training label series."""
        _, train_y = self._combine_and_shuffle_xy(
            [self.manual_train_x, self.osm_train_x],
            [self.manual_train_y, self.osm_train_y],
        )
        return train_y

    def get_val_x(self) -> pd.DataFrame:
        """Return the combined manual and OSM validation feature matrix."""
        val_x, _ = self._combine_and_shuffle_xy(
            [self.manual_val_x, self.osm_val_x],
            [self.manual_val_y, self.osm_val_y],
        )
        return val_x

    def get_val_y(self) -> pd.Series:
        """Return the combined manual and OSM validation label series."""
        _, val_y = self._combine_and_shuffle_xy(
            [self.manual_val_x, self.osm_val_x],
            [self.manual_val_y, self.osm_val_y],
        )
        return val_y
