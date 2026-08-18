
from dataclasses import dataclass

import pandas as pd


@dataclass
class ModelData:
    """Container for train, test, and optional validation splits used by models."""

    manual_train_x: pd.DataFrame | None
    manual_train_y: pd.Series | None
    osm_train_x: pd.DataFrame | None
    osm_train_y: pd.Series | None

    test_x: pd.DataFrame
    test_y: pd.Series

    # optional for ann
    val_x: pd.DataFrame | None = None
    val_y: pd.Series | None = None

    def get_train_x(self) -> pd.DataFrame:
        """Return the combined manual and OSM training feature matrix."""
        return pd.concat([self.manual_train_x, self.osm_train_x], ignore_index=True)

    def get_train_y(self) -> pd.Series:
        """Return the combined manual and OSM training label series."""
        return pd.concat([self.manual_train_y, self.osm_train_y], ignore_index=True)
