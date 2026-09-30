# Copyright (c) 2026 InfAI (CC SES)
"""Project-wide settings shared across pipeline stages.

Values defined here must stay consistent between dataset building (manual and
OSM) and model building. Stage-specific configs such as `ManualLabelsConfig`,
`OSMBuildConfig`, and `DataConfig` take their defaults from this module instead
of defining their own copies.
"""

from typing import Final

MIN_SPEED_KMH: Final[float] = 3.0
"""Minimum street-measurement speed in km/h; rows at or below it are dropped.

Below ~3 km/h the car is mostly standing and the vibration signal carries no road
information. Slow driving above it, for example over very bad road sections, is
kept. The same value applies to manual and OSM data so that models trained on one
source and evaluated on the other see the same speed range.
"""

VEHICLE_TYPE: Final[str] = "Car"
"""Vehicle type whose street measurements are used for manual and OSM datasets.

Manual labels were collected from a car, so manual datasets and the manual test
set contain car measurements only. OSM datasets use the same vehicle type so that
OSM training data comes from the same vibration domain as the evaluation data.
"""
