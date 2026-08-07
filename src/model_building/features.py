import pandas as pd

GRAVITY = 9.813
VIBRATION_COLUMNS = ["vibration_x", "vibration_y", "vibration_z"]
SPEED_COLUMN = "speed"


def _calc_vib_magnitude(df: pd.DataFrame) -> pd.Series:
    """Calculate non-negative vibration magnitude after removing gravity."""
    squared_sum = sum(df[col] ** 2 for col in VIBRATION_COLUMNS)
    return (squared_sum ** 0.5 - GRAVITY).clip(lower=0)


def _damage_score(vib_magnitude: pd.Series, speed: pd.Series, k: float) -> pd.Series:
    """Calculate speed-normalized damage score for one scoring exponent."""
    return vib_magnitude.div(speed.pow(k)).where(speed > 0, float("nan"))


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add mild, standard, and strict damage score columns to a DataFrame.

    Required input columns are `vibration_x`, `vibration_y`, `vibration_z`, and
    `speed`. The formulas equals: `max(0, sqrt(x*x + y*y + z*z) - 9.813) / speed**k`.
    """
    required_cols = VIBRATION_COLUMNS + [SPEED_COLUMN]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    scored_df = df.copy()
    vib_magnitude = _calc_vib_magnitude(scored_df)
    speed = scored_df[SPEED_COLUMN]

    scored_df["score_mild"] = _damage_score(vib_magnitude, speed, 0.5)
    scored_df["score_standard"] = _damage_score(vib_magnitude, speed, 1.0)
    scored_df["score_strict"] = _damage_score(vib_magnitude, speed, 2.0)

    return scored_df
