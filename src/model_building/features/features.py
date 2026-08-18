import pandas as pd

GRAVITY = 9.813
VIBRATION_COLUMNS = ["vibration_x", "vibration_y", "vibration_z"]
SPEED_COLUMN = "speed"

def _calc_vib_magnitude(df: pd.DataFrame) -> pd.Series:
    """Calculate non-negative vibration magnitude after subtracting gravity."""
    squared_sum = sum(df[col] ** 2 for col in VIBRATION_COLUMNS)
    return (squared_sum ** 0.5 - GRAVITY).clip(lower=0)


def _damage_score(vib_magnitude: pd.Series, speed: pd.Series, k: float) -> pd.Series:
    """Calculate a speed-normalized damage score for one exponent value."""
    return vib_magnitude.div(speed.pow(k)).where(speed > 0, float("nan"))


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add mild, standard, and strict damage score columns to a DataFrame.

    Required input columns are `vibration_x`, `vibration_y`, `vibration_z`, and
    `speed`. The formula is `max(0, sqrt(x*x + y*y + z*z) - 9.813) / speed**k`.
    """
    required_cols = VIBRATION_COLUMNS + [SPEED_COLUMN]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    scored_df = df.copy()
    vib_magnitude = _calc_vib_magnitude(scored_df)
    speed = scored_df[SPEED_COLUMN]

    scored_df["vibration_magnitude"] = vib_magnitude
    scored_df["score_mild"] = _damage_score(vib_magnitude, speed, 0.5)
    scored_df["score_standard"] = _damage_score(vib_magnitude, speed, 1.0)
    scored_df["score_strict"] = _damage_score(vib_magnitude, speed, 2.0)

    return scored_df

def label_category_from_continuous(label_col: pd.Series) -> pd.Series:
    """Map continuous numeric road-quality labels to good, medium, and bad classes."""
    def _classify(x:float):
        """Classify one numeric label value into a road-quality class."""
        if x <= 0.33:
            return 'good'
        elif 0.33 < x <= 1.33:
            return 'medium'
        elif 1.33 < x:
            return 'bad'
        else: raise ValueError(f"Invalid continous value for label found: {x}")

    return label_col.map(_classify)
