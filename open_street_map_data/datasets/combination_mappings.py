import os
from typing import Callable, final
import pandas as pd

from open_street_map_data.datasets.mapping_strategy import COMBINATION_SCENARIOS
ChooseFn = Callable[[pd.Series, pd.Series], pd.Series]


def calc_prio_case(
    prio: str,
    labeled_locations_df: pd.DataFrame,
    sm_score_column: pd.Series,
    surf_score_columns: list[pd.Series],
) -> pd.DataFrame:
    """Combine smoothness and surface scores by prioritising one over the other.

    If `prio` is "smoothness", take smoothness where available, otherwise surface.
    If `prio` is "surface", do the reverse.

    Args:
        prio: Which score to prioritise ("smoothness" or "surface").
        labeled_locations_df: Base labelled locations DataFrame.
        sm_score_column: Smoothness numeric score Series.
        surf_score_columns: One or more surface numeric score Series.

    Returns:
        Concatenated DataFrame containing one row per input row per surface column.
    """
    def choose(sm_score: pd.Series, surf_score: pd.Series) -> pd.Series:
        """Return the prioritised label Series, falling back to the other when missing."""
        prio_num = sm_score if prio == "smoothness" else surf_score
        second_num = surf_score if prio == "smoothness" else sm_score
        return prio_num.fillna(second_num)

    rows = calc_row_batches(choose, labeled_locations_df, sm_score_column, surf_score_columns)
    return pd.concat(rows, ignore_index=True, axis=0)


def calc_avg_case(
    labeled_locations_df: pd.DataFrame,
    sm_score_column: pd.Series,
    surf_score_columns: list[pd.Series],
) -> pd.DataFrame:
    """Combine smoothness and surface scores by averaging when both are present.

    If both scores exist, uses their arithmetic mean. If only one exists, uses the
    available score.

    Args:
        labeled_locations_df: Base labelled locations DataFrame.
        sm_score_column: Smoothness numeric score Series.
        surf_score_columns: One or more surface numeric score Series.

    Returns:
        Concatenated DataFrame containing one row per input row per surface column.
    """
    def choose(sm_score: pd.Series, surf_score: pd.Series) -> pd.Series:
        """Return averaged labels when possible, otherwise the available single score."""
        avg_score = (sm_score + surf_score) / 2
        both_present = sm_score.notna() & surf_score.notna()

        return avg_score.where(both_present, sm_score.combine_first(surf_score))

    rows = calc_row_batches(choose, labeled_locations_df, sm_score_column, surf_score_columns)
    return pd.concat(rows, ignore_index=True, axis=0)


def calc_dupl_add_case(
    labeled_locations_df: pd.DataFrame,
    sm_score_column: pd.Series,
    surf_score_columns: list[pd.Series],
) -> pd.DataFrame:
    """Return the union of both priority cases (smoothness-first and surface-first)."""
    df1 = calc_prio_case("smoothness", labeled_locations_df, sm_score_column, surf_score_columns)
    df2 = calc_prio_case("surface", labeled_locations_df, sm_score_column, surf_score_columns)
    return pd.concat([df1, df2], ignore_index=True, axis=0)


def calc_row_batches(
    choose: ChooseFn,
    labeled_locations_df: pd.DataFrame,
    sm_score_column: pd.Series,
    surf_score_columns: list[pd.Series],
) -> list[pd.DataFrame]:
    """Build per-surface-column DataFrames with combined label outputs.

    Creates a base frame (lat/lon/smoothness/surface) and, for each surface score
    Series, adds smoothness/surface score columns plus a combined `label`.

    Returns:
        List of DataFrames, one per provided surface score Series.
    """
    rows: list[pd.DataFrame] = []

    base = labeled_locations_df[["lat", "lon", "smoothness", "surface"]].copy()

    for surf in surf_score_columns:
        df_batch = base.copy()
        df_batch["smoothness_score"] = sm_score_column.to_numpy()
        df_batch["surface_score"] = surf.to_numpy()
        df_batch["label"] = choose(sm_score_column, surf).to_numpy()
        df_batch["old_idx"] = labeled_locations_df.index.to_numpy()
        rows.append(df_batch)

    return rows


def add_scenario_description_columns(df: pd.DataFrame, sm_scenario: str, surf_scenario: str, comb_scenario: str) -> pd.DataFrame:
    """Add scenario id columns (sm/surf/comb) to a result DataFrame."""
    df = df.copy()
    df["sm_scenario"] = sm_scenario
    df["surf_scenario"] = surf_scenario
    df["comb_scenario"] = comb_scenario
    return df


def calc_scenario(
    comb_scenario: str,
    labeled_locations_df: pd.DataFrame,
    sm_scenario: str,
    surf_scenario: str,
    sm_score_column: pd.Series,
    surf_score_columns: list[pd.Series],
) -> pd.DataFrame:
    """Calculate one combination scenario DataFrame and de-duplicate rows.

    Runs the requested combination strategy (c1..c4), sorts, drops duplicates, and
    adds scenario description columns.

    Returns:
        DataFrame with combined `label` and scenario metadata columns.
    """

    print(f'CALCULATING COMBINATION SCENARIO: {comb_scenario} | {surf_scenario} | {sm_scenario}')
    df = pd.DataFrame()
    match comb_scenario:
        case "c1":
            df = calc_prio_case("smoothness", labeled_locations_df, sm_score_column, surf_score_columns)
        case "c2":
            df = calc_prio_case("surface", labeled_locations_df, sm_score_column, surf_score_columns)
        case "c3":
            df = calc_avg_case(labeled_locations_df, sm_score_column, surf_score_columns)
        case "c4":
            df = calc_dupl_add_case(labeled_locations_df, sm_score_column, surf_score_columns)
        case _:
            raise ValueError(f"Unknown scenario: {comb_scenario}")

    df.sort_values(by=['surface', "old_idx"], inplace=True)
    df.drop_duplicates(subset=["lat", "lon", "smoothness", "surface", "label"], keep="first", inplace=True)
    return add_scenario_description_columns(df, sm_scenario, surf_scenario, comb_scenario)


def calc_save_combination_scenarios(
    df: pd.DataFrame,
    sm_scenario: str,
    surf_scenario: str,
    sm_score_column: pd.Series,
    surf_score_columns: list[pd.Series],
    save_dir: str,
) -> None:
    """Compute all combination scenarios and save each as a CSV under `save_dir`."""
    os.makedirs(save_dir, exist_ok=True)

    for case_id in COMBINATION_SCENARIOS:
        case_df = calc_scenario(case_id, df, sm_scenario, surf_scenario, sm_score_column, surf_score_columns)
        file_path = os.path.join(save_dir, f"osm_labels_{sm_scenario}_{surf_scenario}_{case_id}.csv")
        case_df.to_csv(file_path, index=False)