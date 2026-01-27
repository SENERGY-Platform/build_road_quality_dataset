import pandas as pd
from pandas import Series

from open_street_map_data.datasets.mapping_strategy import SURFACE_SCENARIOS, SURFACE_TO_LABEL_1, SURFACE_TO_LABEL_2, \
    SURFACE_TO_LABEL_3


def print_stats(df: pd.DataFrame, scenario:str) -> None:
    """Print basic mapping coverage stats for the `surface` -> numeric label mapping.

    Reports:
    - total missing numeric mappings (`surface_num` is NaN)
    - how many `surface` values are None/NaN
    - which non-null surface strings are unmapped

    Args:
        df: DataFrame expected to contain `surface`, `surface_num` columns.
        scenario: Scenario identifier used in the printed message.
    """
    num_missing = df['surface_num'].isna().sum()
    num_none = df['surface'].isna().sum()
    unmapped_str_mask = df['surface_num'].isna() & df['surface'].notna()
    unmapped_strings = df.loc[unmapped_str_mask, 'surface']
    print(f'Got a total of {num_missing} missing surface mappings scenario {scenario}, with {num_none} Nones and '
          f'{len(unmapped_strings)} of any of these values: \n\t{unmapped_strings.unique()}')

def calc_surface_score(labeled_locations_df:pd.DataFrame, mapping_dict: dict[str, float], scenario:str, stats=False) -> list[Series]:
    """Map `surface` strings to numeric labels for a given mapping scenario.

    Creates a copy of the input DataFrame, adds a `surface_num` column by mapping
    `surface` via `mapping_dict`, and optionally prints coverage stats.

    Args:
        labeled_locations_df: Input DataFrame containing a `surface` column.
        mapping_dict: Dict mapping surface strings to numeric label(s).
        scenario: Scenario identifier.
        stats: If True, print mapping stats.

    Returns:
        Single-element list containing the `surface_num` Series.
    """
    df = labeled_locations_df.copy()
    df['surface_num'] = df['surface'].map(mapping_dict) # nans and not existing keys turn to nan
    if stats:
        print_stats(df, scenario)
    return [df['surface_num']]

def calc_surface_score_c3(labeled_locations_df:pd.DataFrame, mapping_dict: dict[str, tuple[float]], scenario:str, stats=False) -> list[Series]:
    """Map `surface` strings to one or two numeric labels (scenario c3 style).

    Adds `surface_num` from the first tuple entry and `surface_num2` from the second
    tuple entry (if present), otherwise None. Optionally prints coverage stats.

    Args:
        labeled_locations_df: Input DataFrame containing a `surface` column.
        mapping_dict: Dict mapping surface strings to a tuple of numeric labels.
        scenario: Scenario identifier.
        stats: If True, print mapping stats.

    Returns:
        List containing `surface_num` and `surface_num2` Series.
    """
    df = labeled_locations_df.copy()
    df['surface_num'] = df['surface'].apply(lambda x: mapping_dict[x][0] if x in mapping_dict.keys() else None) # nans and not existing keys turn to nan
    df['surface_num2'] = df['surface'].apply(lambda x: mapping_dict[x][1] if x in mapping_dict.keys() and len(mapping_dict[x])>1 else None)  # nans and not existing keys turn to nan
    if stats:
        print_stats(df, scenario)
    return [df['surface_num'], df['surface_num2']]

def calc_scenario(scenario:str, labeled_locations_df:pd.DataFrame) -> list[pd.Series]:
    """Compute surface label column(s) for a named surface scenario."""
    match scenario:
        case 'surf1':
            return calc_surface_score(labeled_locations_df, SURFACE_TO_LABEL_1, scenario, stats=True)
        case 'surf2':
            return calc_surface_score(labeled_locations_df, SURFACE_TO_LABEL_2, scenario, stats=True)
        case 'surf3':
            return calc_surface_score_c3(labeled_locations_df, SURFACE_TO_LABEL_3, scenario, stats=True)
        case _:
            raise ValueError(f'Unknown scenario: {scenario}')

def calc_surface_scenarios(df: pd.DataFrame) -> dict[str, list[pd.Series]]:
    """Compute surface score columns for all configured surface scenarios.

    Args:
        df: Input DataFrame containing a `surface` column.

    Returns:
        Dict mapping scenario id -> list of surface score Series for that scenario.
    """
    surf_score_columns = {}
    for case_id in SURFACE_SCENARIOS:
        score_column = calc_scenario(case_id, df)
        surf_score_columns[case_id] = score_column
    return  surf_score_columns