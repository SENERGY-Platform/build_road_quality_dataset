"""Calculate numeric road-quality scores from OSM smoothness tags."""

import pandas as pd
from mapping_strategy import SMOOTHNESS_SCENARIOS, SMOOTHNESS_TO_LABEL_1, SMOOTHNESS_TO_LABEL_2

def print_stats(df: pd.DataFrame, scenario:str) -> None:
    """Print basic mapping coverage stats for the `smoothness` -> numeric label mapping.

    Args:
        df: DataFrame expected to contain `smoothness`, `smoothness_num` columns.
        scenario: Scenario identifier used in the printed message.
    """
    num_missing = df['smoothness_num'].isna().sum()
    num_none = df['smoothness'].isna().sum()
    unmapped_str_mask = df['smoothness_num'].isna() & df['smoothness'].notna()
    unmapped_strings = df.loc[unmapped_str_mask, 'smoothness']
    print(f'Got a total of {num_missing} missing smoothness mappings for scenario {scenario}, with {num_none} Nones and '
          f'{len(unmapped_strings)} of any of these values: \n\t{unmapped_strings.unique()}')

def calc_smoothness_score(labeled_locations_df:pd.DataFrame, mapping_dict: dict[str, float], scenario:str, stats=False) -> pd.Series:
    """Map `smoothness` strings to numeric labels for a given mapping scenario.

    Args:
        labeled_locations_df: Input DataFrame containing a `smoothness` column.
        mapping_dict: Dict mapping smoothness strings to numeric labels.
        scenario: Scenario identifier.
        stats: If True, print mapping stats.

    Returns:
        Series of numeric smoothness scores (`smoothness_num`).
    """
    df = labeled_locations_df.copy()
    df['smoothness_num'] = df['smoothness'].map(mapping_dict) # nans and not existing keys turn to nan
    if stats:
        print_stats(df, scenario)
    return df['smoothness_num']

def calc_smoothness_scenario(scenario:str, labeled_locations_df:pd.DataFrame) -> pd.Series:
    """Compute the smoothness score Series for a named smoothness scenario."""
    match scenario:
        case 'sm1':
            return calc_smoothness_score(labeled_locations_df, SMOOTHNESS_TO_LABEL_1, scenario, stats=True)
        case 'sm2':
            return calc_smoothness_score(labeled_locations_df, SMOOTHNESS_TO_LABEL_2, scenario, stats=True)
        case _:
            raise ValueError(f'Unknown scenario: {scenario}')

def calc_smoothness_scenarios(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute smoothness score Series for all configured smoothness scenarios.

    Args:
        df: Input DataFrame containing a `smoothness` column.

    Returns:
        Dict mapping scenario id -> smoothness score Series.
    """
    sm_score_columns = {}
    for case_id in SMOOTHNESS_SCENARIOS:
        score_column = calc_smoothness_scenario(case_id, df)
        sm_score_columns[case_id] = score_column
    return  sm_score_columns
