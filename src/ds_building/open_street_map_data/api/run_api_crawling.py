"""Crawl raw OpenStreetMap data for street-measurement coordinates.

When run as a script, this module reads street CSV coordinates, queries Overpass,
and optionally saves successful raw payloads plus requested-point audit CSVs.
"""

import os
from datetime import datetime
import time
import random

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import pandas as pd

from api_io import read_new_locations, save_raw_data, _log
from api_coms import request_api_labels_multi

def make_batches(seq: List[Tuple[float, float]], size: int) -> Iterable[List[Tuple[float, float]]]:
    """Split a list of (lon, lat) points into consecutive fixed-size batches.

    Args:
        seq: List of (lon, lat) tuples.
        size: Maximum batch size.

    Returns:
        List of batches, each containing up to `size` points, preserving order.
    """
    batches = [seq[i : i + size] for i in range(0, len(seq), size)]
    return batches


def crawl_api_data_points(
    points: Set[Tuple[float, float]],
    save_dir: Optional[str] = None,
    radius_m: int = 25,
    batch_size: int = 250,
    timeout_s: int = 90,
    include_surface_features: bool = False,
    save_raw_json: bool = False,
    max_consecutive_failures: int = 10,
    max_minutes: Optional[float] = None,
) -> pd.DataFrame:
    """Crawl Overpass/OSM payloads for a set of points in batched requests.

    Processes input locations as (lon, lat) tuples. Each batch is sent as a single
    Overpass request via `request_api_labels_multi`. On success, optionally persists
    raw payloads via `save_raw_data`. Implements throttling between successful
    requests and stops after `max_consecutive_failures` or an optional time budget.

    Args:
        points: Set of (lon, lat) points to query.
        save_dir: If set, directory used for optional raw payload persistence.
        radius_m: Search radius (metres) around each point.
        batch_size: Number of points per request.
        timeout_s: Overpass timeout (seconds).
        include_surface_features: If True, expand queries to include explicit surface/
            smoothness features.
        save_raw_json: If True and `save_dir` is set, save successful batch payloads.
        max_consecutive_failures: Stop after this many consecutive request failures.
        max_minutes: Optional overall time limit (minutes), checked between batches.

    Returns:
        DataFrame of crawled rows (currently constructed from `all_rows`).
    """
    now = datetime.now()
    start_ts = datetime.now()
    consecutive_failures = 0
    stop_due_to_failures = False
    points_list = sorted(points)
    if not points_list:
        return pd.DataFrame()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []

    points_df = pd.DataFrame(columns=['lon', 'lat'])

    for batch_idx, batch in enumerate(make_batches(points_list, batch_size), start=1):
        batch = list(batch)
        # Stop cleanly if we exceeded the global time budget (checked between batches)
        if max_minutes is not None:
            elapsed_min = (datetime.now() - start_ts).total_seconds() / 60.0
            if elapsed_min >= max_minutes:
                _log("WARN", f"Stopping crawl: reached max_minutes={max_minutes} (elapsed={elapsed_min:.2f} min) before batch {batch_idx}.")
                break
        if not batch:
            _log("WARN", f"Batch {batch_idx}: empty batch, skipping")
            continue
        _log("INFO", f"Batch {batch_idx}: querying {len(batch)} points")

        try:
            payload = request_api_labels_multi(
                batch,
                radius_m=radius_m,
                timeout_s=timeout_s,
                include_surface_features=include_surface_features,
            )
        except Exception as e:
            # Important: do NOT save requested points for failed batches,
            # so reruns will pick them up again.
            consecutive_failures += 1
            _log("ERROR", f"Batch {batch_idx}: skipped due to Overpass failure (consecutive_failures={consecutive_failures}/{max_consecutive_failures}): {e}")
            if consecutive_failures >= max_consecutive_failures:
                red = "\033[91m"
                reset = "\033[0m"
                print(
                    f"{red}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] "
                    f"Stopping crawl gracefully: Overpass failed {consecutive_failures} times in a row across both servers.{reset}"
                )
                stop_due_to_failures = True
                break

            continue

        if stop_due_to_failures:
            break
        # Successful payload retrieval
        consecutive_failures = 0
        # Only save on SUCCESS
        if save_dir is not None and save_raw_json:
            points_df = save_raw_data(save_dir, points_df, payload, batch, batch_idx, now)
            _log("INFO", f"Batch {batch_idx}: saved payload + requested_points CSV")

        # Stop cleanly if we exceeded the global time budget (checked after completing batch work)
        if max_minutes is not None:
            elapsed_min = (datetime.now() - start_ts).total_seconds() / 60.0
            if elapsed_min >= max_minutes:
                _log("WARN", f"Stopping crawl: reached max_minutes={max_minutes} (elapsed={elapsed_min:.2f} min) after batch {batch_idx}.")
                break

        # Throttle between successful requests to avoid hammering Overpass
        sleep_s = random.uniform(0.5, 1.5)
        _log("DEBUG", f"Batch {batch_idx}: throttling sleep {sleep_s:.2f}s")
        time.sleep(sleep_s)
    return pd.DataFrame(all_rows)


def crawl_api_data(
    dir_path: str,
    save_dir: str,
    api_radius_m: int = 25,
    max_distance_m: float = 2,
    batch_size: int = 5,
    timeout_s: int = 90,
    nrows_per_file: Optional[int] = None,
    round_to: int = 6,
    include_surface_features: bool = False,
    save_raw_json: bool = False,
    max_consecutive_failures: int = 10,
    max_minutes: Optional[float] = None,
) -> pd.DataFrame:
    """Convenience wrapper to crawl only *new* points discovered in a directory.

    Reads unique (lon, lat) points from CSVs in `dir_path`, filters out points that
    already exist under `<save_dir>/requested_points`, then calls
    `crawl_api_data_points` for the remaining points.

    Args:
        dir_path: Directory containing input street CSV files.
        save_dir: Output directory used for deduping and persistence.
        api_radius_m: Overpass query radius (metres).
        max_distance_m: Reserved for compatibility with downstream matching settings.
        batch_size: Number of points per request.
        timeout_s: Overpass timeout (seconds).
        nrows_per_file: Optional per-file read limit.
        round_to: Rounding precision used when deduping points.
        include_surface_features: If True, include explicit surface/smoothness queries.
        save_raw_json: If True, save successful batch payloads.
        max_consecutive_failures: Stop after this many consecutive failures.
        max_minutes: Optional overall time limit (minutes).

    Returns:
        DataFrame of newly crawled label rows.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_subdir = os.path.join(save_dir, "requested_points")
    new_points = read_new_locations(
        dir_path,
        save_subdir,
        n_rows_per_file=nrows_per_file,
    )

    df = crawl_api_data_points(
        new_points,
        save_dir=save_dir,
        radius_m=api_radius_m,
        batch_size=batch_size,
        timeout_s=timeout_s,
        include_surface_features=include_surface_features,
        save_raw_json=save_raw_json,
        max_consecutive_failures=max_consecutive_failures,
        max_minutes=max_minutes,
    )
    return df


if __name__ == "__main__":
    street_raw_path = "data/molewa/raw"
    osm_labels_raw = "data/open_street_map/label_steps/raw_api_data"

    df_labels = crawl_api_data(
        street_raw_path,
        osm_labels_raw,
        nrows_per_file=None,
        api_radius_m=10,
        batch_size=50,
        max_distance_m=2,
        save_raw_json=True,
    )

    if not df_labels.empty:
        out_csv = os.path.join(osm_labels_raw, "osm_labels_latest.csv")
        df_labels.to_csv(out_csv, index=False)
        _log("INFO", f"Wrote {len(df_labels)} rows to {out_csv}")
    else:
        _log("INFO", "No new labels created.")
