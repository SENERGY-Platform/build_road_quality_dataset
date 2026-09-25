"""Run the complete OpenStreetMap dataset-building pipeline.

This entrypoint runs the three OSM stages in order:

1. Crawl Overpass payloads for new street-measurement coordinates.
2. Convert raw Overpass payloads to nearest-road location labels.
3. Map OSM tags to numeric scenarios and join labels to street measurements.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.ds_building.open_street_map_data.api.api_io import log_msg
from src.ds_building.open_street_map_data.api.run_api_crawling import crawl_api_data
from src.ds_building.open_street_map_data.api.run_location_label_mapping import build_location_labels
from src.ds_building.open_street_map_data.datasets.run_osm_ds_build import (
    DEFAULT_MIN_SPEED_THRESHOLD,
    build_mapped_labels,
    join_labels_to_streets,
)


@dataclass(frozen=True)
class OSMBuildConfig:
    """Configuration for the full OSM build."""

    street_raw_dir: str = "data/molewa/raw"
    raw_api_data_dir: str = "data/open_street_map/label_steps/raw_api_data"
    labeled_locations_file: str = "data/open_street_map/label_steps/labeled_location_data/labeled_locations.parquet"
    mapped_labels_dir: str = "data/open_street_map/label_steps/mapped_labels"
    osm_dataset_dir: str = "data/open_street_map/datasets"

    api_radius_m: int = 10
    batch_size: int = 50
    timeout_s: int = 90
    max_distance_m: float = 2
    nrows_per_file: int | None = None
    include_surface_features: bool = False
    save_raw_json: bool = True
    max_consecutive_failures: int = 10
    max_minutes: float | None = None

    num_payload_files: int | None = None
    min_speed_threshold: float = DEFAULT_MIN_SPEED_THRESHOLD


def run_pipeline(config: OSMBuildConfig) -> None:
    """Run all OSM build stages in dependency order."""
    log_msg("INFO", "OSM pipeline stage 1/3: crawling new Overpass payloads.")
    crawl_api_data(
        dir_path=config.street_raw_dir,
        save_dir=config.raw_api_data_dir,
        api_radius_m=config.api_radius_m,
        max_distance_m=config.max_distance_m,
        batch_size=config.batch_size,
        timeout_s=config.timeout_s,
        nrows_per_file=config.nrows_per_file,
        include_surface_features=config.include_surface_features,
        save_raw_json=config.save_raw_json,
        max_consecutive_failures=config.max_consecutive_failures,
        max_minutes=config.max_minutes,
    )

    log_msg("INFO", "OSM pipeline stage 2/3: building nearest-road location labels.")
    build_location_labels(
        payload_dir=str(Path(config.raw_api_data_dir) / "payloads"),
        save_file=config.labeled_locations_file,
        duplicates_doc_file=str(Path(config.labeled_locations_file).with_name("duplicates.csv")),
        num_files=config.num_payload_files,
    )

    log_msg("INFO", "OSM pipeline stage 3/3: building mapped labels and joined street datasets.")
    build_mapped_labels(config.labeled_locations_file, config.mapped_labels_dir)
    join_labels_to_streets(
        config.mapped_labels_dir,
        config.street_raw_dir,
        config.osm_dataset_dir,
        min_speed_threshold=config.min_speed_threshold,
    )
    log_msg("INFO", "OSM pipeline finished.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete OSM dataset-building pipeline.",
    )
    parser.add_argument("--street-raw-dir", default=OSMBuildConfig.street_raw_dir)
    parser.add_argument("--raw-api-data-dir", default=OSMBuildConfig.raw_api_data_dir)
    parser.add_argument("--labeled-locations-file", default=OSMBuildConfig.labeled_locations_file)
    parser.add_argument("--mapped-labels-dir", default=OSMBuildConfig.mapped_labels_dir)
    parser.add_argument("--osm-dataset-dir", default=OSMBuildConfig.osm_dataset_dir)
    parser.add_argument("--api-radius-m", type=int, default=OSMBuildConfig.api_radius_m)
    parser.add_argument("--batch-size", type=int, default=OSMBuildConfig.batch_size)
    parser.add_argument("--timeout-s", type=int, default=OSMBuildConfig.timeout_s)
    parser.add_argument("--max-distance-m", type=float, default=OSMBuildConfig.max_distance_m)
    parser.add_argument("--nrows-per-file", type=int, default=OSMBuildConfig.nrows_per_file)
    parser.add_argument("--include-surface-features", action="store_true")
    parser.add_argument(
        "--no-save-raw-json",
        action="store_false",
        dest="save_raw_json",
        help="Do not persist successful raw Overpass payloads.",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=OSMBuildConfig.max_consecutive_failures,
    )
    parser.add_argument("--max-minutes", type=float, default=OSMBuildConfig.max_minutes)
    parser.add_argument(
        "--num-payload-files",
        type=int,
        default=OSMBuildConfig.num_payload_files,
        help="Optional limit for stage 2 payload JSON files, mainly for debugging.",
    )
    parser.add_argument(
        "--min-speed-threshold",
        type=float,
        default=OSMBuildConfig.min_speed_threshold,
    )
    return parser.parse_args()


def main() -> None:
    """Parse CLI arguments and run the complete OSM build."""
    args = _parse_args()
    config = OSMBuildConfig(
        street_raw_dir=args.street_raw_dir,
        raw_api_data_dir=args.raw_api_data_dir,
        labeled_locations_file=args.labeled_locations_file,
        mapped_labels_dir=args.mapped_labels_dir,
        osm_dataset_dir=args.osm_dataset_dir,
        api_radius_m=args.api_radius_m,
        batch_size=args.batch_size,
        timeout_s=args.timeout_s,
        max_distance_m=args.max_distance_m,
        nrows_per_file=args.nrows_per_file,
        include_surface_features=args.include_surface_features,
        save_raw_json=args.save_raw_json,
        max_consecutive_failures=args.max_consecutive_failures,
        max_minutes=args.max_minutes,
        num_payload_files=args.num_payload_files,
        min_speed_threshold=args.min_speed_threshold,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
