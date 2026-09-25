"""Top-level entrypoint for dataset-building pipelines."""

from __future__ import annotations

import argparse

from src.ds_building.manual_labels.run_manual_ds_build import (
    run_default_pipelines as run_manual_pipeline,
)
from src.ds_building.open_street_map_data.run_osm_pipeline import (
    OSMBuildConfig,
    run_pipeline as run_osm_pipeline,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build road-quality datasets from manual labels and OpenStreetMap.",
    )
    pipeline_group = parser.add_mutually_exclusive_group()
    pipeline_group.add_argument("--manual", action="store_true", help="Run only the manual-label pipeline.")
    pipeline_group.add_argument("--osm", action="store_true", help="Run only the OpenStreetMap pipeline.")
    pipeline_group.add_argument("--all", action="store_true", help="Run manual labels first, then OSM.")
    return parser.parse_args()


def main() -> None:
    """Run selected dataset-building pipelines."""
    args = _parse_args()
    run_manual = args.manual or args.all or not args.osm
    run_osm = args.osm or args.all or not args.manual

    if run_manual:
        run_manual_pipeline()
    if run_osm:
        run_osm_pipeline(OSMBuildConfig())


if __name__ == "__main__":
    main()
