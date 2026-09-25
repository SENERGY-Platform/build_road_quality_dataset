# OpenStreetMap Data Mini-Pipeline

This pipeline builds road-quality labels from OpenStreetMap tags and joins the
resulting labels back onto street sensor measurements.

The pipeline has three stages:

1. Crawl Overpass API data for street-measurement locations.
2. Convert raw Overpass payloads into nearest-road location labels.
3. Map OSM `smoothness` and `surface` tags into numeric label scenarios and join
   them to street measurements.

## Quick Start

Run the full OSM build from the repository root:

```bash
uv run python -m src.ds_building.open_street_map_data.run_osm_pipeline
```

If your environment is already active and has the project dependencies installed,
`python` can be used instead of `uv run python`.

The full entrypoint runs all three stages in order:

1. Reads all street CSVs in `data/molewa/raw/` and crawls only coordinates that
   are not already listed under
   `data/open_street_map/label_steps/raw_api_data/requested_points/`.
2. Rebuilds `labeled_locations.parquet` from all saved Overpass payload JSON
   files.
3. Rebuilds all mapped OSM label scenarios and joined training datasets.

Common options:

```bash
uv run python -m src.ds_building.open_street_map_data.run_osm_pipeline \
  --street-raw-dir data/molewa/raw \
  --api-radius-m 10 \
  --batch-size 50 \
  --max-minutes 120
```

Use `--help` to see all path and crawl settings.

## Inputs

- Street measurements: `data/molewa/raw/`
- Raw Overpass payloads: `data/open_street_map/label_steps/raw_api_data/payloads/`
- Labelled OSM locations: `data/open_street_map/label_steps/labeled_location_data/labeled_locations.parquet`

The street CSVs must contain `lat` and `lon` columns. Dataset joining also expects
the sensor-measurement columns that should be retained in the final training data.

Every `*.csv` file in `data/molewa/raw/` is treated as input. Add new street files
there, or replace the old file with a combined file, but avoid keeping both an old
file and a new combined file in the folder because that duplicates rows in the
joined datasets.

The manual-label CSV in `data/molewa/labels/` is not used by this OSM pipeline.

## Stage 1: Crawl OSM Data

Entrypoint:

```bash
uv run python -m src.ds_building.open_street_map_data.api.run_api_crawling
```

This reads unique street-measurement coordinates from all raw street CSVs, skips
coordinates already listed under `requested_points`, queries Overpass in batches,
and optionally saves raw JSON payloads plus requested-point CSVs.

Important settings are in `api/run_api_crawling.py` and
`api/parameter_settings.py`:

- `api_radius_m`: radius around each coordinate for Overpass way lookup.
- `batch_size`: number of points per Overpass request.
- `timeout_s`: Overpass request timeout.
- `max_consecutive_failures`: graceful stop threshold for repeated API failures.
- `OVERPASS_URLS`: endpoint fallback list.

Outputs:

- `data/open_street_map/label_steps/raw_api_data/payloads/*.json`
- `data/open_street_map/label_steps/raw_api_data/requested_points/*.csv`

## Stage 2: Build Location Labels

Entrypoint:

```bash
uv run python -m src.ds_building.open_street_map_data.api.run_location_label_mapping
```

This stage filters Overpass elements to car-drivable ways, computes the nearest
road geometry for each requested point, extracts `smoothness` and `surface`, and
writes a deduplicated location-label parquet file.

Important settings:

- `HIGHWAY_ALLOWED_CAR_STREETS`: OSM `highway` values retained as relevant roads.
- `MAX_POINT_DISTANCE_M`: maximum point-to-way distance before a point is dropped.
- `CLOSEST_WAYS_MARGIN_M`: distance margin used to detect equally close candidate
  ways.

Outputs:

- `data/open_street_map/label_steps/labeled_location_data/labeled_locations.parquet`
- `data/open_street_map/label_steps/labeled_location_data/duplicates.csv`

## Stage 3: Build Dataset Scenarios

Entrypoint:

```bash
uv run python -m src.ds_building.open_street_map_data.datasets.run_osm_ds_build
```

This stage maps OSM labels into numeric road-quality labels using all configured
smoothness, surface, and combination scenarios, then joins each label scenario
onto the street-measurement CSVs.

Scenario definitions live in `datasets/mapping_strategy.py`:

- `SMOOTHNESS_SCENARIOS`: smoothness tag to numeric-label variants.
- `SURFACE_SCENARIOS`: surface tag to numeric-label variants.
- `COMBINATION_SCENARIOS`: strategies for combining smoothness and surface scores.

Outputs:

- Mapped label scenario parquet files:
  `data/open_street_map/label_steps/mapped_labels/osm_labels_*.parquet`
- Joined street datasets:
  `data/open_street_map/datasets/osm_dataset_*.parquet`

## After Rebuilding OSM Datasets

The model-building pipeline may reuse cached feature datasets from:

- `data/molewa/model_building/feature_ds/osm`
- `data/molewa/model_building/feature_ds/manual`

After adding new OSM/street data, either run model building with
`skip_feature_build_if_exists=False` or clear the old feature-cache parquet files
so the model pipeline reads the updated OSM datasets.

## Files

- `run_osm_pipeline.py`: full three-stage OSM entrypoint.
- `api/api_coms.py`: Overpass query construction, request execution, retries, and
  endpoint fallback.
- `api/api_io.py`: coordinate loading, already-requested filtering, raw payload
  saving, and logging.
- `api/run_api_crawling.py`: crawl entrypoint.
- `api/run_location_label_mapping.py`: nearest-way matching and OSM tag extraction.
- `datasets/mapping_strategy.py`: scenario constants and tag-to-label mappings.
- `datasets/smoothness_mappings.py`: smoothness score calculation.
- `datasets/surface_mappings.py`: surface score calculation.
- `datasets/combination_mappings.py`: smoothness/surface score combination.
- `datasets/run_osm_ds_build.py`: mapped-label generation and street-data joins.
