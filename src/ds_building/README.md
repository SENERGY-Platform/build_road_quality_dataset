# Dataset Building

This folder contains the dataset-building entrypoints for road-quality data.

There are two dataset sources:

- Manual labels: joins manually labelled road-quality points to nearby street
  sensor measurements.
- OpenStreetMap: crawls OSM tags for street-measurement locations and maps those
  tags into road-quality label scenarios.

## Quick Start

Run all dataset-building steps from the repository root:

```bash
uv run python -m src.ds_building.run_ds_building
```

By default this runs the manual-label pipeline first, then the OSM pipeline. The
OSM pipeline can call public Overpass API servers, so it may take time and depends
on external service availability.

Run only one source:

```bash
uv run python -m src.ds_building.run_ds_building --manual
uv run python -m src.ds_building.run_ds_building --osm
```

`--all` is also available explicitly and is equivalent to the default.

## Inputs

- Manual labels: `data/molewa/labels/*.csv`
- Street measurements: `data/molewa/raw/*.csv`
- Existing OSM payloads and labels under `data/open_street_map/label_steps/`

Do not keep duplicate data source CSVs in the same input
folder as every `*.csv` file in the input folders is included.

## Outputs

- Manual datasets: `data/molewa/datasets/`
- OSM mapped labels: `data/open_street_map/label_steps/mapped_labels/`
- OSM joined datasets: `data/open_street_map/datasets/`

After rebuilding datasets, the model-building pipeline may still reuse cached
feature datasets under `data/molewa/model_building/feature_ds/`. Rebuild or clear
that cache before model training if the input datasets changed, see SKIP_FEATURE_BUILD_IF_EXISTS flag in model pipeline.

## Sub-Pipelines

Manual labels only:

```bash
uv run python -m src.ds_building.manual_labels.run_manual_ds_build
```

OpenStreetMap only:

```bash
uv run python -m src.ds_building.open_street_map_data.run_osm_pipeline
```

See the sub-pipeline READMEs for stage details and additional options.
Use the sub-pipeline runners directly when you need to override input/output
paths or OSM crawl settings.
