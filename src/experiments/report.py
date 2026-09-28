"""Build a self-contained HTML overview of the optimisation trials logged in one MLflow experiment.

Run from the repository root:

    python -m src.experiments.report
    python -m src.experiments.report --experiment road_quality_test --output data/reports/test.html

The page loads plotly.js from a CDN, so it needs internet access when it is opened.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.experiments.experiment_config_defaults import EXPERIMENT_NAME

PLOTLY_JS_URL = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
MODEL_ORDER = ["Linear", "XGBoost", "ANN"]
MODEL_COLORS = {"Linear": "#8c8c8c", "XGBoost": "#2a7ab9", "ANN": "#d9822b"}
CASE_LABELS = {
    "case_a__osm_no_points": "A (manual only)",
    "case_b__osm_limited_points": "B (manual + limited OSM)",
    "case_b__osm_all_points": "B (manual + all OSM)",
    "case_c__osm_limited_points": "C (limited OSM only)",
    "case_c__osm_all_points": "C (all OSM only)",
}
# Model parameters that never vary within an optimisation and only add noise to the report.
CONSTANT_MODEL_PARAMS = {"model_device", "model_random_state", "model_objective", "model_tree_method"}


def _get_tracking_uri() -> str:
    """Return the MLflow tracking URI from the environment or the unversioned settings file."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri is None:
        from src.experiments.mlflow_secret import MLFLOW_TRACKING_URI

        tracking_uri = MLFLOW_TRACKING_URI
    return tracking_uri


def load_trial_runs(experiment_name: str) -> pd.DataFrame:
    """Load all finished trial runs of an experiment into one flat table."""
    import mlflow

    mlflow.set_tracking_uri(_get_tracking_uri())
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string="tags.run_type = 'trial' and attributes.status = 'FINISHED'",
        max_results=100_000,
    )
    if runs.empty:
        raise SystemExit(f"No finished trial runs found in experiment '{experiment_name}'.")

    runs = runs.rename(columns=lambda c: c.split(".", 1)[1] if c.startswith(("tags.", "params.", "metrics.")) else c)
    runs["case"] = runs["dataset_case_group"].map(CASE_LABELS).fillna(runs["dataset_case_group"])
    time_threshold = runs["manual_ds_manual_time_threshold"].fillna("?")
    runs["manual_ds"] = (runs["manual_ds_manual_mapping_procedure"].fillna("?")
                         + (" / t" + time_threshold).where(time_threshold != "not_applicable", ""))
    runs["osm_ds"] = ("sm" + runs["osm_ds_osm_smoothness_mapping"].fillna("?")
                      + " surf" + runs["osm_ds_osm_surface_mapping"].fillna("?")
                      + " c" + runs["osm_ds_osm_combination_mapping"].fillna("?"))
    runs.loc[runs["osm_ds_osm_smoothness_mapping"] == "not_applicable", "osm_ds"] = "none"
    return runs


def _case_order(runs: pd.DataFrame) -> list[str]:
    """Return the case labels present in the runs, in pipeline order."""
    present = set(runs["case"])
    ordered = [label for label in CASE_LABELS.values() if label in present]
    return ordered + sorted(present - set(ordered))


def _models(runs: pd.DataFrame) -> list[str]:
    """Return the models present in the runs, in a stable order."""
    present = set(runs["model"])
    return [m for m in MODEL_ORDER if m in present] + sorted(present - set(MODEL_ORDER))


def _sort_values(values: pd.Series) -> list[str]:
    """Sort parameter values numerically where possible, otherwise alphabetically."""
    def key(value: str) -> tuple[int, float | str]:
        try:
            return 0, float(value)
        except ValueError:
            return 1, value
    return sorted(values.dropna().unique(), key=key)


def _clean(value):
    """Make a value JSON-safe for plotly, mapping NaN to null."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _figure(figure_id: str, title: str, traces: list[dict], layout: dict, height: int = 420) -> str:
    """Render one plotly chart as an HTML block."""
    layout = {"title": {"text": title, "font": {"size": 15}}, "height": height,
              "margin": {"l": 60, "r": 20, "t": 50, "b": 60}, "template": "plotly_white", **layout}
    payload = json.dumps({"id": figure_id, "data": traces, "layout": layout}, default=_clean)
    # Charts are only queued here and drawn once the page is laid out, see build_report. Drawing them
    # while the page is still parsed makes plotly measure half-filled grids and size the charts too wide.
    return f'<div class="chart" id="{figure_id}"></div><script>FIGURES.push({payload});</script>'


def leaderboard_table(runs: pd.DataFrame) -> str:
    """Summarise every model per case: run count, best and median macro-F1, best MAE and training time."""
    rows = []
    for case in _case_order(runs):
        for model in _models(runs):
            group = runs[(runs["case"] == case) & (runs["model"] == model)]
            if group.empty:
                continue
            best = group.loc[group["f1_macro"].idxmax()]
            rows.append(
                f"<tr><td>{html.escape(case)}</td><td>{html.escape(model)}</td><td>{len(group)}</td>"
                f"<td><b>{best['f1_macro']:.3f}</b> ± {best['f1_macro_std']:.3f}</td>"
                f"<td>{group['f1_macro'].median():.3f}</td><td>{best['f1_bad']:.3f}</td>"
                f"<td>{group['mae'].min():.3f}</td><td>{group['train_time_s'].median():.2f}</td>"
                f"<td>{html.escape(str(best['parameter_set_id']))}</td>"
                f"<td>{html.escape(best['manual_ds'])}</td><td>{html.escape(best['osm_ds'])}</td></tr>"
            )
    header = ("<tr><th>Case</th><th>Model</th><th>Trials</th><th>Best f1_macro</th><th>Median f1_macro</th>"
              "<th>f1_bad of best</th><th>Best MAE</th><th>Median train s</th><th>Best param set</th>"
              "<th>Best manual ds</th><th>Best OSM ds</th></tr>")
    return f'<table class="table">{header}{"".join(rows)}</table>'


def f1_distribution_chart(runs: pd.DataFrame) -> str:
    """Box plot of macro-F1 over all trials, per case and model."""
    traces = [
        {"type": "box", "name": model, "x": runs.loc[runs["model"] == model, "case"].tolist(),
         "y": runs.loc[runs["model"] == model, "f1_macro"].round(4).tolist(),
         "marker": {"color": MODEL_COLORS.get(model)}, "boxpoints": False}
        for model in _models(runs)
    ]
    layout = {"boxmode": "group", "yaxis": {"title": "f1_macro"},
              "xaxis": {"categoryorder": "array", "categoryarray": _case_order(runs)}}
    return _figure("f1-distribution", "Macro-F1 over all trials", traces, layout)


def dataset_heatmaps(runs: pd.DataFrame) -> str:
    """Heatmaps of the best macro-F1 per manual x OSM dataset variant, one per case and model."""
    blocks = []
    for case in _case_order(runs):
        case_runs = runs[runs["case"] == case]
        for model in _models(case_runs):
            pivot = (case_runs[case_runs["model"] == model]
                     .pivot_table(index="osm_ds", columns="manual_ds", values="f1_macro", aggfunc="max"))
            if pivot.empty:
                continue
            pivot = pivot.loc[pivot.max(axis=1).sort_values().index]
            trace = {"type": "heatmap", "z": pivot.round(3).values.tolist(), "x": pivot.columns.tolist(),
                     "y": pivot.index.tolist(), "colorscale": "Viridis", "zmin": 0, "zmax": 1,
                     "texttemplate": "%{z:.3f}", "hovertemplate": "manual %{x}<br>OSM %{y}<br>f1 %{z}<extra></extra>"}
            height = max(260, 60 + 22 * len(pivot.index))
            figure_id = f"heatmap-{len(blocks)}"
            blocks.append(_figure(figure_id, f"{case} · {model}", [trace],
                                  {"margin": {"l": 120, "r": 20, "t": 50, "b": 60}}, height))
    return "".join(blocks)


def hyperparameter_charts(runs: pd.DataFrame) -> str:
    """Box plots of macro-F1 per value of every varied hyperparameter, per model and case."""
    blocks = []
    for model in _models(runs):
        model_runs = runs[runs["model"] == model]
        params = [c for c in model_runs.columns
                  if c.startswith("model_") and c not in CONSTANT_MODEL_PARAMS
                  and model_runs[c].notna().any() and model_runs[c].nunique() > 1]
        if not params:
            continue
        blocks.append(f"<h3>{html.escape(model)}</h3><div class='grid'>")
        for param in params:
            values = _sort_values(model_runs[param])
            traces = []
            for case in _case_order(model_runs):
                case_runs = model_runs[model_runs["case"] == case]
                traces.append({"type": "box", "name": case, "x": case_runs[param].tolist(),
                               "y": case_runs["f1_macro"].round(4).tolist(), "boxpoints": False})
            layout = {"boxmode": "group", "showlegend": len(traces) > 1, "legend": {"orientation": "h", "y": -0.25},
                      "xaxis": {"type": "category", "categoryorder": "array", "categoryarray": values,
                                "title": param.removeprefix("model_")},
                      "yaxis": {"title": "f1_macro"}}
            blocks.append(_figure(f"hp-{model}-{param}", param.removeprefix("model_"), traces, layout, 340))
        blocks.append("</div>")
    return "".join(blocks)


def scatter_charts(runs: pd.DataFrame) -> str:
    """Scatter plots for class trade-offs and training cost, coloured by model."""
    def traces(x: str, y: str) -> list[dict]:
        return [
            {"type": "scatter", "mode": "markers", "name": model,
             "x": runs.loc[runs["model"] == model, x].round(4).tolist(),
             "y": runs.loc[runs["model"] == model, y].round(4).tolist(),
             "text": (runs.loc[runs["model"] == model, "case"] + " · set "
                      + runs.loc[runs["model"] == model, "parameter_set_id"].astype(str)).tolist(),
             "marker": {"color": MODEL_COLORS.get(model), "size": 4, "opacity": 0.5}}
            for model in _models(runs)
        ]
    classes = _figure("f1-classes", "f1_bad vs f1_good per trial", traces("f1_good", "f1_bad"),
                      {"xaxis": {"title": "f1_good"}, "yaxis": {"title": "f1_bad"}})
    cost = _figure("cost", "Training time vs macro-F1 per trial", traces("train_time_s", "f1_macro"),
                   {"xaxis": {"title": "train_time_s (log)", "type": "log"}, "yaxis": {"title": "f1_macro"}})
    return f"<div class='grid'>{classes}{cost}</div>"


def top_trials_table(runs: pd.DataFrame, n: int = 5) -> str:
    """List the best trials per case and model with their dataset variants and hyperparameters."""
    blocks = []
    for case in _case_order(runs):
        for model in _models(runs):
            group = runs[(runs["case"] == case) & (runs["model"] == model)].nlargest(n, "f1_macro")
            if group.empty:
                continue
            params = [c for c in group.columns if c.startswith("model_") and c not in CONSTANT_MODEL_PARAMS
                      and group[c].notna().any()]
            header = "".join(f"<th>{html.escape(p.removeprefix('model_'))}</th>" for p in params)
            rows = "".join(
                f"<tr><td>{r['f1_macro']:.3f}</td><td>{r['f1_bad']:.3f}</td><td>{r['mae']:.3f}</td>"
                f"<td>{html.escape(str(r['parameter_set_id']))}</td><td>{html.escape(r['manual_ds'])}</td>"
                f"<td>{html.escape(r['osm_ds'])}</td>"
                + "".join(f"<td>{html.escape(str(r[p]))}</td>" for p in params) + "</tr>"
                for _, r in group.iterrows()
            )
            blocks.append(
                f"<details><summary>{html.escape(case)} · {html.escape(model)}</summary>"
                f"<div class='scroll'><table class='table'><tr><th>f1_macro</th><th>f1_bad</th><th>MAE</th>"
                f"<th>Param set</th><th>Manual ds</th><th>OSM ds</th>{header}</tr>{rows}</table></div></details>"
            )
    return "".join(blocks)


def build_report(runs: pd.DataFrame, experiment_name: str) -> str:
    """Assemble all sections into one HTML page."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    counts = runs.groupby(["case", "model"]).size()
    summary = f"{len(runs)} finished trials · {counts.index.get_level_values(0).nunique()} cases · generated {generated}"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(experiment_name)} report</title>
<script src="{PLOTLY_JS_URL}"></script>
<script>const FIGURES = [];</script>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0 auto; max-width: 1400px; padding: 16px 24px; color: #1f2328; background: #fff; }}
  h1 {{ margin-bottom: 4px; }} h2 {{ margin-top: 40px; border-bottom: 1px solid #d0d7de; padding-bottom: 4px; }}
  .meta, .hint {{ color: #59636e; }} .hint {{ margin-top: -4px; }}
  .table {{ border-collapse: collapse; font-size: 13px; }}
  .table th, .table td {{ border: 1px solid #d0d7de; padding: 4px 8px; text-align: left; white-space: nowrap; }}
  .table th {{ background: #f6f8fa; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 8px; }}
  .chart {{ min-width: 0; overflow: hidden; }}
  .scroll {{ overflow-x: auto; }} details {{ margin: 6px 0; }} summary {{ cursor: pointer; font-weight: 600; }}
</style>
</head>
<body>
<h1>{html.escape(experiment_name)}</h1>
<div class="meta">{summary}</div>

<h2>Leaderboard</h2>
<p class="hint">Best trial per case and model, ranked by macro-F1 (± std over the cross-validation folds).</p>
<div class="scroll">{leaderboard_table(runs)}</div>
{f1_distribution_chart(runs)}

<h2>Dataset variants</h2>
<p class="hint">Best macro-F1 per manual (columns) and OSM (rows) dataset variant, over all hyperparameter sets.</p>
<div class="grid">{dataset_heatmaps(runs)}</div>

<h2>Hyperparameters</h2>
<p class="hint">Macro-F1 of all trials per value of each varied hyperparameter. Wide boxes mean the value matters less than the dataset.</p>
{hyperparameter_charts(runs)}

<h2>Class trade-off and cost</h2>
{scatter_charts(runs)}

<h2>Top trials</h2>
{top_trials_table(runs)}
<script>
  window.addEventListener("load", () => {{
    for (const f of FIGURES) {{
      Plotly.newPlot(f.id, f.data, f.layout, {{responsive: true, displaylogo: false}});
    }}
  }});
</script>
</body>
</html>
"""


def main() -> None:
    """Parse arguments, load the runs and write the report."""
    parser = argparse.ArgumentParser(description="Build an HTML overview of the optimisation trials in MLflow.")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    parser.add_argument("--output", default=None, help="HTML file to write, default data/reports/<experiment>.html")
    args = parser.parse_args()

    output = Path(args.output or f"data/reports/{args.experiment}.html")
    runs = load_trial_runs(args.experiment)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(runs, args.experiment), encoding="utf-8")
    print(f"Wrote report for {len(runs)} trials to {output}")


if __name__ == "__main__":
    main()
