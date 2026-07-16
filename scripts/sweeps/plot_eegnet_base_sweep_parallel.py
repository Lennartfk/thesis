import argparse
import math
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_DIR = PROJECT_ROOT / "data/results/sweeps/eegnet_base"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs/sweeps/eegnet_base"


AXES = [
    ("learning_rate", "learning rate", "log"),
    ("batch_size", "batch size", "linear"),
    ("weight_decay", "weight decay", "log_zero"),
    ("eegnet_dropout", "dropout", "linear"),
    ("eegnet_f1", "F1 filters", "linear"),
    ("eegnet_depth_multiplier", "depth multiplier", "linear"),
    ("eegnet_temporal_kernel_length", "temporal k.", "linear"),
    ("eegnet_separable_kernel_length", "separable k.", "linear"),
]

METRIC_COLUMN = "balanced_accuracy_mean"
METRIC_LABEL = "k-fold balanced acc."
OUTPUT_HTML = DEFAULT_SWEEP_DIR / "eegnet_base_parallel_coordinates.html"
OUTPUT_PNG = DEFAULT_SWEEP_DIR / "eegnet_base_parallel_coordinates.png"


def extract_trial_index(run_name):
    match = re.search(r"_t(\d{3})_", run_name)
    if not match:
        raise ValueError(f"Could not extract trial index from run name: {run_name}")
    return match.group(1)


def load_trial_config(config_dir, run_name):
    trial_index = extract_trial_index(run_name)
    config_path = config_dir / f"trial_{trial_index}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing trial config: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_completed_runs(sweep_dir, config_dir):
    rows = []
    for run_dir in sorted(path for path in sweep_dir.iterdir() if path.is_dir()):
        summary_path = run_dir / "summary_metrics.csv"
        if not summary_path.exists():
            continue

        summary = pd.read_csv(summary_path)
        if summary.empty:
            continue

        config = load_trial_config(config_dir, run_dir.name)
        row = {
            "run_name": run_dir.name,
            "config_path": str(config_dir / f"trial_{extract_trial_index(run_dir.name)}.yaml"),
            METRIC_COLUMN: float(summary.loc[0, METRIC_COLUMN]),
        }

        for key, _, _ in AXES:
            if key not in config:
                raise KeyError(f"Missing {key} in config for {run_dir.name}")
            row[key] = config[key]

        row["tune_decision_threshold"] = bool(config.get("tune_decision_threshold", False))
        rows.append(row)

    if not rows:
        raise SystemExit(f"No completed runs found under {sweep_dir}")

    return pd.DataFrame(rows)


def build_axis_scale(values, mode):
    clean = sorted(set(values))
    if len(clean) == 1:
        return {clean[0]: 0.5}

    if mode == "log":
        transformed = [math.log10(float(value)) for value in clean]
    elif mode == "log_zero":
        transformed = [math.log10(float(value) + 1e-6) if float(value) > 0 else math.log10(1e-6) - 1.0 for value in clean]
    else:
        transformed = [float(value) for value in clean]

    min_value = min(transformed)
    max_value = max(transformed)
    if math.isclose(min_value, max_value):
        return {value: 0.5 for value in clean}

    return {
        value: 0.08 + 0.84 * ((transformed[index] - min_value) / (max_value - min_value))
        for index, value in enumerate(clean)
    }


def build_figure(df):
    x_positions = list(range(len(AXES) + 1))
    axis_maps = {key: build_axis_scale(df[key].tolist(), scale_mode) for key, _, scale_mode in AXES}

    min_metric = float(df[METRIC_COLUMN].min())
    max_metric = float(df[METRIC_COLUMN].max())
    metric_range = max_metric - min_metric if not math.isclose(min_metric, max_metric) else 1.0
    strip_x0 = len(AXES) + 0.03
    strip_x1 = len(AXES) + 0.18

    def metric_to_color(metric_value):
        normalized = (float(metric_value) - min_metric) / metric_range
        return sample_colorscale("Plasma", normalized)[0]

    best_row = df.sort_values(METRIC_COLUMN, ascending=False).iloc[0]
    best_metric = float(best_row[METRIC_COLUMN])
    best_run_name = str(best_row["run_name"])

    fig = go.Figure()

    for row_index, row in df.reset_index(drop=True).iterrows():
        y_values = [axis_maps[key][row[key]] for key, _, _ in AXES]
        y_values.append(0.08 + 0.84 * ((float(row[METRIC_COLUMN]) - min_metric) / metric_range))

        is_best = str(row["run_name"]) == best_run_name
        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=y_values,
                mode="lines",
                line=dict(
                    shape="spline",
                    smoothing=1.15,
                    color="#111111" if is_best else metric_to_color(row[METRIC_COLUMN]),
                    width=4.0 if is_best else 1.2,
                ),
                opacity=0.98 if is_best else 0.22,
                hoverinfo="skip",
                showlegend=False,
                name=row["run_name"],
            )
        )

    axis_titles = [label for _, label, _ in AXES] + [METRIC_LABEL]
    x_min, x_max = -0.35, len(AXES) + 0.48

    annotations = []
    shapes = []

    for axis_index, (key, label, _) in enumerate(AXES):
        shapes.append(
            dict(
                type="line",
                x0=axis_index,
                x1=axis_index,
                y0=0.02,
                y1=0.98,
                line=dict(color="#2a2a2a", width=1),
                layer="below",
            )
        )
        annotations.append(
            dict(
                x=axis_index,
                y=1.045,
                text=f"<b>{label}</b>",
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=16, color="#111111"),
            )
        )

        tick_map = axis_maps[key]
        for value, y in sorted(tick_map.items(), key=lambda item: item[1]):
            annotations.append(
                dict(
                    x=axis_index - 0.03,
                    y=y,
                    text=f"{value:g}",
                    showarrow=False,
                    xanchor="right",
                    yanchor="middle",
                    font=dict(size=12, color="#111111"),
                )
            )

    metric_x = len(AXES)
    shapes.append(
        dict(
            type="line",
            x0=metric_x,
            x1=metric_x,
            y0=0.02,
            y1=0.98,
            line=dict(color="#2a2a2a", width=1),
            layer="below",
        )
    )

    bar_steps = 54
    for step in range(bar_steps):
        y0 = 0.08 + 0.84 * (step / bar_steps)
        y1 = 0.08 + 0.84 * ((step + 1) / bar_steps)
        value = min_metric + metric_range * ((step + 0.5) / bar_steps)
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=strip_x0,
                x1=strip_x1,
                y0=y0,
                y1=y1,
                line=dict(width=0),
                fillcolor=metric_to_color(value),
                layer="below",
            )
        )

    annotations.append(
        dict(
            x=(strip_x0 + strip_x1) / 2,
            y=1.045,
            text=f"<b>{METRIC_LABEL}</b>",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=16, color="#111111"),
        )
    )

    metric_ticks = [
        min_metric,
        float(df[METRIC_COLUMN].quantile(0.25)),
        float(df[METRIC_COLUMN].median()),
        float(df[METRIC_COLUMN].quantile(0.75)),
        max_metric,
    ]
    for value in metric_ticks:
        y = 0.08 + 0.84 * ((value - min_metric) / metric_range)
        annotations.append(
            dict(
                x=strip_x1 + 0.02,
                y=y,
                text=f"{value:.3f}",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=12, color="#111111"),
            )
        )

    fig.add_annotation(
        x=0.0,
        y=-0.085,
        xref="paper",
        yref="paper",
        text=f"Best run: {best_row['run_name']} | {METRIC_LABEL} = {best_metric:.4f}",
        showarrow=False,
        xanchor="left",
        font=dict(size=13, color="#111111"),
    )

    fig.update_layout(
        title=dict(
            text="EEGNet base sweep: hyperparameters vs. k-fold balanced accuracy",
            x=0.5,
            xanchor="center",
            font=dict(size=24),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        width=1710,
        height=680,
        margin=dict(l=42, r=24, t=82, b=56),
        annotations=annotations,
        shapes=shapes,
        xaxis=dict(range=[x_min, x_max], visible=False, fixedrange=True),
        yaxis=dict(range=[0.0, 1.08], visible=False, fixedrange=True),
    )

    return fig


def render_figure(df, html_path, png_path=None):
    fig = build_figure(df)
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

    if png_path is not None:
        try:
            fig.write_image(str(png_path), scale=2)
        except Exception as exc:
            print(f"Skipped PNG export because Plotly image export is unavailable: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Render a Plotly curved parallel-style plot for the EEGNet base sweep.")
    parser.add_argument("--sweep-dir", default=str(DEFAULT_SWEEP_DIR))
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--output-html", default=str(OUTPUT_HTML))
    parser.add_argument("--output-png", default=str(OUTPUT_PNG))
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    config_dir = Path(args.config_dir)
    output_html = Path(args.output_html)
    output_png = Path(args.output_png) if args.output_png else None

    df = load_completed_runs(sweep_dir, config_dir)
    render_figure(df, output_html, output_png)
    print(f"Saved figure to {output_html}")
    if output_png is not None:
        print(f"Saved figure to {output_png}")


if __name__ == "__main__":
    main()