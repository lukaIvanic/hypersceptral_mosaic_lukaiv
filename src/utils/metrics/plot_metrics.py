from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from .schema import (
    ACCURACY_METRIC_UNITS,
    METRIC_DISPLAY_NAMES,
    METRIC_GUIDANCE,
    SPEED_METRICS,
    TRAINING_METRIC_UNITS,
)
from .storage import load_metrics_history

SECTION_LABELS: Dict[str, str] = {
    "train": "Train",
    "val": "Val (legacy)",
    "val_native": "Val (Native)",
    "val_resized": "Val (Train Resized)",
    "eval_resized": "Eval (Resized)",
}

SECTION_STYLES: Dict[str, Dict[str, Any]] = {
    "train": {"color": "#1f77b4", "marker": "o"},
    "val": {"color": "#9467bd", "marker": "D"},
    "val_native": {"color": "#2ca02c", "marker": "s"},
    "val_resized": {"color": "#ff7f0e", "marker": "^"},
    "eval_resized": {"color": "#17becf", "marker": "v", "linestyle": "--"},
}

DEFAULT_MARKER = "o"
DEFAULT_LINEWIDTH = 2.0
DEFAULT_MARKERSIZE = 4.5

OVERVIEW_METRICS = ["sam", "sid", "ergas", "loss"]


def safe_filename(name: str) -> str:
    return name.lower().replace("/", "_").replace(" ", "_")


def display_name(metric: str) -> str:
    return METRIC_DISPLAY_NAMES.get(metric, metric.replace("_", " ").title())


def metric_category(metric: str) -> str:
    return "speed" if metric in SPEED_METRICS else "accuracy"


def fallback_unit(metric: str) -> str:
    return (
        ACCURACY_METRIC_UNITS.get(metric)
        or TRAINING_METRIC_UNITS.get(metric)
        or "1"
    )


def _is_metric_section(section: str) -> bool:
    if section in {"train", "val"}:
        return True
    return section.startswith("val_") or section.startswith("eval_")


def build_metric_series(history: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    series: Dict[str, Dict[str, Any]] = {}

    for entry in history:
        if not isinstance(entry, dict):
            continue
        epoch_raw = entry.get("epoch")
        try:
            epoch = int(epoch_raw)
        except (TypeError, ValueError):
            continue
        units_entry = entry.get("units") if isinstance(entry.get("units"), dict) else {}

        for section, metrics in entry.items():
            if not isinstance(section, str) or not _is_metric_section(section):
                continue
            if not isinstance(metrics, dict):
                continue
            for metric_name, raw_value in metrics.items():
                if raw_value is None:
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue

                metric_data = series.setdefault(
                    metric_name,
                    {
                        "section_data": {},
                        "units": {},
                        "category": metric_category(metric_name),
                    },
                )
                section_data = metric_data["section_data"].setdefault(
                    section, {"epochs": [], "values": []}
                )
                section_data["epochs"].append(epoch)
                section_data["values"].append(value)

                unit_candidate = None
                if isinstance(units_entry, dict):
                    section_units = units_entry.get(section)
                    if isinstance(section_units, dict):
                        unit_candidate = section_units.get(metric_name)
                if unit_candidate:
                    metric_data["units"][section] = unit_candidate

    for metric_name, metric_data in series.items():
        for section_data in metric_data["section_data"].values():
            points = sorted(zip(section_data["epochs"], section_data["values"]))
            if points:
                epochs_sorted, values_sorted = zip(*points)
                section_data["epochs"] = list(epochs_sorted)
                section_data["values"] = list(values_sorted)
            else:
                section_data["epochs"] = []
                section_data["values"] = []

        unit = next(iter(metric_data["units"].values()), None)
        if not unit:
            unit = fallback_unit(metric_name)
        metric_data["unit"] = unit

    return series


def build_ylabel(unit: str) -> str:
    if not unit or unit == "1":
        return "Value (unitless)"
    return f"Value ({unit})"


def plot_section_line(ax: plt.Axes, section_key: str, data: Dict[str, List[float]]) -> bool:
    epochs = data.get("epochs", [])
    values = data.get("values", [])
    if not epochs:
        return False

    style = SECTION_STYLES.get(section_key, {})
    label = SECTION_LABELS.get(section_key, section_key.replace("_", " ").title())
    plot_kwargs: Dict[str, Any] = {
        "label": label,
        "linewidth": style.get("linewidth", DEFAULT_LINEWIDTH),
        "marker": style.get("marker", DEFAULT_MARKER),
        "markersize": style.get("markersize", DEFAULT_MARKERSIZE),
    }

    color = style.get("color")
    if color:
        plot_kwargs["color"] = color
    linestyle = style.get("linestyle")
    if linestyle:
        plot_kwargs["linestyle"] = linestyle

    ax.plot(epochs, values, **plot_kwargs)
    return True


def apply_guidance(ax: plt.Axes, metric: str, data_min: float, data_max: float) -> None:
    guidance = METRIC_GUIDANCE.get(metric)
    if not guidance:
        return

    goal = guidance.get("goal")
    good = guidance.get("good")
    warn = guidance.get("warn")
    x_min, x_max = ax.get_xlim()

    def annotate(value: float, text: str, color: str, vertical_align: str) -> None:
        ax.axhline(
            value,
            color=color,
            linestyle="--" if text == "target" else ":",
            linewidth=1,
            alpha=0.65,
        )
        ax.annotate(
            f"{text}: {value:.3g}",
            xy=(x_max, value),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va=vertical_align,
            fontsize=8,
            color=color,
            alpha=0.75,
        )

    if good is not None:
        annotate(good, "target", "tab:green", "bottom" if goal == "max" else "top")
    if warn is not None:
        annotate(warn, "caution", "tab:orange", "bottom" if goal == "max" else "top")
    if goal:
        ax.text(
            0.02,
            0.94,
            f"Goal: {'higher' if goal == 'max' else 'lower'}",
            transform=ax.transAxes,
            fontsize=8,
            color="#444",
            alpha=0.8,
        )


def plot_metric(
    metric_name: str,
    metric_data: Dict[str, Any],
    output_dir: Path,
    dpi: int,
) -> bool:
    sections = metric_data["section_data"]
    if not any(data.get("epochs") for data in sections.values()):
        return False

    fig, ax = plt.subplots(figsize=(9, 5))
    all_values: List[float] = []
    for section_key, data in sections.items():
        if plot_section_line(ax, section_key, data):
            all_values.extend(data.get("values", []))

    if not all_values:
        plt.close(fig)
        return False

    ax.set_xlabel("Epoch")
    ax.set_ylabel(build_ylabel(metric_data["unit"]))
    ax.set_title(display_name(metric_name))
    ax.grid(True, alpha=0.3)
    if len(ax.get_lines()) > 1:
        ax.legend()

    ax.relim()
    ax.autoscale_view()
    apply_guidance(ax, metric_name, min(all_values), max(all_values))

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{safe_filename(metric_name)}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_comparison(
    metric_name: str,
    metric_data: Dict[str, Any],
    output_dir: Path,
    dpi: int,
) -> bool:
    sections = metric_data["section_data"]
    native = sections.get("val_native")
    resized = sections.get("val_resized")
    if not native or not resized:
        return False
    if not native.get("epochs") or not resized.get("epochs"):
        return False

    fig, (ax_main, ax_diff) = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        height_ratios=[3, 1.2],
    )

    values: List[float] = []
    plot_section_line(ax_main, "val_native", native)
    values.extend(native["values"])
    plot_section_line(ax_main, "val_resized", resized)
    values.extend(resized["values"])

    ax_main.set_ylabel(build_ylabel(metric_data["unit"]))
    ax_main.set_title(f"{display_name(metric_name)}: Native vs Resized")
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()
    ax_main.relim()
    ax_main.autoscale_view()
    apply_guidance(ax_main, metric_name, min(values), max(values))

    native_map = dict(zip(native["epochs"], native["values"]))
    resized_map = dict(zip(resized["epochs"], resized["values"]))
    shared_epochs = sorted(set(native_map) & set(resized_map))
    if not shared_epochs:
        plt.close(fig)
        return False

    diffs = [native_map[epoch] - resized_map[epoch] for epoch in shared_epochs]
    ax_diff.plot(shared_epochs, diffs, color="#8c564b", marker="o", linewidth=1.5)
    ax_diff.axhline(0.0, color="#333", linewidth=1)

    guidance = METRIC_GUIDANCE.get(metric_name, {})
    goal = guidance.get("goal")
    max_diff = max(diffs + [0.0])
    min_diff = min(diffs + [0.0])

    if goal == "max":
        if max_diff > 0:
            ax_diff.axhspan(0, max_diff, color="tab:green", alpha=0.12)
        if min_diff < 0:
            ax_diff.axhspan(min_diff, 0, color="tab:red", alpha=0.08)
        orientation = "Above zero → native higher (better)"
    elif goal == "min":
        if max_diff > 0:
            ax_diff.axhspan(0, max_diff, color="tab:red", alpha=0.08)
        if min_diff < 0:
            ax_diff.axhspan(min_diff, 0, color="tab:green", alpha=0.12)
        orientation = "Below zero → native lower (better)"
    else:
        orientation = "Reference: native - resized"

    diff_extent = max(abs(min_diff), abs(max_diff))
    if diff_extent > 0:
        ax_diff.set_ylim(-diff_extent * 1.2, diff_extent * 1.2)

    ax_diff.set_xlabel("Epoch")
    unit = metric_data["unit"]
    if unit and unit != "1":
        ax_diff.set_ylabel(f"Δ ({unit})")
    else:
        ax_diff.set_ylabel("Δ")
    ax_diff.grid(True, alpha=0.3)
    ax_diff.text(
        0.5,
        0.85,
        orientation,
        transform=ax_diff.transAxes,
        ha="center",
        fontsize=8,
        color="#555",
    )

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{safe_filename(metric_name)}_comparison.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)
    return True


def plot_overview(
    metric_series: Dict[str, Dict[str, Any]],
    output_dir: Path,
    dpi: int,
) -> bool:
    available = [metric for metric in OVERVIEW_METRICS if metric in metric_series]
    if not available:
        return False

    cols = 2
    rows = math.ceil(len(available) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, metric in enumerate(available):
        ax = axes_flat[idx]
        metric_data = metric_series[metric]
        sections = metric_data["section_data"]
        plotted = False
        values: List[float] = []

        for section_key in ("val_native", "val_resized", "eval_resized", "val"):
            section_data = sections.get(section_key)
            if section_data and plot_section_line(ax, section_key, section_data):
                values.extend(section_data["values"])
                plotted = True

        if metric == "loss":
            train_data = sections.get("train")
            if train_data and plot_section_line(ax, "train", train_data):
                values.extend(train_data["values"])
                plotted = True

        if not plotted:
            ax.set_visible(False)
            continue

        ax.set_title(display_name(metric))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(build_ylabel(metric_data["unit"]))
        ax.grid(True, alpha=0.3)
        if len(ax.get_lines()) > 1:
            ax.legend(fontsize=8)
        ax.relim()
        ax.autoscale_view()
        apply_guidance(ax, metric, min(values), max(values))

    for idx in range(len(available), len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    fig.suptitle("Validation Metrics Overview", fontsize=14, fontweight="semibold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "metrics_overview.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def generate_plots(
    metric_series: Dict[str, Dict[str, Any]],
    base_output: Path,
    dpi: int,
    style: str,
) -> Tuple[Dict[str, Any], Path, Path, Path]:
    plt.style.use(style)

    accuracy_dir = base_output / "accuracy"
    speed_dir = base_output / "speed"
    comparisons_dir = base_output / "comparisons"

    counts: Dict[str, Any] = {
        "accuracy": 0,
        "speed": 0,
        "comparisons": 0,
        "overview": False,
    }

    for metric_name, metric_data in metric_series.items():
        category = metric_data.get("category", "accuracy")
        target_dir = speed_dir if category == "speed" else accuracy_dir
        if plot_metric(metric_name, metric_data, target_dir, dpi):
            counts[category] += 1
        if category != "speed":
            if plot_comparison(metric_name, metric_data, comparisons_dir, dpi):
                counts["comparisons"] += 1

    counts["overview"] = plot_overview(metric_series, comparisons_dir, dpi)
    return counts, accuracy_dir, speed_dir, comparisons_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training and validation metrics for a run.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the run directory (e.g., src/models/simple_cnn/runs/<run-name>).",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        help="Path to metrics JSON file (defaults to <run-dir>/metrics.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory to write plots (defaults to <run-dir>/metrics).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Image resolution for saved plots.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="seaborn-v0_8",
        help="Matplotlib style to use for plots.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    metrics_path = (
        Path(args.metrics_file).expanduser().resolve()
        if args.metrics_file
        else run_dir / "metrics.json"
    )
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run_dir / "metrics"
    )

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    try:
        history = load_metrics_history(metrics_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not history:
        raise SystemExit("Metrics history is empty; nothing to plot.")

    history_sorted = sorted(history, key=lambda entry: entry.get("epoch", 0))
    metric_series = build_metric_series(history_sorted)
    if not metric_series:
        raise SystemExit("No metrics found in history after parsing sections.")

    counts, accuracy_dir, speed_dir, comparisons_dir = generate_plots(
        metric_series,
        output_root,
        dpi=args.dpi,
        style=args.style,
    )

    print(f"[Plot] Accuracy figures: {counts['accuracy']} → {accuracy_dir}")
    if counts["speed"]:
        print(f"[Plot] Speed figures: {counts['speed']} → {speed_dir}")
    else:
        print("[Plot] Speed figures: none generated (no timing metrics found).")

    if counts["comparisons"]:
        print(f"[Plot] Native vs Resized comparisons: {counts['comparisons']} → {comparisons_dir}")
    else:
        print("[Plot] Comparisons: none generated (need both val_native and val_resized).")

    if counts["overview"]:
        overview_path = comparisons_dir / "metrics_overview.png"
        print(f"[Plot] Overview saved to {overview_path}")
    else:
        print("[Plot] Overview not generated (missing required metrics).")


if __name__ == "__main__":
    main()

