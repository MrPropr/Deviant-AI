"""Generate the five public-safe pilot figures from aggregate metrics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


CONDITIONS = ("direct", "polite", "multi_turn", "polite_multi_turn")


def existing_file(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Input path is not a file: {path}")
    return path


def output_dir(value: str) -> Path:
    path = Path(value)
    if path.exists() and not path.is_dir():
        raise argparse.ArgumentTypeError(f"Output path is not a directory: {path}")
    return path


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def value(row: dict, field: str) -> float:
    try:
        return float(row.get(field, ""))
    except (TypeError, ValueError):
        return math.nan


def load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("A working matplotlib/numpy installation is required to generate figures.") from exc
    return plt


def grouped_bar(plt, rows: list[dict], field: str, title: str, ylabel: str, path: Path) -> None:
    models = sorted({row["model_id"] for row in rows})
    width = 0.8 / max(len(models), 1)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for model_index, model_id in enumerate(models):
        model_rows = {row["condition"]: row for row in rows if row["model_id"] == model_id}
        positions = [index - 0.4 + width / 2 + model_index * width for index in range(len(CONDITIONS))]
        axis.bar(positions, [value(model_rows.get(condition, {}), field) for condition in CONDITIONS], width, label=model_id)
    axis.set_xticks(range(len(CONDITIONS)), CONDITIONS)
    axis.set_ylim(0, 1)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_asr_heatmap(plt, rows: list[dict], path: Path) -> None:
    models = sorted({row["model_id"] for row in rows})
    matrix = []
    for model_id in models:
        model_rows = {row["condition"]: row for row in rows if row["model_id"] == model_id}
        matrix.append([value(model_rows.get(condition, {}), "strict_asr") for condition in CONDITIONS])
    figure, axis = plt.subplots(figsize=(9, max(3, 0.7 * len(models) + 2)))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(CONDITIONS)), CONDITIONS)
    axis.set_yticks(range(len(models)), models)
    axis.set_title("Strict ASR by Model and Condition")
    for row_index, row_values in enumerate(matrix):
        for column_index, cell in enumerate(row_values):
            if not math.isnan(cell):
                axis.text(column_index, row_index, f"{cell:.2f}", ha="center", va="center", color="white")
    figure.colorbar(image, ax=axis, label="Strict ASR")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_discovery_curve(plt, rows: list[dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for row in rows:
        if row["condition"] not in {"multi_turn", "polite_multi_turn"}:
            continue
        rates = [value(row, f"turn_{turn}_discovery_rate") for turn in (1, 2, 3)]
        axis.plot([1, 2, 3], rates, marker="o", label=f"{row['model_id']} / {row['condition']}")
    axis.set_xticks([1, 2, 3])
    axis.set_ylim(0, 1)
    axis.set_xlabel("Turn")
    axis.set_ylabel("Cumulative harmful discovery rate")
    axis.set_title("Multi-turn Harmful Discovery Curve")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_safety_utility(plt, rows: list[dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for row in rows:
        benign_success = value(row, "benign_success_rate")
        strict_asr = value(row, "strict_asr")
        if math.isnan(benign_success) or math.isnan(strict_asr):
            continue
        axis.scatter(benign_success, 1.0 - strict_asr)
        axis.annotate(f"{row['model_id']} / {row['condition']}", (benign_success, 1.0 - strict_asr), fontsize=8)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Benign success rate (utility)")
    axis.set_ylabel("1 - strict ASR (safety)")
    axis.set_title("Safety versus Utility")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=existing_file, required=True, help="Aggregate pilot metrics CSV.")
    parser.add_argument("--output-dir", type=output_dir, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    if not rows:
        raise SystemExit(f"No aggregate rows found in {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        plt = load_matplotlib()
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    outputs = [
        args.output_dir / "asr_heatmap.png",
        args.output_dir / "harmfulness_score_bar.png",
        args.output_dir / "refusal_rate_chart.png",
        args.output_dir / "multi_turn_discovery_curve.png",
        args.output_dir / "safety_utility_scatter.png",
    ]
    plot_asr_heatmap(plt, rows, outputs[0])
    grouped_bar(plt, rows, "mean_harmfulness_score", "Mean Harmfulness Score", "Mean score", outputs[1])
    grouped_bar(plt, rows, "refusal_rate", "Refusal Rate", "Refusal rate", outputs[2])
    plot_discovery_curve(plt, rows, outputs[3])
    plot_safety_utility(plt, rows, outputs[4])
    print(f"Generated {len(outputs)} aggregate figures in {args.output_dir}")


if __name__ == "__main__":
    main()
