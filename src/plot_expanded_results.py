"""Reproduce the four public expanded-Qwen figures from aggregate CSV tables."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


CONDITIONS = ("direct", "polite", "multi_turn", "polite_multi_turn")
MULTI_TURN_CONDITIONS = ("multi_turn", "polite_multi_turn")
CONDITION_LABELS = {
    "direct": "Direct",
    "polite": "Polite",
    "multi_turn": "Multi-turn",
    "polite_multi_turn": "Polite multi-turn",
}
CONDITION_COLORS = {
    "direct": "#4C78A8",
    "polite": "#72B7B2",
    "multi_turn": "#E45756",
    "polite_multi_turn": "#F2CF5B",
}
FORBIDDEN_COLUMNS = {
    "prompt",
    "prompt_text",
    "response",
    "response_text",
    "judge_notes",
    "raw_output",
    "conversation_text",
    "harmful_prompt",
}
METRIC_FIELDS = (
    "strict_asr",
    "refusal_rate",
    "partial_compliance_rate",
    "mean_harmfulness_score",
    "benign_success_rate",
    "turn_1_discovery_rate",
    "turn_2_discovery_rate",
    "turn_3_discovery_rate",
)
METRICS_REQUIRED_COLUMNS = {"condition", *METRIC_FIELDS}
GRAPH_REQUIRED_COLUMNS = {"figure", "condition", "metric", "estimate", "ci_low", "ci_high"}
OUTPUT_FILENAMES = (
    "qwen_expanded_asr.png",
    "qwen_expanded_harmfulness.png",
    "qwen_expanded_refusal.png",
    "qwen_expanded_discovery_curve.png",
)
BAR_SPECS = (
    ("asr", "strict_asr", "Strict Attack Success Rate by Condition", "Strict ASR", OUTPUT_FILENAMES[0]),
    (
        "harmfulness",
        "mean_harmfulness_score",
        "Mean Harmfulness by Condition",
        "Mean harmfulness score",
        OUTPUT_FILENAMES[1],
    ),
    ("refusal", "refusal_rate", "Refusal Rate on Harmful Scenarios", "Refusal rate", OUTPUT_FILENAMES[2]),
)


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


def read_public_csv(path: Path, required_columns: set[str]) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        forbidden = sorted(column for column in columns if column.lower() in FORBIDDEN_COLUMNS)
        if forbidden:
            raise ValueError(f"Private columns are not allowed in {path}: {', '.join(forbidden)}")
        missing = sorted(required_columns - columns)
        if missing:
            raise ValueError(f"Missing aggregate columns in {path}: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No aggregate rows found in {path}")
    return rows


def unit_interval(value: object, field: str, path: Path) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric in {path}") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1 in {path}")
    return parsed


def load_metrics(path: Path) -> dict[str, dict]:
    rows = read_public_csv(path, METRICS_REQUIRED_COLUMNS)
    indexed: dict[str, dict] = {}
    for row in rows:
        condition = str(row["condition"])
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition in {path}: {condition}")
        if condition in indexed:
            raise ValueError(f"Duplicate condition in {path}: {condition}")
        parsed = dict(row)
        for field in METRIC_FIELDS:
            parsed[field] = unit_interval(row[field], field, path)
        indexed[condition] = parsed
    if set(indexed) != set(CONDITIONS):
        raise ValueError(f"Expected exactly four prompt conditions in {path}")
    return indexed


def load_graph_values(path: Path) -> list[dict]:
    rows = read_public_csv(path, GRAPH_REQUIRED_COLUMNS)
    parsed_rows: list[dict] = []
    for row in rows:
        condition = str(row["condition"])
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition in {path}: {condition}")
        parsed = dict(row)
        for field in ("estimate", "ci_low", "ci_high"):
            parsed[field] = unit_interval(row[field], field, path)
        if not parsed["ci_low"] <= parsed["estimate"] <= parsed["ci_high"]:
            raise ValueError(f"Confidence interval does not contain its estimate in {path}")
        parsed_rows.append(parsed)
    return parsed_rows


def select_graph_row(rows: list[dict], figure: str, condition: str, metric: str) -> dict:
    matches = [
        row
        for row in rows
        if row["figure"] == figure and row["condition"] == condition and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one graph row for {figure}/{condition}/{metric}; found {len(matches)}")
    return matches[0]


def validate_graph_values(metrics: dict[str, dict], graph_rows: list[dict]) -> None:
    for figure, metric, _title, _ylabel, _filename in BAR_SPECS:
        for condition in CONDITIONS:
            row = select_graph_row(graph_rows, figure, condition, metric)
            if not math.isclose(row["estimate"], metrics[condition][metric], abs_tol=1e-12):
                raise ValueError(f"Graph estimate disagrees with aggregate metrics for {condition}/{metric}")
    for condition in MULTI_TURN_CONDITIONS:
        for turn in (1, 2, 3):
            metric = f"turn_{turn}_discovery_rate"
            row = select_graph_row(graph_rows, "discovery_curve", condition, metric)
            if not math.isclose(row["estimate"], metrics[condition][metric], abs_tol=1e-12):
                raise ValueError(f"Discovery estimate disagrees with aggregate metrics for {condition}/{metric}")


def load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("A working matplotlib installation is required to generate figures.") from exc
    return plt


def save_bar_chart(plt, rows: list[dict], title: str, ylabel: str, path: Path) -> None:
    estimates = [row["estimate"] for row in rows]
    lower_errors = [row["estimate"] - row["ci_low"] for row in rows]
    upper_errors = [row["ci_high"] - row["estimate"] for row in rows]
    positions = list(range(len(CONDITIONS)))
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.bar(
        positions,
        estimates,
        yerr=[lower_errors, upper_errors],
        color=[CONDITION_COLORS[condition] for condition in CONDITIONS],
        edgecolor="#333333",
        linewidth=0.6,
        capsize=6,
    )
    for position, estimate in zip(positions, estimates):
        axis.text(position, min(estimate + 0.035, 0.97), f"{estimate:.3f}", ha="center", va="bottom")
    axis.set_xticks(positions, [CONDITION_LABELS[condition] for condition in CONDITIONS])
    axis.set_ylim(0, 1)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_discovery_chart(plt, graph_rows: list[dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.5))
    turns = [1, 2, 3]
    for condition in MULTI_TURN_CONDITIONS:
        rows = [
            select_graph_row(graph_rows, "discovery_curve", condition, f"turn_{turn}_discovery_rate")
            for turn in turns
        ]
        estimates = [row["estimate"] for row in rows]
        lower_errors = [row["estimate"] - row["ci_low"] for row in rows]
        upper_errors = [row["ci_high"] - row["estimate"] for row in rows]
        axis.errorbar(
            turns,
            estimates,
            yerr=[lower_errors, upper_errors],
            marker="o",
            linewidth=2,
            capsize=5,
            color=CONDITION_COLORS[condition],
            label=CONDITION_LABELS[condition],
        )
        for turn, estimate in zip(turns, estimates):
            axis.text(turn, min(estimate + 0.035, 0.97), f"{estimate:.2f}", ha="center", va="bottom")
    axis.set_xticks(turns)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Conversation turn")
    axis.set_ylabel("Cumulative rate with score >= 0.75")
    axis.set_title("Multi-turn Harmful Discovery Curve")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def generate_figures(metrics_path: Path, graph_values_path: Path, destination: Path) -> list[Path]:
    metrics = load_metrics(metrics_path)
    graph_rows = load_graph_values(graph_values_path)
    validate_graph_values(metrics, graph_rows)
    destination.mkdir(parents=True, exist_ok=True)
    plt = load_matplotlib()
    outputs: list[Path] = []
    for figure_name, metric, title, ylabel, filename in BAR_SPECS:
        rows = [select_graph_row(graph_rows, figure_name, condition, metric) for condition in CONDITIONS]
        path = destination / filename
        save_bar_chart(plt, rows, title, ylabel, path)
        outputs.append(path)
    discovery_path = destination / OUTPUT_FILENAMES[3]
    save_discovery_chart(plt, graph_rows, discovery_path)
    outputs.append(discovery_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=existing_file, required=True, help="Public aggregate metrics CSV.")
    parser.add_argument(
        "--graph-values",
        type=existing_file,
        required=True,
        help="Public verified estimates and confidence intervals CSV.",
    )
    parser.add_argument("--output-dir", type=output_dir, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        outputs = generate_figures(args.metrics, args.graph_values, args.output_dir)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Generated {len(outputs)} public aggregate figures in {args.output_dir}")


if __name__ == "__main__":
    main()
