"""Plot public Gemma behavioral and paired aggregate results."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


MODEL_ID = "google/gemma-2-9b-it"
CONDITIONS = ("direct", "polite", "multi_turn", "polite_multi_turn")
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
COMPARISONS = (
    "polite_minus_direct",
    "multi_turn_minus_direct",
    "polite_multi_turn_minus_multi_turn",
)
COMPARISON_LABELS = {
    "polite_minus_direct": "Polite - direct",
    "multi_turn_minus_direct": "Multi-turn - direct",
    "polite_multi_turn_minus_multi_turn": "Polite multi-turn - multi-turn",
}
PAIRED_METRICS = (
    "mean_token_logprob",
    "perplexity",
    "mean_token_entropy",
)
METRIC_LABELS = {
    "mean_token_logprob": "Mean token log-probability",
    "perplexity": "Perplexity",
    "mean_token_entropy": "Mean token entropy",
}
FORBIDDEN_COLUMNS = {
    "prompt",
    "prompt_text",
    "response",
    "response_text",
    "judge_notes",
    "scenario_id",
    "response_id",
    "token_id",
    "token_ids",
    "token_text",
    "token_string",
    "token_strings",
    "messages",
    "logits",
    "raw_output",
    "continuation_text",
    "fixed_continuation",
}
BEHAVIORAL_SPECS = (
    ("strict_asr", "Strict Attack Success Rate", "Strict ASR", "gemma_expanded_asr.png"),
    (
        "mean_harmfulness_score",
        "Mean Harmfulness by Condition",
        "Mean harmfulness score",
        "gemma_expanded_harmfulness.png",
    ),
    ("refusal_rate", "Refusal Rate on Harmful Scenarios", "Refusal rate", "gemma_expanded_refusal.png"),
    ("benign_success_rate", "Benign Success by Condition", "Benign success rate", "gemma_expanded_benign_success.png"),
)


def existing_file(value: str) -> Path:
    path = Path(value)
    if not path.exists() or not path.is_file():
        raise argparse.ArgumentTypeError(f"Input file does not exist: {path}")
    return path


def output_path(value: str) -> Path:
    path = Path(value)
    if path.exists() and path.is_dir():
        raise argparse.ArgumentTypeError(f"Output path is a directory: {path}")
    return path


def read_public_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"Duplicate CSV headers in {path}")
        normalized = {field.lower() for field in fieldnames}
        forbidden = sorted(normalized & FORBIDDEN_COLUMNS)
        if forbidden:
            raise ValueError(f"Private columns are not allowed: {', '.join(forbidden)}")
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(f"Missing aggregate columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No aggregate rows found in {path}")
    return rows


def finite_float(value: object, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be numeric") from None
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def load_behavioral_metrics(path: Path) -> dict[str, dict[str, Any]]:
    fields = {"model_id", "condition", *(item[0] for item in BEHAVIORAL_SPECS)}
    rows = read_public_csv(path, fields)
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["model_id"] != MODEL_ID:
            raise ValueError("Behavioral CSV contains an unexpected model")
        condition = row["condition"]
        if condition not in CONDITIONS or condition in indexed:
            raise ValueError("Behavioral CSV has invalid or duplicate conditions")
        parsed: dict[str, Any] = dict(row)
        for field, *_rest in BEHAVIORAL_SPECS:
            value = finite_float(row[field], field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be between zero and one")
            parsed[field] = value
        indexed[condition] = parsed
    if tuple(indexed) != CONDITIONS:
        raise ValueError("Behavioral CSV must contain the four ordered conditions")
    return indexed


def load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("A working matplotlib installation is required") from exc
    return plt


def save_behavioral_bar(
    plt: Any,
    metrics: dict[str, dict[str, Any]],
    field: str,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    values = [metrics[condition][field] for condition in CONDITIONS]
    positions = list(range(len(CONDITIONS)))
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.bar(
        positions,
        values,
        color=[CONDITION_COLORS[condition] for condition in CONDITIONS],
        edgecolor="#333333",
        linewidth=0.6,
    )
    for position, value in zip(positions, values):
        axis.text(position, min(value + 0.035, 0.97), f"{value:.3f}", ha="center", va="bottom")
    axis.set_xticks(positions, [CONDITION_LABELS[condition] for condition in CONDITIONS])
    axis.set_ylim(0, 1)
    axis.set_ylabel(ylabel)
    axis.set_title(f"Gemma 2 9B: {title}")
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def generate_behavioral_figures(metrics_path: Path, output_dir: Path) -> list[Path]:
    metrics = load_behavioral_metrics(metrics_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = load_matplotlib()
    outputs: list[Path] = []
    for field, title, ylabel, filename in BEHAVIORAL_SPECS:
        path = output_dir / filename
        save_behavioral_bar(plt, metrics, field, title, ylabel, path)
        outputs.append(path)
    return outputs


def load_harmful_paired_rows(path: Path, analysis: str) -> list[dict[str, Any]]:
    if analysis == "token_probability":
        low_field, high_field = "bootstrap_ci_low", "bootstrap_ci_high"
    elif analysis == "fixed_continuation":
        low_field, high_field = "ci_low", "ci_high"
    else:
        raise ValueError(f"Unknown paired analysis: {analysis}")
    required = {
        "subset",
        "comparison",
        "metric",
        "n_pairs",
        "mean_difference",
        low_field,
        high_field,
    }
    rows = read_public_csv(path, required)
    selected: list[dict[str, Any]] = []
    for comparison in COMPARISONS:
        for metric in PAIRED_METRICS:
            matches = [
                row
                for row in rows
                if row["subset"] == "harmful"
                and row["comparison"] == comparison
                and row["metric"] == metric
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one harmful row for {comparison}/{metric}")
            row: dict[str, Any] = dict(matches[0])
            row["n_pairs"] = int(row["n_pairs"])
            if row["n_pairs"] < 5:
                raise ValueError("Paired figure requires at least five pairs per cell")
            row["mean_difference"] = finite_float(row["mean_difference"], "mean_difference")
            row["ci_low"] = finite_float(row[low_field], low_field)
            row["ci_high"] = finite_float(row[high_field], high_field)
            if not row["ci_low"] <= row["mean_difference"] <= row["ci_high"]:
                raise ValueError("Confidence interval does not contain the mean difference")
            selected.append(row)
    return selected


def generate_paired_figure(input_path: Path, output: Path, analysis: str) -> Path:
    rows = load_harmful_paired_rows(input_path, analysis)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt = load_matplotlib()
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    for axis, metric in zip(axes, PAIRED_METRICS):
        metric_rows = [row for row in rows if row["metric"] == metric]
        positions = list(range(len(COMPARISONS)))
        means = [row["mean_difference"] for row in metric_rows]
        lower = [row["mean_difference"] - row["ci_low"] for row in metric_rows]
        upper = [row["ci_high"] - row["mean_difference"] for row in metric_rows]
        axis.errorbar(
            positions,
            means,
            yerr=[lower, upper],
            fmt="o",
            color="#4C78A8",
            ecolor="#333333",
            capsize=5,
            markersize=7,
        )
        axis.axhline(0, color="#666666", linewidth=1, linestyle="--")
        axis.set_xticks(positions, [COMPARISON_LABELS[item] for item in COMPARISONS], rotation=18, ha="right")
        axis.set_ylabel("Mean difference (right - left)")
        axis.set_title(METRIC_LABELS[metric])
        axis.grid(axis="y", alpha=0.2)
        for position, row in zip(positions, metric_rows):
            is_last = position == positions[-1]
            axis.annotate(
                f"n={row['n_pairs']}",
                (position, row["mean_difference"]),
                xytext=(-5 if is_last else 5, 6),
                textcoords="offset points",
                ha="right" if is_last else "left",
                fontsize=8,
            )
    analysis_label = (
        "Generated-continuation teacher forcing"
        if analysis == "token_probability"
        else "Fixed-continuation teacher forcing"
    )
    figure.suptitle(f"Gemma 2 9B: {analysis_label}, harmful scenarios")
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    behavioral = subparsers.add_parser("behavioral")
    behavioral.add_argument("--input", type=existing_file, required=True)
    behavioral.add_argument("--output-dir", type=Path, default=Path("figures"))

    for command in ("token-probability", "fixed-continuation"):
        paired = subparsers.add_parser(command)
        paired.add_argument("--input", type=existing_file, required=True)
        paired.add_argument("--output", type=output_path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "behavioral":
            outputs = generate_behavioral_figures(args.input, args.output_dir)
        else:
            analysis = args.command.replace("-", "_")
            outputs = [generate_paired_figure(args.input, args.output, analysis)]
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Generated {len(outputs)} public aggregate Gemma figure(s).")
    return 0


if __name__ == "__main__":
    main()
