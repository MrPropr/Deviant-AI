"""Create public-safe figures for token-probability analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.analyze_token_probabilities import (
    COMPARISONS,
    CONDITIONS,
    METRICS,
)


SCRIPT_VERSION = "1.0.0"
FIGURE_DPI = 300
FIGURE_SIZE = (7.2, 4.8)
MINIMUM_WIDTH = 1800
MINIMUM_HEIGHT = 1200

SUMMARY_FIELDS = (
    "subset",
    "condition",
    "metric",
    "n",
    "mean",
    "median",
    "ci_low",
    "ci_high",
)

PAIRED_TEST_FIELDS = (
    "comparison",
    "left_condition",
    "right_condition",
    "subset",
    "metric",
    "n_pairs",
    "left_mean",
    "right_mean",
    "mean_difference",
    "median_difference",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "permutation_p_value",
    "permutation_method",
    "wilcoxon_statistic",
    "wilcoxon_p_value",
)

CORRELATION_FIELDS = (
    "analysis_type",
    "group",
    "comparison",
    "metric",
    "n",
    "spearman_rho",
    "p_value",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
)

PRIVATE_PLOT_FIELDS = (
    "label",
    "comparison",
    "harmfulness_delta",
    "left_mean_token_logprob",
    "right_mean_token_logprob",
    "delta_mean_token_logprob",
)

SUMMARY_SUBSETS = {
    "all",
    "harmful",
    "benign",
    "harmful_refusal",
    "harmful_unsafe",
    "finish_length",
    "finish_non_length",
}

PAIRED_SUBSETS = {
    "all",
    "harmful",
    "benign",
    "both_refusal",
    "both_unsafe",
    "either_length",
    "both_non_length",
}

COMPARISON_NAMES = tuple(
    comparison
    for comparison, _left, _right in COMPARISONS
)

CONDITION_LABELS = {
    "direct": "Direct",
    "polite": "Polite",
    "multi_turn": "Multi-turn",
    "polite_multi_turn": "Polite\nmulti-turn",
}

CONDITION_COLORS = (
    "#2F5597",
    "#2A7F62",
    "#C45A2D",
    "#7A5195",
)

CONDITION_MARKERS = (
    "o",
    "s",
    "D",
    "^",
)

AGGREGATE_SPECS = (
    {
        "filename": "qwen_mean_token_logprob.png",
        "metric": "mean_token_logprob",
        "title": "Mean Token Log-Probability",
        "ylabel": "Mean token log-probability",
    },
    {
        "filename": "qwen_geometric_mean_probability.png",
        "metric": "geometric_mean_token_probability",
        "title": "Geometric Mean Token Probability",
        "ylabel": "Geometric mean probability",
    },
    {
        "filename": "qwen_first16_logprob.png",
        "metric": "mean_first_16_token_logprob",
        "title": "First 16 Tokens: Mean Log-Probability",
        "ylabel": "Mean first-16 token log-probability",
    },
    {
        "filename": "qwen_entropy.png",
        "metric": "mean_token_entropy",
        "title": "Mean Token Entropy",
        "ylabel": "Mean token entropy (nats)",
    },
)

PAIRED_FILENAME = "qwen_polite_direct_paired.png"
SCATTER_FILENAME = "qwen_harmfulness_delta_scatter.png"


def existing_file(value: str) -> Path:
    path = Path(value)

    if not path.exists():
        raise argparse.ArgumentTypeError(
            f"Input file does not exist: {path}"
        )

    if not path.is_file():
        raise argparse.ArgumentTypeError(
            f"Input path is not a file: {path}"
        )

    return path


def output_file(value: str) -> Path:
    path = Path(value)

    if path.exists() and path.is_dir():
        raise argparse.ArgumentTypeError(
            f"Output path is a directory: {path}"
        )

    return path


def read_csv_checked(
    path: Path,
    expected_fields: Sequence[str],
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        actual_fields = tuple(reader.fieldnames or ())

        if actual_fields != tuple(expected_fields):
            raise ValueError(
                f"Unexpected public CSV schema in {path.name}."
            )

        rows = list(reader)

    if not rows:
        raise ValueError(
            f"Public CSV is empty: {path.name}."
        )

    return rows


def read_private_pairs(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        actual_fields = set(reader.fieldnames or ())
        missing = set(PRIVATE_PLOT_FIELDS) - actual_fields

        if missing:
            raise ValueError(
                "Private paired input lacks required plotting fields."
            )

        rows = list(reader)

    if not rows:
        raise ValueError(
            "Private paired input contains no rows."
        )

    validate_private_pairs(rows)
    return rows


def finite_float(value: object, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected a finite numeric value for {field}."
        ) from exc

    if not math.isfinite(parsed):
        raise ValueError(
            f"Expected a finite numeric value for {field}."
        )

    return parsed


def positive_int(value: object, field: str) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected a positive integer for {field}."
        ) from exc

    if parsed <= 0:
        raise ValueError(
            f"Expected a positive integer for {field}."
        )

    return parsed


def validate_public_rows(
    summary_rows: list[dict[str, str]],
    paired_rows: list[dict[str, str]],
    correlation_rows: list[dict[str, str]],
) -> None:
    for row in summary_rows:
        if row["subset"] not in SUMMARY_SUBSETS:
            raise ValueError("Unknown summary subset.")

        if row["condition"] not in CONDITIONS:
            raise ValueError("Unknown summary condition.")

        if row["metric"] not in METRICS:
            raise ValueError("Unknown summary metric.")

    for row in paired_rows:
        if row["comparison"] not in COMPARISON_NAMES:
            raise ValueError("Unknown paired comparison.")

        if row["left_condition"] not in CONDITIONS:
            raise ValueError("Unknown paired left condition.")

        if row["right_condition"] not in CONDITIONS:
            raise ValueError("Unknown paired right condition.")

        if row["subset"] not in PAIRED_SUBSETS:
            raise ValueError("Unknown paired subset.")

        if row["metric"] not in METRICS:
            raise ValueError("Unknown paired metric.")

    for row in correlation_rows:
        comparison = row["comparison"]

        if comparison and comparison not in COMPARISON_NAMES:
            raise ValueError("Unknown correlation comparison.")

        if row["metric"] not in {
            "mean_token_logprob",
            "mean_first_16_token_logprob",
            "mean_token_entropy",
        }:
            raise ValueError("Unknown correlation metric.")


def load_public_inputs(
    summary_path: Path,
    paired_tests_path: Path,
    correlations_path: Path,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    summary_rows = read_csv_checked(
        summary_path,
        SUMMARY_FIELDS,
    )

    paired_rows = read_csv_checked(
        paired_tests_path,
        PAIRED_TEST_FIELDS,
    )

    correlation_rows = read_csv_checked(
        correlations_path,
        CORRELATION_FIELDS,
    )

    validate_public_rows(
        summary_rows,
        paired_rows,
        correlation_rows,
    )

    return (
        summary_rows,
        paired_rows,
        correlation_rows,
    )


def select_aggregate_rows(
    summary_rows: list[dict[str, str]],
    metric: str,
) -> list[dict[str, object]]:
    selected = [
        row
        for row in summary_rows
        if row["subset"] == "all"
        and row["metric"] == metric
    ]

    rows_by_condition: dict[str, dict[str, str]] = {}

    for row in selected:
        condition = row["condition"]

        if condition in rows_by_condition:
            raise ValueError(
                "Duplicate aggregate summary row."
            )

        rows_by_condition[condition] = row

    if tuple(rows_by_condition) != tuple(CONDITIONS):
        if set(rows_by_condition) != set(CONDITIONS):
            raise ValueError(
                "Aggregate summary lacks required conditions."
            )

    output: list[dict[str, object]] = []

    for condition in CONDITIONS:
        row = rows_by_condition[condition]
        mean = finite_float(row["mean"], "mean")
        ci_low = finite_float(row["ci_low"], "ci_low")
        ci_high = finite_float(row["ci_high"], "ci_high")

        if ci_low > ci_high:
            raise ValueError(
                "Aggregate confidence interval is reversed."
            )

        output.append(
            {
                "condition": condition,
                "n": positive_int(row["n"], "n"),
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    return output


def validate_private_pairs(
    rows: list[dict[str, str]],
) -> None:
    for row in rows:
        if row["label"] not in {"harmful", "benign"}:
            raise ValueError("Unknown private pair label.")

        if row["comparison"] not in COMPARISON_NAMES:
            raise ValueError("Unknown private pair comparison.")

        left = finite_float(
            row["left_mean_token_logprob"],
            "left mean",
        )

        right = finite_float(
            row["right_mean_token_logprob"],
            "right mean",
        )

        delta = finite_float(
            row["delta_mean_token_logprob"],
            "mean difference",
        )

        if not math.isclose(
            delta,
            right - left,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Private pair difference is not right minus left."
            )

        if row["label"] == "harmful":
            finite_float(
                row["harmfulness_delta"],
                "harmfulness difference",
            )


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def prepare_axes(title: str, ylabel: str):
    figure, axis = plt.subplots(
        figsize=FIGURE_SIZE,
        dpi=FIGURE_DPI,
        constrained_layout=True,
    )

    axis.set_title(
        title,
        pad=14,
        weight="semibold",
    )

    axis.set_ylabel(ylabel)
    axis.grid(
        axis="y",
        color="#D8D8D8",
        linewidth=0.7,
        alpha=0.8,
    )

    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    return figure, axis


def add_vertical_padding(
    axis,
    values: Sequence[float],
) -> None:
    low = min(values)
    high = max(values)
    span = high - low

    if span == 0.0:
        span = max(abs(low), 1.0) * 0.2

    padding = span * 0.16
    axis.set_ylim(low - padding, high + padding)


def save_figure(figure, path: Path) -> dict[str, object]:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        path,
        format="png",
        dpi=FIGURE_DPI,
        metadata={
            "Software": (
                f"Deviant-AI plotter {SCRIPT_VERSION}"
            )
        },
    )

    plt.close(figure)

    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError("Generated image is not PNG.")

        width, height = image.size

        if (
            width < MINIMUM_WIDTH
            or height < MINIMUM_HEIGHT
        ):
            raise ValueError(
                "Generated image is below the minimum size."
            )

        pixels = np.asarray(image.convert("RGB"))

        if float(np.var(pixels)) == 0.0:
            raise ValueError("Generated image is blank.")

    digest = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    return {
        "width": width,
        "height": height,
        "dpi": FIGURE_DPI,
        "sha256": digest,
    }


def plot_aggregate(
    rows: list[dict[str, object]],
    spec: dict[str, str],
    output_path: Path,
    source_name: str,
) -> dict[str, object]:
    figure, axis = prepare_axes(
        spec["title"],
        spec["ylabel"],
    )

    x_positions = np.arange(
        len(CONDITIONS),
        dtype=float,
    )

    for index, row in enumerate(rows):
        mean = float(row["mean"])
        lower = mean - float(row["ci_low"])
        upper = float(row["ci_high"]) - mean

        axis.errorbar(
            x_positions[index],
            mean,
            yerr=np.asarray([[lower], [upper]]),
            fmt=CONDITION_MARKERS[index],
            markersize=7.5,
            color=CONDITION_COLORS[index],
            markeredgecolor="white",
            markeredgewidth=0.8,
            ecolor=CONDITION_COLORS[index],
            elinewidth=1.8,
            capsize=5,
            capthick=1.5,
            zorder=3,
        )

    axis.set_xticks(
        x_positions,
        [
            CONDITION_LABELS[condition]
            for condition in CONDITIONS
        ],
    )

    axis.set_xlabel("Prompt condition")

    add_vertical_padding(
        axis,
        [
            float(row["ci_low"])
            for row in rows
        ]
        + [
            float(row["ci_high"])
            for row in rows
        ],
    )

    technical = save_figure(
        figure,
        output_path,
    )

    return {
        "filename": spec["filename"],
        "source_tables": [source_name],
        "subset": "all",
        "conditions": list(CONDITIONS),
        "comparisons": [],
        "metric": spec["metric"],
        "n": [
            int(row["n"])
            for row in rows
        ],
        "plotted_values": rows,
        **technical,
        "verification_status": (
            "generated_and_technically_verified"
        ),
        "created_by_script_version": SCRIPT_VERSION,
    }


def symmetric_offsets(
    count: int,
    width: float = 0.18,
) -> np.ndarray:
    if count <= 1:
        return np.asarray([0.0])

    return np.linspace(
        -width,
        width,
        count,
    )


def plot_polite_direct_pairs(
    private_rows: list[dict[str, str]],
    output_path: Path,
    source_name: str,
) -> dict[str, object]:
    selected = [
        row
        for row in private_rows
        if row["comparison"]
        == "polite_minus_direct"
    ]

    if not selected:
        raise ValueError(
            "No polite-minus-direct private pairs were found."
        )

    figure, axis = prepare_axes(
        "Paired Change: Polite Minus Direct",
        "Change in mean token log-probability",
    )

    styles = {
        "harmful": ("#B23A48", "o", -0.14, "Harmful"),
        "benign": ("#2F6B9A", "s", 0.14, "Benign"),
    }

    all_deltas: list[float] = []
    label_counts: dict[str, int] = {}

    for label in ("harmful", "benign"):
        rows = [
            row
            for row in selected
            if row["label"] == label
        ]

        if not rows:
            raise ValueError(
                "A required private pair label has no rows."
            )

        label_counts[label] = len(rows)

        color, marker, center, display = styles[label]
        deltas = [
            finite_float(
                row["delta_mean_token_logprob"],
                "mean difference",
            )
            for row in rows
        ]

        all_deltas.extend(deltas)

        axis.scatter(
            center + symmetric_offsets(len(rows)),
            deltas,
            s=52,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
            label=display,
            zorder=3,
        )

    if not all_deltas:
        raise ValueError("No private pair differences were found.")

    axis.axhline(
        0.0,
        color="#444444",
        linewidth=1.0,
        linestyle="--",
        zorder=2,
    )

    axis.set_xlim(-0.55, 0.55)
    axis.set_xticks([])
    axis.legend(
        frameon=False,
        loc="best",
    )

    max_abs = max(
        max(abs(value) for value in all_deltas),
        0.05,
    )

    axis.set_ylim(-1.15 * max_abs, 1.15 * max_abs)

    technical = save_figure(
        figure,
        output_path,
    )

    return {
        "filename": PAIRED_FILENAME,
        "source_tables": [source_name],
        "subset": "all",
        "conditions": ["direct", "polite"],
        "comparisons": ["polite_minus_direct"],
        "metric": "delta_mean_token_logprob",
        "n_pairs": len(selected),
        "plotted_values": {
            "label_counts": label_counts,
            "mean_difference": float(np.mean(all_deltas)),
        },
        **technical,
        "verification_status": (
            "generated_and_technically_verified"
        ),
        "created_by_script_version": SCRIPT_VERSION,
    }


def plot_harmfulness_scatter(
    private_rows: list[dict[str, str]],
    output_path: Path,
    source_name: str,
) -> dict[str, object]:
    selected = [
        row
        for row in private_rows
        if row["label"] == "harmful"
        and row["comparison"] in COMPARISON_NAMES
    ]

    if not selected:
        raise ValueError(
            "No harmful private pairs were found."
        )

    figure, axis = prepare_axes(
        "Harmfulness and Token Log-Probability Changes",
        "Change in mean token log-probability",
    )

    axis.set_xlabel("Change in harmfulness score")

    series_styles = {
        "polite_minus_direct": (
            "#2A7F62",
            "o",
            "Polite - direct",
        ),
        "multi_turn_minus_direct": (
            "#C45A2D",
            "s",
            "Multi-turn - direct",
        ),
        "polite_multi_turn_minus_multi_turn": (
            "#7A5195",
            "^",
            "Polite multi-turn - multi-turn",
        ),
    }

    aggregate_series: list[dict[str, object]] = []

    for comparison in COMPARISON_NAMES:
        rows = [
            row
            for row in selected
            if row["comparison"] == comparison
        ]

        if not rows:
            raise ValueError(
                "A required private comparison has no harmful rows."
            )

        x_values = [
            finite_float(
                row["harmfulness_delta"],
                "harmfulness difference",
            )
            for row in rows
        ]

        y_values = [
            finite_float(
                row["delta_mean_token_logprob"],
                "mean difference",
            )
            for row in rows
        ]

        color, marker, display = series_styles[comparison]

        axis.scatter(
            x_values,
            y_values,
            s=50,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.82,
            label=display,
            zorder=3,
        )

        aggregate_series.append(
            {
                "comparison": comparison,
                "n_pairs": len(rows),
                "mean_harmfulness_difference": float(
                    np.mean(x_values)
                ),
                "mean_token_logprob_difference": float(
                    np.mean(y_values)
                ),
            }
        )

    axis.axhline(
        0.0,
        color="#444444",
        linewidth=1.0,
        linestyle="--",
        zorder=2,
    )

    axis.axvline(
        0.0,
        color="#444444",
        linewidth=1.0,
        linestyle="--",
        zorder=2,
    )

    axis.legend(
        frameon=False,
        loc="best",
    )

    technical = save_figure(
        figure,
        output_path,
    )

    return {
        "filename": SCATTER_FILENAME,
        "source_tables": [source_name],
        "subset": "harmful",
        "conditions": list(CONDITIONS),
        "comparisons": list(COMPARISON_NAMES),
        "metric": "delta_mean_token_logprob",
        "n_pairs": len(selected),
        "plotted_values": aggregate_series,
        **technical,
        "verification_status": (
            "generated_and_technically_verified"
        ),
        "created_by_script_version": SCRIPT_VERSION,
    }


def write_manifest(
    path: Path,
    manifest: dict[str, object],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
            + "\n"
        )


def generate_figures(
    summary_path: Path,
    paired_tests_path: Path,
    correlations_path: Path,
    output_dir: Path,
    manifest_path: Path,
    private_pairs_path: Path | None = None,
) -> dict[str, object]:
    configure_matplotlib()

    summary_rows, _paired_rows, _correlation_rows = (
        load_public_inputs(
            summary_path,
            paired_tests_path,
            correlations_path,
        )
    )

    private_rows = (
        read_private_pairs(private_pairs_path)
        if private_pairs_path is not None
        else None
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures: list[dict[str, object]] = []

    for spec in AGGREGATE_SPECS:
        rows = select_aggregate_rows(
            summary_rows,
            spec["metric"],
        )

        figures.append(
            plot_aggregate(
                rows=rows,
                spec=spec,
                output_path=(
                    output_dir / spec["filename"]
                ),
                source_name=summary_path.name,
            )
        )

    missing: list[str] = []

    if private_pairs_path is None:
        missing = [
            PAIRED_FILENAME,
            SCATTER_FILENAME,
        ]

        for filename in missing:
            stale_path = output_dir / filename

            if stale_path.exists():
                stale_path.unlink()
    else:
        figures.append(
            plot_polite_direct_pairs(
                private_rows=private_rows,
                output_path=(
                    output_dir / PAIRED_FILENAME
                ),
                source_name="private_pair_input",
            )
        )

        figures.append(
            plot_harmfulness_scatter(
                private_rows=private_rows,
                output_path=(
                    output_dir / SCATTER_FILENAME
                ),
                source_name="private_pair_input",
            )
        )

    status = (
        "PASS"
        if not missing
        else "BLOCKED_PRIVATE_INPUT"
    )

    manifest: dict[str, object] = {
        "status": status,
        "created_by_script_version": SCRIPT_VERSION,
        "condition_order": list(CONDITIONS),
        "figures": figures,
        "missing_safe_artifacts": missing,
    }

    write_manifest(
        manifest_path,
        manifest,
    )

    return manifest


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--summary-input",
        type=existing_file,
        required=True,
    )

    parser.add_argument(
        "--paired-tests-input",
        type=existing_file,
        required=True,
    )

    parser.add_argument(
        "--correlations-input",
        type=existing_file,
        required=True,
    )

    parser.add_argument(
        "--private-pairs-input",
        type=existing_file,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--manifest-output",
        type=output_file,
        required=True,
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        manifest = generate_figures(
            summary_path=args.summary_input,
            paired_tests_path=(
                args.paired_tests_input
            ),
            correlations_path=(
                args.correlations_input
            ),
            output_dir=args.output_dir,
            manifest_path=args.manifest_output,
            private_pairs_path=(
                args.private_pairs_input
            ),
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Status: {manifest['status']}")
    print(
        f"Public figures generated: "
        f"{len(manifest['figures'])}."
    )

    missing = manifest["missing_safe_artifacts"]

    if missing:
        print(
            "Missing safe artifacts: "
            + ", ".join(missing)
        )

        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
