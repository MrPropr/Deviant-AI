from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.image as mpimg
import pytest

from src.plot_gemma_results import (
    BEHAVIORAL_SPECS,
    COMPARISONS,
    CONDITIONS,
    MODEL_ID,
    PAIRED_METRICS,
    generate_behavioral_figures,
    generate_paired_figure,
    read_public_csv,
)


def write_behavioral_csv(path: Path) -> None:
    fields = ["model_id", "condition", *(item[0] for item in BEHAVIORAL_SPECS)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, condition in enumerate(CONDITIONS):
            writer.writerow(
                {
                    "model_id": MODEL_ID,
                    "condition": condition,
                    "strict_asr": index / 10,
                    "mean_harmfulness_score": 0.25 + index / 10,
                    "refusal_rate": 1.0 - index / 10,
                    "benign_success_rate": 0.4 + index / 10,
                }
            )


def test_generates_four_behavioral_figures(tmp_path) -> None:
    source = tmp_path / "aggregate.csv"
    destination = tmp_path / "figures"
    write_behavioral_csv(source)
    outputs = generate_behavioral_figures(source, destination)
    assert [path.name for path in outputs] == [item[3] for item in BEHAVIORAL_SPECS]
    for output in outputs:
        assert output.stat().st_size > 0
        image = mpimg.imread(output)
        assert image.shape[0] > 100
        assert image.shape[1] > 100


def test_rejects_private_csv_columns(tmp_path) -> None:
    source = tmp_path / "unsafe.csv"
    source.write_text("condition,prompt\ndirect,[SANITIZED_PLACEHOLDER]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Private columns"):
        read_public_csv(source, {"condition"})


def test_generates_public_paired_figure(tmp_path) -> None:
    source = tmp_path / "paired.csv"
    fields = [
        "subset",
        "comparison",
        "metric",
        "n_pairs",
        "mean_difference",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
    ]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for comparison_index, comparison in enumerate(COMPARISONS):
            for metric_index, metric in enumerate(PAIRED_METRICS):
                estimate = (comparison_index - 1) / 10 + metric_index / 100
                writer.writerow(
                    {
                        "subset": "harmful",
                        "comparison": comparison,
                        "metric": metric,
                        "n_pairs": 10,
                        "mean_difference": estimate,
                        "bootstrap_ci_low": estimate - 0.05,
                        "bootstrap_ci_high": estimate + 0.05,
                    }
                )
    output = tmp_path / "paired.png"
    generate_paired_figure(source, output, "token_probability")
    image = mpimg.imread(output)
    assert output.stat().st_size > 0
    assert image.shape[0] > 100
    assert image.shape[1] > image.shape[0]


def test_generates_fixed_continuation_paired_figure(tmp_path) -> None:
    source = tmp_path / "fixed.csv"
    fields = [
        "subset",
        "comparison",
        "metric",
        "n_pairs",
        "mean_difference",
        "ci_low",
        "ci_high",
    ]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for comparison_index, comparison in enumerate(COMPARISONS):
            for metric_index, metric in enumerate(PAIRED_METRICS):
                estimate = (comparison_index - 1) / 10 + metric_index / 100
                writer.writerow(
                    {
                        "subset": "harmful",
                        "comparison": comparison,
                        "metric": metric,
                        "n_pairs": 10,
                        "mean_difference": estimate,
                        "ci_low": estimate - 0.05,
                        "ci_high": estimate + 0.05,
                    }
                )
    output = tmp_path / "fixed.png"
    generate_paired_figure(source, output, "fixed_continuation")
    image = mpimg.imread(output)
    assert output.stat().st_size > 0
    assert image.shape[1] > image.shape[0]
