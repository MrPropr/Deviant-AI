from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from src.analyze_token_probabilities import (
    CONDITIONS,
    FORBIDDEN_FIELDS,
    METRICS,
)
from src.analyze_token_probability_sensitivity import (
    OUTPUT_FIELDS,
    SUBSETS,
    build_sensitivity_rows,
    parse_args,
    run,
    validate_rows,
)


def make_row(
    condition: str,
    label: str,
    response_class: str,
    finish_reason: str,
    value: float,
) -> dict:
    row = {
        "condition": condition,
        "label": label,
        "response_class": response_class,
        "finish_reason": finish_reason,
    }

    for metric in METRICS:
        row[metric] = value

    return row


def write_input(path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_builds_every_public_subset_and_metric() -> None:
    rows = [
        make_row(
            condition=condition,
            label=(
                "harmful"
                if index % 2 == 0
                else "benign"
            ),
            response_class=(
                "refusal"
                if index % 2 == 0
                else "compliant"
            ),
            finish_reason=(
                "length"
                if index == 0
                else "eos"
            ),
            value=float(index + 1),
        )
        for index, condition in enumerate(CONDITIONS)
    ]

    output = build_sensitivity_rows(
        validate_rows(rows),
        bootstrap_iterations=100,
        base_seed=42,
    )

    assert len(output) == (
        len(SUBSETS) * len(CONDITIONS) * len(METRICS)
    )

    length_rows = [
        row
        for row in output
        if (
            row["subset"] == "finish_length"
            and row["metric"] == METRICS[0]
        )
    ]

    assert sum(row["n"] for row in length_rows) == 1


def test_single_value_statistics_are_deterministic() -> None:
    rows = [
        make_row(
            condition="direct",
            label="harmful",
            response_class="unsafe",
            finish_reason="length",
            value=2.5,
        )
    ]

    output = build_sensitivity_rows(
        validate_rows(rows),
        bootstrap_iterations=100,
        base_seed=42,
    )

    selected = next(
        row
        for row in output
        if (
            row["subset"] == "all"
            and row["condition"] == "direct"
            and row["metric"] == METRICS[0]
        )
    )

    assert selected["mean"] == 2.5
    assert selected["standard_deviation"] == 0.0
    assert selected["ci_low"] == 2.5
    assert selected["ci_high"] == 2.5


def test_run_writes_only_the_public_schema(tmp_path) -> None:
    input_path = tmp_path / "synthetic.jsonl"
    output_path = tmp_path / "summary.csv"
    write_input(
        input_path,
        [
            make_row(
                condition="polite",
                label="benign",
                response_class="compliant",
                finish_reason="eos",
                value=1.0,
            )
        ],
    )

    args = parse_args(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--bootstrap-iterations",
            "100",
        ]
    )

    assert run(args) == (
        len(SUBSETS) * len(CONDITIONS) * len(METRICS)
    )

    with output_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == OUTPUT_FIELDS
        assert len(list(reader)) == (
            len(SUBSETS) * len(CONDITIONS) * len(METRICS)
        )


def test_validate_only_does_not_create_output(
    tmp_path,
    capsys,
) -> None:
    input_path = tmp_path / "synthetic.jsonl"
    output_path = tmp_path / "unused.csv"
    write_input(
        input_path,
        [
            make_row(
                condition="multi_turn",
                label="harmful",
                response_class="refusal",
                finish_reason="other",
                value=-1.0,
            )
        ],
    )

    args = parse_args(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--validate-only",
        ]
    )

    assert run(args) == 1
    assert not output_path.exists()
    assert "Validation passed for 1 records." in capsys.readouterr().out


def test_validation_rejects_non_public_fields_without_echoing_data() -> None:
    row = make_row(
        condition="direct",
        label="benign",
        response_class="compliant",
        finish_reason="dry_run",
        value=0.0,
    )
    restricted_field = sorted(FORBIDDEN_FIELDS)[0]
    row[restricted_field] = "[MODEL_RESPONSE_PLACEHOLDER]"

    with pytest.raises(ValueError) as exc_info:
        validate_rows([row])

    message = str(exc_info.value)
    assert restricted_field not in message
    assert "[MODEL_RESPONSE_PLACEHOLDER]" not in message


def test_validation_rejects_invalid_labels() -> None:
    row = make_row(
        condition="direct",
        label="unknown",
        response_class="refusal",
        finish_reason="eos",
        value=0.0,
    )

    with pytest.raises(ValueError, match="invalid label"):
        validate_rows([row])


def test_public_findings_match_paired_statistics() -> None:
    root = Path(__file__).resolve().parents[1]
    findings = (
        root / "docs" / "qwen_token_probability_findings.md"
    ).read_text(encoding="utf-8")

    metric_labels = {
        "Mean token log probability": "mean_token_logprob",
        "Geometric mean token probability": (
            "geometric_mean_token_probability"
        ),
        "Perplexity": "perplexity",
        "First-8-token log probability": (
            "mean_first_8_token_logprob"
        ),
        "First-16-token log probability": (
            "mean_first_16_token_logprob"
        ),
        "Mean token entropy": "mean_token_entropy",
        "Top-1 versus top-2 margin": (
            "mean_top1_top2_probability_margin"
        ),
    }
    comparison_order = (
        "polite_minus_direct",
        "multi_turn_minus_direct",
        "polite_multi_turn_minus_multi_turn",
    )
    value = r"(?:-?\d+\.\d+|<0\.0001)"
    pattern = re.compile(
        rf"^\| (?P<label>{'|'.join(map(re.escape, metric_labels))}) "
        rf"\| (?P<mean>{value}) "
        rf"\| \[(?P<low>{value}), (?P<high>{value})\] "
        rf"\| (?P<permutation>{value}) "
        rf"\| (?P<wilcoxon>{value}) \|$",
        re.MULTILINE,
    )
    displayed = list(pattern.finditer(findings))

    with (
        root / "tables" / "qwen_token_probability_paired_tests.csv"
    ).open("r", encoding="utf-8", newline="") as handle:
        source = {
            (row["comparison"], row["metric"]): row
            for row in csv.DictReader(handle)
            if row["subset"] == "all"
        }

    assert len(displayed) == (
        len(comparison_order) * len(metric_labels)
    )

    for index, match in enumerate(displayed):
        comparison = comparison_order[
            index // len(metric_labels)
        ]
        metric = metric_labels[match.group("label")]
        source_row = source[(comparison, metric)]

        checks = {
            "mean": "mean_difference",
            "low": "bootstrap_ci_low",
            "high": "bootstrap_ci_high",
            "permutation": "permutation_p_value",
            "wilcoxon": "wilcoxon_p_value",
        }

        for displayed_name, source_name in checks.items():
            text_value = match.group(displayed_name)
            source_value = float(source_row[source_name])

            if text_value == "<0.0001":
                assert source_value < 0.0001
                continue

            decimals = len(text_value.rsplit(".", maxsplit=1)[1])
            tolerance = 0.5 * (10 ** -decimals) + 1e-12
            assert float(text_value) == pytest.approx(
                source_value,
                abs=tolerance,
            )
