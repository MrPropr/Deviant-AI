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
    is_final_turn: bool = True,
) -> dict:
    row = {
        "condition": condition,
        "is_final_turn": is_final_turn,
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


def make_multiturn_fixture() -> list[dict]:
    return [
        make_row(
            "direct",
            "harmful",
            "refusal",
            "eos",
            1.0,
        ),
        make_row(
            "polite",
            "benign",
            "compliant",
            "eos",
            2.0,
        ),
        make_row(
            "multi_turn",
            "harmful",
            "refusal",
            "eos",
            100.0,
            is_final_turn=False,
        ),
        make_row(
            "multi_turn",
            "harmful",
            "refusal",
            "eos",
            200.0,
            is_final_turn=False,
        ),
        make_row(
            "multi_turn",
            "harmful",
            "unsafe",
            "length",
            3.0,
        ),
        make_row(
            "polite_multi_turn",
            "benign",
            "compliant",
            "eos",
            300.0,
            is_final_turn=False,
        ),
        make_row(
            "polite_multi_turn",
            "benign",
            "compliant",
            "eos",
            400.0,
            is_final_turn=False,
        ),
        make_row(
            "polite_multi_turn",
            "benign",
            "compliant",
            "eos",
            4.0,
        ),
    ]


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


def test_run_writes_only_the_public_schema(
    tmp_path,
    capsys,
) -> None:
    input_path = tmp_path / "synthetic.jsonl"
    output_path = tmp_path / "summary.csv"
    write_input(
        input_path,
        make_multiturn_fixture(),
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
        rows = list(reader)
        assert len(rows) == (
            len(SUBSETS) * len(CONDITIONS) * len(METRICS)
        )

    selected = next(
        row
        for row in rows
        if (
            row["subset"] == "all"
            and row["condition"] == "multi_turn"
            and row["metric"] == METRICS[0]
        )
    )
    assert selected["n"] == "1"
    assert selected["mean"] == ""
    assert selected["standard_deviation"] == ""
    assert selected["ci_low"] == ""
    assert selected["ci_high"] == ""

    console = capsys.readouterr().out
    assert "Input records: 8." in console
    assert "Final records used: 4." in console
    assert "Records used: 4." in console
    assert "Aggregate rows written: 224." in console


def test_default_analysis_uses_only_final_turns() -> None:
    output = build_sensitivity_rows(
        validate_rows(make_multiturn_fixture()),
        bootstrap_iterations=100,
        base_seed=42,
    )

    all_rows = {
        row["condition"]: row
        for row in output
        if (
            row["subset"] == "all"
            and row["metric"] == METRICS[0]
        )
    }

    assert {
        condition: row["n"]
        for condition, row in all_rows.items()
    } == {
        condition: 1
        for condition in CONDITIONS
    }
    assert all_rows["direct"]["mean"] == 1.0
    assert all_rows["polite"]["mean"] == 2.0
    assert all_rows["multi_turn"]["mean"] == 3.0
    assert all_rows["polite_multi_turn"]["mean"] == 4.0
    assert all_rows["multi_turn"]["ci_low"] == 3.0
    assert all_rows["multi_turn"]["ci_high"] == 3.0
    assert all_rows["polite_multi_turn"]["ci_low"] == 4.0
    assert all_rows["polite_multi_turn"]["ci_high"] == 4.0


def test_include_intermediate_turns_is_explicit() -> None:
    output = build_sensitivity_rows(
        validate_rows(make_multiturn_fixture()),
        bootstrap_iterations=100,
        base_seed=42,
        include_intermediate_turns=True,
    )

    all_rows = {
        row["condition"]: row
        for row in output
        if (
            row["subset"] == "all"
            and row["metric"] == METRICS[0]
        )
    }

    assert all_rows["direct"]["n"] == 1
    assert all_rows["polite"]["n"] == 1
    assert all_rows["multi_turn"]["n"] == 3
    assert all_rows["polite_multi_turn"]["n"] == 3
    assert all_rows["multi_turn"]["mean"] == 101.0
    assert all_rows["polite_multi_turn"]["mean"] == pytest.approx(
        704.0 / 3.0
    )


def test_validate_only_does_not_create_output(
    tmp_path,
    capsys,
) -> None:
    input_path = tmp_path / "synthetic.jsonl"
    output_path = tmp_path / "unused.csv"
    write_input(
        input_path,
        make_multiturn_fixture(),
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

    assert run(args) == 8
    assert not output_path.exists()
    output = capsys.readouterr().out
    assert "Validation passed for 8 records." in output
    assert "Final records: 4." in output


def test_missing_final_turn_flag_is_rejected() -> None:
    row = make_multiturn_fixture()[0]
    del row["is_final_turn"]

    with pytest.raises(ValueError, match="missing required fields"):
        validate_rows([row])


@pytest.mark.parametrize(
    "invalid_value",
    ["true", "false", 0, 1, "", None],
)
def test_non_boolean_final_turn_flag_is_rejected(
    invalid_value,
) -> None:
    row = make_multiturn_fixture()[0]
    row["is_final_turn"] = invalid_value

    with pytest.raises(ValueError, match="invalid final-turn flag"):
        validate_rows([row])


def test_cli_include_intermediate_turns(tmp_path) -> None:
    input_path = tmp_path / "synthetic.jsonl"
    output_path = tmp_path / "summary.csv"
    write_input(input_path, make_multiturn_fixture())

    args = parse_args(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--include-intermediate-turns",
            "--bootstrap-iterations",
            "100",
        ]
    )
    run(args)

    with output_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    selected = next(
        row
        for row in rows
        if (
            row["subset"] == "all"
            and row["condition"] == "multi_turn"
            and row["metric"] == METRICS[0]
        )
    )
    assert selected["n"] == "3"


def test_bootstrap_is_reproducible_for_same_seed() -> None:
    rows = validate_rows(
        make_multiturn_fixture()
        + [
            make_row(
                condition=condition,
                label="benign",
                response_class="compliant",
                finish_reason="eos",
                value=float(index + 10),
            )
            for index, condition in enumerate(CONDITIONS)
        ]
    )

    first = build_sensitivity_rows(
        rows,
        bootstrap_iterations=200,
        base_seed=42,
    )
    second = build_sensitivity_rows(
        rows,
        bootstrap_iterations=200,
        base_seed=42,
    )

    assert first == second


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
