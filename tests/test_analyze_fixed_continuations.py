from __future__ import annotations

import csv
import json

import pytest

from src.analyze_fixed_continuations import (
    ANALYSIS_METRICS,
    PAIRED_FIELDS,
    SUMMARY_FIELDS,
    build_paired_rows,
    build_summary_rows,
    parse_args,
    run,
    validate_detail_rows,
)
from src.analyze_token_probabilities import CONDITIONS, METRICS
from src.run_local_model import DEFAULT_MODEL_ID


CONDITION_VALUES = {
    "direct": 1.0,
    "polite": 2.0,
    "multi_turn": 4.0,
    "polite_multi_turn": 6.0,
}

PRIVATE_PUBLIC_FORBIDDEN = {
    "scenario_id",
    "fixed_continuation",
    "prompt",
    "messages",
    "response_text",
    "judge_notes",
}


def make_detail_rows(scenario_count: int = 3) -> list[dict]:
    rows: list[dict] = []
    for scenario_index in range(scenario_count):
        label = "harmful" if scenario_index % 2 == 0 else "benign"
        for condition in CONDITIONS:
            base_value = CONDITION_VALUES[condition] + scenario_index / 10.0
            row = {
                "scenario_id": f"synthetic_{scenario_index}",
                "label": label,
                "anchor_condition": "direct",
                "condition": condition,
                "model_id": DEFAULT_MODEL_ID,
                "run_id": "synthetic_run",
                "continuation_token_count": 12,
                "status": "dry_run",
            }
            for metric_index, metric in enumerate(METRICS):
                row[metric] = base_value + metric_index / 100.0
            rows.append(row)
    return rows


def write_jsonl(path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_summary_uses_input_counts_not_hardcoded_sizes() -> None:
    scenario_map = validate_detail_rows(make_detail_rows(3))
    summary = build_summary_rows(
        scenario_map,
        bootstrap_iterations=100,
        base_seed=42,
    )
    selected = {
        (row["subset"], row["condition"]): row["n"]
        for row in summary
        if row["metric"] == "mean_token_logprob"
    }
    assert selected[("all", "direct")] == 3
    assert selected[("harmful", "direct")] == 2
    assert selected[("benign", "direct")] == 1
    assert all(
        selected[("all", condition)] == 3
        for condition in CONDITIONS
    )


def test_paired_differences_use_right_minus_left() -> None:
    scenario_map = validate_detail_rows(make_detail_rows(3))
    paired = build_paired_rows(
        scenario_map,
        bootstrap_iterations=100,
        permutation_iterations=100,
        base_seed=42,
    )
    selected = {
        row["comparison"]: row
        for row in paired
        if row["subset"] == "all" and row["metric"] == "mean_token_logprob"
    }
    assert selected["polite_minus_direct"]["mean_difference"] == 1.0
    assert selected["multi_turn_minus_direct"]["mean_difference"] == 3.0
    assert (
        selected["polite_multi_turn_minus_multi_turn"]["mean_difference"]
        == 2.0
    )
    assert selected["polite_minus_direct"]["left_condition"] == "direct"
    assert selected["polite_minus_direct"]["right_condition"] == "polite"


def test_bootstrap_and_permutation_are_reproducible() -> None:
    scenario_map = validate_detail_rows(make_detail_rows(17))
    first = build_paired_rows(
        scenario_map,
        bootstrap_iterations=200,
        permutation_iterations=200,
        base_seed=42,
    )
    second = build_paired_rows(
        scenario_map,
        bootstrap_iterations=200,
        permutation_iterations=200,
        base_seed=42,
    )
    assert first == second


def test_summary_bootstrap_is_reproducible() -> None:
    scenario_map = validate_detail_rows(make_detail_rows(5))
    first = build_summary_rows(
        scenario_map,
        bootstrap_iterations=200,
        base_seed=42,
    )
    second = build_summary_rows(
        scenario_map,
        bootstrap_iterations=200,
        base_seed=42,
    )
    assert first == second


def test_public_csv_schemas_exclude_private_fields(tmp_path) -> None:
    input_path = tmp_path / "private_detail.jsonl"
    summary_path = tmp_path / "summary.csv"
    paired_path = tmp_path / "paired.csv"
    write_jsonl(input_path, make_detail_rows(3))
    args = parse_args(
        [
            "--input",
            str(input_path),
            "--summary-output",
            str(summary_path),
            "--paired-output",
            str(paired_path),
            "--bootstrap-iterations",
            "100",
            "--permutation-iterations",
            "100",
        ]
    )
    result = run(args)
    assert result == {
        "scenarios": 3,
        "summary_rows": 3 * len(CONDITIONS) * len(ANALYSIS_METRICS),
        "paired_rows": 3 * 3 * len(ANALYSIS_METRICS),
    }

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary_reader = csv.DictReader(handle)
        assert tuple(summary_reader.fieldnames or ()) == SUMMARY_FIELDS
        summary_rows = list(summary_reader)
    with paired_path.open("r", encoding="utf-8", newline="") as handle:
        paired_reader = csv.DictReader(handle)
        assert tuple(paired_reader.fieldnames or ()) == PAIRED_FIELDS
        paired_rows = list(paired_reader)

    assert summary_rows and paired_rows
    assert not (set(summary_rows[0]) & PRIVATE_PUBLIC_FORBIDDEN)
    assert not (set(paired_rows[0]) & PRIVATE_PUBLIC_FORBIDDEN)
    assert all("synthetic_" not in value for row in summary_rows for value in row.values())
    assert all("synthetic_" not in value for row in paired_rows for value in row.values())


def test_harmful_and_benign_paired_subsets_are_present() -> None:
    scenario_map = validate_detail_rows(make_detail_rows(4))
    paired = build_paired_rows(
        scenario_map,
        bootstrap_iterations=100,
        permutation_iterations=100,
        base_seed=42,
    )
    selected = {
        row["subset"]: row["n_pairs"]
        for row in paired
        if (
            row["comparison"] == "multi_turn_minus_direct"
            and row["metric"] == "mean_token_logprob"
        )
    }
    assert selected == {"all": 4, "harmful": 2, "benign": 2}


def test_rejects_inconsistent_continuation_token_counts() -> None:
    rows = make_detail_rows(2)
    rows[1]["continuation_token_count"] = 13
    with pytest.raises(ValueError, match="token counts differ across conditions"):
        validate_detail_rows(rows)


def test_continuation_token_count_is_analyzed_but_not_exposed_per_scenario() -> None:
    scenario_map = validate_detail_rows(make_detail_rows(2))
    summary = build_summary_rows(
        scenario_map,
        bootstrap_iterations=100,
        base_seed=42,
    )
    selected = next(
        row
        for row in summary
        if (
            row["subset"] == "all"
            and row["condition"] == "direct"
            and row["metric"] == "continuation_token_count"
        )
    )
    assert selected["n"] == 2
    assert selected["mean"] == 12.0



def test_default_public_run_suppresses_small_cells(
    tmp_path,
) -> None:
    input_path = tmp_path / "private_detail.jsonl"
    summary_path = tmp_path / "summary.csv"
    paired_path = tmp_path / "paired.csv"

    write_jsonl(
        input_path,
        make_detail_rows(3),
    )

    args = parse_args(
        [
            "--input",
            str(input_path),
            "--summary-output",
            str(summary_path),
            "--paired-output",
            str(paired_path),
            "--bootstrap-iterations",
            "100",
            "--permutation-iterations",
            "100",
        ]
    )

    run(args)

    with summary_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        summary_rows = list(csv.DictReader(handle))

    with paired_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        paired_rows = list(csv.DictReader(handle))

    summary_row = next(
        row
        for row in summary_rows
        if (
            row["subset"] == "all"
            and row["condition"] == "direct"
            and row["metric"] == "mean_token_logprob"
        )
    )

    assert summary_row["n"] == "3"
    assert summary_row["mean"] == ""
    assert summary_row["standard_deviation"] == ""
    assert summary_row["ci_low"] == ""
    assert summary_row["ci_high"] == ""

    paired_row = next(
        row
        for row in paired_rows
        if (
            row["subset"] == "all"
            and row["comparison"] == "polite_minus_direct"
            and row["metric"] == "mean_token_logprob"
        )
    )

    assert paired_row["n_pairs"] == "3"
    assert paired_row["mean_difference"] == ""
    assert paired_row["ci_low"] == ""
    assert paired_row["ci_high"] == ""
    assert paired_row["permutation_p_value"] == ""
    assert paired_row["wilcoxon_p_value"] == ""
