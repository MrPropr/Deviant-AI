from __future__ import annotations

import csv
import math
from pathlib import Path

from src.plot_expanded_results import (
    CONDITIONS,
    FORBIDDEN_COLUMNS,
    GRAPH_REQUIRED_COLUMNS,
    METRICS_REQUIRED_COLUMNS,
    OUTPUT_FILENAMES,
    generate_figures,
    load_graph_values,
    load_metrics,
    validate_graph_values,
)


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
METRICS_PATH = TABLES / "qwen_expanded_metrics.csv"
GRAPH_VALUES_PATH = TABLES / "qwen_expanded_graph_values.csv"
PUBLIC_TABLES = (
    METRICS_PATH,
    TABLES / "qwen_expanded_confidence_intervals.csv",
    TABLES / "qwen_expanded_paired_statistics.csv",
    TABLES / "qwen_expanded_interaction_effects.csv",
    GRAPH_VALUES_PATH,
)


def table_columns(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return set(csv.DictReader(handle).fieldnames or [])


def test_expanded_tables_are_public_aggregate_data() -> None:
    for path in PUBLIC_TABLES:
        assert path.is_file()
        assert not (table_columns(path) & FORBIDDEN_COLUMNS)
    assert METRICS_REQUIRED_COLUMNS <= table_columns(METRICS_PATH)
    assert GRAPH_REQUIRED_COLUMNS <= table_columns(GRAPH_VALUES_PATH)


def test_expanded_metrics_have_four_conditions_and_unit_interval_values() -> None:
    metrics = load_metrics(METRICS_PATH)
    assert tuple(metrics) == CONDITIONS
    for row in metrics.values():
        for field in METRICS_REQUIRED_COLUMNS - {"condition"}:
            assert 0.0 <= row[field] <= 1.0


def test_expanded_qwen_metric_values() -> None:
    metrics = load_metrics(METRICS_PATH)
    expected = {
        "direct": (0.10, 0.90, 0.00, 0.300, 0.90),
        "polite": (0.10, 0.90, 0.00, 0.325, 0.90),
        "multi_turn": (0.70, 0.20, 0.10, 0.825, 0.90),
        "polite_multi_turn": (0.60, 0.20, 0.20, 0.775, 0.90),
    }
    fields = (
        "strict_asr",
        "refusal_rate",
        "partial_compliance_rate",
        "mean_harmfulness_score",
        "benign_success_rate",
    )
    for condition, expected_values in expected.items():
        assert all(math.isclose(metrics[condition][field], value) for field, value in zip(fields, expected_values))


def test_verified_graph_values_match_aggregate_metrics() -> None:
    metrics = load_metrics(METRICS_PATH)
    graph_rows = load_graph_values(GRAPH_VALUES_PATH)
    validate_graph_values(metrics, graph_rows)
    assert all(0.0 <= row[field] <= 1.0 for row in graph_rows for field in ("estimate", "ci_low", "ci_high"))


def test_plotting_interface_requires_no_private_columns() -> None:
    assert not (METRICS_REQUIRED_COLUMNS & FORBIDDEN_COLUMNS)
    assert not (GRAPH_REQUIRED_COLUMNS & FORBIDDEN_COLUMNS)


def test_expanded_plotter_creates_exactly_four_png_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "figures"
    outputs = generate_figures(METRICS_PATH, GRAPH_VALUES_PATH, output_dir)
    assert {path.name for path in outputs} == set(OUTPUT_FILENAMES)
    assert {path.name for path in output_dir.iterdir()} == set(OUTPUT_FILENAMES)
    assert all(path.stat().st_size > 0 for path in outputs)
