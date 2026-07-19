from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import src.plot_token_probabilities as plot_module
from src.analyze_token_probabilities import (
    COMPARISONS,
    CONDITIONS,
)
from src.plot_token_probabilities import (
    AGGREGATE_SPECS,
    CORRELATION_FIELDS,
    PAIRED_FILENAME,
    PAIRED_TEST_FIELDS,
    SCATTER_FILENAME,
    SUMMARY_FIELDS,
    generate_figures,
    load_public_inputs,
    select_aggregate_rows,
)


PRIVATE_TERMS = {
    "scenario_id",
    "response_id",
    "prompt",
    "response_text",
    "judge_notes",
    "raw_output",
    "conversation_text",
    "token_logprobs",
    "logits",
}


def write_csv(
    path: Path,
    fieldnames: tuple[str, ...] | list[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def make_public_inputs(
    root: Path,
) -> tuple[Path, Path, Path]:
    summary_path = root / "summary.csv"
    paired_path = root / "paired.csv"
    correlations_path = root / "correlations.csv"

    summary_rows: list[dict[str, object]] = []

    for metric_index, spec in enumerate(
        AGGREGATE_SPECS,
        start=1,
    ):
        for condition_index, condition in enumerate(CONDITIONS):
            mean = metric_index + condition_index / 10
            summary_rows.append(
                {
                    "subset": "all",
                    "condition": condition,
                    "metric": spec["metric"],
                    "n": 20,
                    "mean": mean,
                    "median": mean,
                    "ci_low": mean - 0.05,
                    "ci_high": mean + 0.05,
                }
            )

    paired_rows = [
        {
            "comparison": "polite_minus_direct",
            "left_condition": "direct",
            "right_condition": "polite",
            "subset": "all",
            "metric": "mean_token_logprob",
            "n_pairs": 20,
            "left_mean": -0.4,
            "right_mean": -0.3,
            "mean_difference": 0.1,
            "median_difference": 0.1,
            "bootstrap_ci_low": 0.0,
            "bootstrap_ci_high": 0.2,
            "permutation_p_value": 0.5,
            "permutation_method": "synthetic",
            "wilcoxon_statistic": 1.0,
            "wilcoxon_p_value": 0.5,
        }
    ]

    correlation_rows = [
        {
            "analysis_type": "paired_delta",
            "group": "harmful",
            "comparison": "polite_minus_direct",
            "metric": "mean_token_logprob",
            "n": 10,
            "spearman_rho": 0.1,
            "p_value": 0.5,
            "bootstrap_ci_low": -0.2,
            "bootstrap_ci_high": 0.3,
        }
    ]

    write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    write_csv(paired_path, PAIRED_TEST_FIELDS, paired_rows)
    write_csv(
        correlations_path,
        CORRELATION_FIELDS,
        correlation_rows,
    )

    return summary_path, paired_path, correlations_path


def make_private_fixture(root: Path) -> Path:
    path = root / "synthetic_private_pairs.csv"
    fieldnames = [
        "scenario_id",
        "label",
        "comparison",
        "harmfulness_delta",
        "left_mean_token_logprob",
        "right_mean_token_logprob",
        "delta_mean_token_logprob",
    ]

    rows: list[dict[str, object]] = []

    for comparison_index, (
        comparison,
        _left,
        _right,
    ) in enumerate(COMPARISONS):
        for index in range(4):
            label = "harmful" if index < 2 else "benign"
            left = -0.8 + index * 0.1
            delta = (
                0.04
                + comparison_index * 0.03
                + index * 0.01
            )
            right = left + delta

            rows.append(
                {
                    "scenario_id": f"synthetic_{index}",
                    "label": label,
                    "comparison": comparison,
                    "harmfulness_delta": (
                        -0.25 + index * 0.25
                        if label == "harmful"
                        else ""
                    ),
                    "left_mean_token_logprob": left,
                    "right_mean_token_logprob": right,
                    "delta_mean_token_logprob": right - left,
                }
            )

    write_csv(path, fieldnames, rows)
    return path


def assert_safe_manifest(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            assert key not in PRIVATE_TERMS
            assert_safe_manifest(child)
    elif isinstance(value, list):
        for child in value:
            assert_safe_manifest(child)
    elif isinstance(value, str):
        lowered = value.lower()
        assert not any(term in lowered for term in PRIVATE_TERMS)


def assert_png(path: Path) -> None:
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.width >= 1800
        assert image.height >= 1200
        pixels = np.asarray(image.convert("RGB"))
        assert float(np.var(pixels)) > 0.0


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def use_fast_render(monkeypatch) -> None:
    monkeypatch.setattr(
        plot_module,
        "FIGURE_SIZE",
        (3.0, 2.0),
    )
    monkeypatch.setattr(
        plot_module,
        "FIGURE_DPI",
        100,
    )
    monkeypatch.setattr(
        plot_module,
        "MINIMUM_WIDTH",
        250,
    )
    monkeypatch.setattr(
        plot_module,
        "MINIMUM_HEIGHT",
        150,
    )


def test_reads_public_csvs_and_selects_fixed_order(
    tmp_path: Path,
) -> None:
    paths = make_public_inputs(tmp_path)
    summary_rows, paired_rows, correlation_rows = (
        load_public_inputs(*paths)
    )

    selected = select_aggregate_rows(
        summary_rows,
        "mean_token_logprob",
    )

    assert [row["condition"] for row in selected] == list(
        CONDITIONS
    )
    assert all(row["n"] == 20 for row in selected)
    assert len(paired_rows) == 1
    assert len(correlation_rows) == 1


def test_selects_only_all_subset_and_requested_metric(
    tmp_path: Path,
) -> None:
    summary_path, paired_path, correlation_path = (
        make_public_inputs(tmp_path)
    )

    with summary_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUMMARY_FIELDS,
            lineterminator="\n",
        )
        writer.writerow(
            {
                "subset": "harmful",
                "condition": "direct",
                "metric": "mean_token_logprob",
                "n": 10,
                "mean": 99,
                "median": 99,
                "ci_low": 98,
                "ci_high": 100,
            }
        )

    summary_rows, _, _ = load_public_inputs(
        summary_path,
        paired_path,
        correlation_path,
    )

    selected = select_aggregate_rows(
        summary_rows,
        "mean_token_logprob",
    )

    assert len(selected) == 4
    assert max(float(row["mean"]) for row in selected) < 10


def test_missing_private_input_generates_only_four_figures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    use_fast_render(monkeypatch)
    paths = make_public_inputs(tmp_path / "inputs")
    output_dir = tmp_path / "figures"
    manifest_path = output_dir / "figure_manifest.json"

    manifest = generate_figures(
        *paths,
        output_dir=output_dir,
        manifest_path=manifest_path,
    )

    assert manifest["status"] == "BLOCKED_PRIVATE_INPUT"
    assert len(manifest["figures"]) == 4
    assert not (output_dir / PAIRED_FILENAME).exists()
    assert not (output_dir / SCATTER_FILENAME).exists()
    assert set(manifest["missing_safe_artifacts"]) == {
        PAIRED_FILENAME,
        SCATTER_FILENAME,
    }
    assert_safe_manifest(manifest)


def test_private_fixture_generates_six_safe_pngs(
    tmp_path: Path,
) -> None:
    paths = make_public_inputs(tmp_path / "inputs")
    private_path = make_private_fixture(tmp_path / "private")
    output_dir = tmp_path / "figures"
    manifest_path = output_dir / "figure_manifest.json"

    manifest = generate_figures(
        *paths,
        output_dir=output_dir,
        manifest_path=manifest_path,
        private_pairs_path=private_path,
    )

    assert manifest["status"] == "PASS"
    assert len(manifest["figures"]) == 6
    assert manifest["missing_safe_artifacts"] == []

    paired = next(
        item
        for item in manifest["figures"]
        if item["filename"] == PAIRED_FILENAME
    )
    assert paired["comparisons"] == ["polite_minus_direct"]
    assert paired["n_pairs"] == 4
    assert paired["plotted_values"]["label_counts"] == {
        "harmful": 2,
        "benign": 2,
    }

    for item in manifest["figures"]:
        assert_png(output_dir / item["filename"])

    assert_safe_manifest(manifest)

    with Image.open(output_dir / PAIRED_FILENAME) as image:
        metadata = json.dumps(image.info).lower()
        assert not any(term in metadata for term in PRIVATE_TERMS)


def test_private_fixture_requires_right_minus_left(
    tmp_path: Path,
) -> None:
    paths = make_public_inputs(tmp_path / "inputs")
    private_path = make_private_fixture(tmp_path / "private")
    rows = list(csv.DictReader(private_path.open(encoding="utf-8")))
    rows[0]["delta_mean_token_logprob"] = "9.0"
    write_csv(private_path, list(rows[0]), rows)

    try:
        generate_figures(
            *paths,
            output_dir=tmp_path / "figures",
            manifest_path=tmp_path / "manifest.json",
            private_pairs_path=private_path,
        )
    except ValueError as exc:
        assert "right minus left" in str(exc)
    else:
        raise AssertionError("Invalid pair difference was accepted.")


def test_generation_is_deterministic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    use_fast_render(monkeypatch)
    paths = make_public_inputs(tmp_path / "inputs")
    private_path = make_private_fixture(tmp_path / "private")
    hashes: list[dict[str, str]] = []
    manifests: list[bytes] = []

    for run in ("first", "second"):
        output_dir = tmp_path / run
        manifest_path = output_dir / "figure_manifest.json"

        manifest = generate_figures(
            *paths,
            output_dir=output_dir,
            manifest_path=manifest_path,
            private_pairs_path=private_path,
        )

        hashes.append(
            {
                item["filename"]: file_hash(
                    output_dir / item["filename"]
                )
                for item in manifest["figures"]
            }
        )
        manifests.append(manifest_path.read_bytes())

    assert hashes[0] == hashes[1]
    assert manifests[0] == manifests[1]


def test_text_artifacts_use_lf_without_trailing_whitespace() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "src" / "plot_token_probabilities.py",
        root / "tests" / "test_plot_token_probabilities.py",
        root / "figures" / "token_probabilities" / "README.md",
        root
        / "figures"
        / "token_probabilities"
        / "figure_manifest.json",
    ]

    for path in paths:
        raw = path.read_bytes()
        assert b"\r\n" not in raw

        text = raw.decode("utf-8")
        assert all(
            line == line.rstrip()
            for line in text.splitlines()
        )
