"""Analyze teacher-forced token probabilities without exposing private text."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import stats


CONDITIONS = (
    "direct",
    "polite",
    "multi_turn",
    "polite_multi_turn",
)

METRICS = (
    "mean_token_logprob",
    "geometric_mean_token_probability",
    "perplexity",
    "mean_first_8_token_logprob",
    "mean_first_16_token_logprob",
    "mean_first_32_token_logprob",
    "mean_token_entropy",
    "mean_top1_top2_probability_margin",
)

COMPARISONS = (
    (
        "polite_minus_direct",
        "direct",
        "polite",
    ),
    (
        "multi_turn_minus_direct",
        "direct",
        "multi_turn",
    ),
    (
        "polite_multi_turn_minus_multi_turn",
        "multi_turn",
        "polite_multi_turn",
    ),
)

ALLOWED_HARMFULNESS = {
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
}

FORBIDDEN_FIELDS = {
    "prompt",
    "prompt_text",
    "response",
    "response_text",
    "judge_notes",
    "messages",
    "logits",
    "token_logprobs",
    "top_tokens",
    "top_k_tokens",
    "raw_output",
}

REQUIRED_FIELDS = {
    "response_id",
    "scenario_id",
    "condition",
    "turn_index",
    "label",
    "model_id",
    "run_id",
    "is_final_turn",
    "response_class",
    "finish_reason",
    "hit_max_new_tokens",
    "manual_harmfulness_score",
    *METRICS,
}


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


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path.name}:{line_number}"
                ) from exc

            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL row is not an object at "
                    f"{path.name}:{line_number}"
                )

            rows.append(value)

    return rows


def finite_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected a numeric value, received {value!r}"
        ) from exc

    if not math.isfinite(parsed):
        raise ValueError(
            f"Expected a finite value, received {value!r}"
        )

    return parsed


def validate_input_rows(rows: list[dict]) -> list[dict]:
    if len(rows) != 160:
        raise ValueError(
            f"Expected 160 merged rows, found {len(rows)}"
        )

    all_fields = {
        field
        for row in rows
        for field in row
    }

    forbidden = sorted(
        all_fields & FORBIDDEN_FIELDS
    )

    if forbidden:
        raise ValueError(
            f"Input contains forbidden fields: {forbidden}"
        )

    errors: list[str] = []
    response_ids: set[str] = set()

    for row_number, row in enumerate(rows, start=1):
        missing = sorted(
            REQUIRED_FIELDS - set(row)
        )

        if missing:
            errors.append(
                f"row {row_number}: missing {missing}"
            )
            continue

        response_id = str(
            row.get("response_id", "")
        )

        if not response_id:
            errors.append(
                f"row {row_number}: missing response_id"
            )
        elif response_id in response_ids:
            errors.append(
                f"row {row_number}: duplicate response_id"
            )
        else:
            response_ids.add(response_id)

        condition = str(
            row.get("condition", "")
        )

        if condition not in CONDITIONS:
            errors.append(
                f"{response_id}: invalid condition"
            )

        label = str(
            row.get("label", "")
        )

        if label not in {
            "harmful",
            "benign",
        }:
            errors.append(
                f"{response_id}: invalid label"
            )

        for metric in METRICS:
            try:
                finite_float(row.get(metric))
            except ValueError:
                errors.append(
                    f"{response_id}: invalid {metric}"
                )

        if label == "harmful":
            try:
                harmfulness = finite_float(
                    row.get(
                        "manual_harmfulness_score"
                    )
                )
            except ValueError:
                errors.append(
                    f"{response_id}: missing harmfulness score"
                )
            else:
                if harmfulness not in ALLOWED_HARMFULNESS:
                    errors.append(
                        f"{response_id}: invalid harmfulness score"
                    )

    final_rows = [
        row
        for row in rows
        if row.get("is_final_turn") is True
    ]

    if len(final_rows) != 80:
        errors.append(
            f"expected 80 final rows, found {len(final_rows)}"
        )

    final_counts = Counter(
        str(row.get("condition", ""))
        for row in final_rows
    )

    expected_counts = {
        condition: 20
        for condition in CONDITIONS
    }

    if dict(final_counts) != expected_counts:
        errors.append(
            "final-turn condition counts do not match "
            f"{expected_counts}: {dict(final_counts)}"
        )

    scenario_conditions: dict[
        str,
        set[str],
    ] = defaultdict(set)

    for row in final_rows:
        scenario_conditions[
            str(row["scenario_id"])
        ].add(
            str(row["condition"])
        )

    if len(scenario_conditions) != 20:
        errors.append(
            f"expected 20 scenarios, found "
            f"{len(scenario_conditions)}"
        )

    expected_condition_set = set(CONDITIONS)

    incomplete = [
        scenario_id
        for scenario_id, conditions
        in scenario_conditions.items()
        if conditions != expected_condition_set
    ]

    if incomplete:
        errors.append(
            f"{len(incomplete)} scenarios lack complete pairs"
        )

    if errors:
        preview = "\n".join(
            f"- {error}"
            for error in errors[:20]
        )

        raise ValueError(
            f"Validation found {len(errors)} errors:\n"
            f"{preview}"
        )

    return final_rows


def derived_seed(
    base_seed: int,
    *parts: str,
) -> int:
    text = "::".join(
        [str(base_seed), *parts]
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
    ) % (2**32)


def bootstrap_mean_ci(
    values: list[float],
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.size == 0:
        return math.nan, math.nan

    if array.size == 1:
        value = float(array[0])
        return value, value

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0,
        array.size,
        size=(iterations, array.size),
    )

    means = array[indices].mean(axis=1)

    low, high = np.quantile(
        means,
        [0.025, 0.975],
    )

    return float(low), float(high)


def safe_spearman(
    first: list[float],
    second: list[float],
) -> tuple[float, float]:
    if len(first) < 3:
        return math.nan, math.nan

    first_array = np.asarray(
        first,
        dtype=np.float64,
    )

    second_array = np.asarray(
        second,
        dtype=np.float64,
    )

    if (
        np.all(first_array == first_array[0])
        or np.all(second_array == second_array[0])
    ):
        return math.nan, math.nan

    result = stats.spearmanr(
        first_array,
        second_array,
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


def bootstrap_spearman_ci(
    first: list[float],
    second: list[float],
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    if len(first) < 3:
        return math.nan, math.nan

    first_array = np.asarray(
        first,
        dtype=np.float64,
    )

    second_array = np.asarray(
        second,
        dtype=np.float64,
    )

    rng = np.random.default_rng(seed)

    bootstrapped: list[float] = []

    for _ in range(iterations):
        indices = rng.integers(
            0,
            len(first_array),
            size=len(first_array),
        )

        rho, _ = safe_spearman(
            first_array[indices].tolist(),
            second_array[indices].tolist(),
        )

        if math.isfinite(rho):
            bootstrapped.append(rho)

    if not bootstrapped:
        return math.nan, math.nan

    low, high = np.quantile(
        np.asarray(bootstrapped),
        [0.025, 0.975],
    )

    return float(low), float(high)


def sign_flip_permutation_test(
    differences: list[float],
    iterations: int,
    seed: int,
) -> tuple[float, str]:
    array = np.asarray(
        differences,
        dtype=np.float64,
    )

    if array.size == 0:
        return math.nan, "not_available"

    observed = abs(
        float(array.mean())
    )

    if np.allclose(array, 0.0):
        return 1.0, "all_zero"

    tolerance = 1e-15

    if array.size <= 16:
        signs = np.asarray(
            list(
                itertools.product(
                    (-1.0, 1.0),
                    repeat=array.size,
                )
            ),
            dtype=np.float64,
        )

        permuted = np.abs(
            (signs @ array)
            / array.size
        )

        p_value = float(
            np.mean(
                permuted
                >= observed - tolerance
            )
        )

        return p_value, "exact_sign_flip"

    rng = np.random.default_rng(seed)

    extreme = 0
    completed = 0
    batch_size = 5000

    while completed < iterations:
        current_batch = min(
            batch_size,
            iterations - completed,
        )

        signs = rng.choice(
            (-1.0, 1.0),
            size=(
                current_batch,
                array.size,
            ),
        )

        permuted = np.abs(
            (signs @ array)
            / array.size
        )

        extreme += int(
            np.sum(
                permuted
                >= observed - tolerance
            )
        )

        completed += current_batch

    p_value = (
        extreme + 1
    ) / (
        iterations + 1
    )

    return (
        float(p_value),
        "monte_carlo_sign_flip",
    )


def wilcoxon_test(
    differences: list[float],
) -> tuple[float, float]:
    array = np.asarray(
        differences,
        dtype=np.float64,
    )

    if array.size == 0:
        return math.nan, math.nan

    if np.allclose(array, 0.0):
        return 0.0, 1.0

    result = stats.wilcoxon(
        array,
        zero_method="wilcox",
        alternative="two-sided",
        method="auto",
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


def summary_subset(
    rows: list[dict],
    subset: str,
) -> list[dict]:
    if subset == "all":
        return list(rows)

    if subset == "harmful":
        return [
            row
            for row in rows
            if row["label"] == "harmful"
        ]

    if subset == "benign":
        return [
            row
            for row in rows
            if row["label"] == "benign"
        ]

    if subset == "harmful_refusal":
        return [
            row
            for row in rows
            if (
                row["label"] == "harmful"
                and row["response_class"]
                == "refusal"
            )
        ]

    if subset == "harmful_unsafe":
        return [
            row
            for row in rows
            if (
                row["label"] == "harmful"
                and row["response_class"]
                == "unsafe"
            )
        ]

    if subset == "finish_length":
        return [
            row
            for row in rows
            if row["finish_reason"] == "length"
        ]

    if subset == "finish_non_length":
        return [
            row
            for row in rows
            if row["finish_reason"] != "length"
        ]

    raise ValueError(
        f"Unknown summary subset: {subset}"
    )


def build_summary_rows(
    final_rows: list[dict],
    bootstrap_iterations: int,
    base_seed: int,
) -> list[dict]:
    output: list[dict] = []

    subsets = (
        "all",
        "harmful",
        "benign",
        "harmful_refusal",
        "harmful_unsafe",
        "finish_length",
        "finish_non_length",
    )

    for subset in subsets:
        subset_rows = summary_subset(
            final_rows,
            subset,
        )

        for condition in CONDITIONS:
            condition_rows = [
                row
                for row in subset_rows
                if row["condition"] == condition
            ]

            for metric in METRICS:
                values = [
                    finite_float(row[metric])
                    for row in condition_rows
                ]

                if values:
                    mean_value = float(
                        np.mean(values)
                    )

                    median_value = float(
                        np.median(values)
                    )

                    ci_low, ci_high = (
                        bootstrap_mean_ci(
                            values=values,
                            iterations=(
                                bootstrap_iterations
                            ),
                            seed=derived_seed(
                                base_seed,
                                "summary",
                                subset,
                                condition,
                                metric,
                            ),
                        )
                    )
                else:
                    mean_value = math.nan
                    median_value = math.nan
                    ci_low = math.nan
                    ci_high = math.nan

                output.append(
                    {
                        "subset": subset,
                        "condition": condition,
                        "metric": metric,
                        "n": len(values),
                        "mean": mean_value,
                        "median": median_value,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )

    return output


def build_final_map(
    final_rows: list[dict],
) -> dict[str, dict[str, dict]]:
    result: dict[
        str,
        dict[str, dict],
    ] = defaultdict(dict)

    for row in final_rows:
        result[
            str(row["scenario_id"])
        ][
            str(row["condition"])
        ] = row

    return dict(result)


def build_pair_rows(
    final_rows: list[dict],
) -> list[dict]:
    final_map = build_final_map(
        final_rows
    )

    output: list[dict] = []

    for (
        comparison,
        left_condition,
        right_condition,
    ) in COMPARISONS:
        for scenario_id in sorted(
            final_map
        ):
            conditions = final_map[
                scenario_id
            ]

            left = conditions[
                left_condition
            ]

            right = conditions[
                right_condition
            ]

            left_harmfulness = (
                left.get(
                    "manual_harmfulness_score"
                )
            )

            right_harmfulness = (
                right.get(
                    "manual_harmfulness_score"
                )
            )

            harmfulness_delta = None

            if (
                left["label"] == "harmful"
                and right["label"] == "harmful"
                and left_harmfulness is not None
                and right_harmfulness is not None
            ):
                harmfulness_delta = (
                    finite_float(
                        right_harmfulness
                    )
                    - finite_float(
                        left_harmfulness
                    )
                )

            pair_row = {
                "scenario_id": scenario_id,
                "label": left["label"],
                "comparison": comparison,
                "left_condition": left_condition,
                "right_condition": right_condition,
                "left_response_class": (
                    left["response_class"]
                ),
                "right_response_class": (
                    right["response_class"]
                ),
                "left_finish_reason": (
                    left["finish_reason"]
                ),
                "right_finish_reason": (
                    right["finish_reason"]
                ),
                "either_length": (
                    left["finish_reason"] == "length"
                    or right["finish_reason"] == "length"
                ),
                "both_length": (
                    left["finish_reason"] == "length"
                    and right["finish_reason"] == "length"
                ),
                "harmfulness_delta": (
                    harmfulness_delta
                ),
            }

            for metric in METRICS:
                left_value = finite_float(
                    left[metric]
                )

                right_value = finite_float(
                    right[metric]
                )

                pair_row[
                    f"left_{metric}"
                ] = left_value

                pair_row[
                    f"right_{metric}"
                ] = right_value

                pair_row[
                    f"delta_{metric}"
                ] = (
                    right_value - left_value
                )

            output.append(pair_row)

    return output


def pair_subset(
    rows: list[dict],
    subset: str,
) -> list[dict]:
    if subset == "all":
        return list(rows)

    if subset == "harmful":
        return [
            row
            for row in rows
            if row["label"] == "harmful"
        ]

    if subset == "benign":
        return [
            row
            for row in rows
            if row["label"] == "benign"
        ]

    if subset == "both_refusal":
        return [
            row
            for row in rows
            if (
                row["label"] == "harmful"
                and row[
                    "left_response_class"
                ] == "refusal"
                and row[
                    "right_response_class"
                ] == "refusal"
            )
        ]

    if subset == "both_unsafe":
        return [
            row
            for row in rows
            if (
                row["label"] == "harmful"
                and row[
                    "left_response_class"
                ] == "unsafe"
                and row[
                    "right_response_class"
                ] == "unsafe"
            )
        ]

    if subset == "either_length":
        return [
            row
            for row in rows
            if row["either_length"] is True
        ]

    if subset == "both_non_length":
        return [
            row
            for row in rows
            if row["either_length"] is False
        ]

    raise ValueError(
        f"Unknown pair subset: {subset}"
    )


def build_paired_rows(
    private_pair_rows: list[dict],
    bootstrap_iterations: int,
    permutation_iterations: int,
    base_seed: int,
) -> list[dict]:
    output: list[dict] = []

    subsets = (
        "all",
        "harmful",
        "benign",
        "both_refusal",
        "both_unsafe",
        "either_length",
        "both_non_length",
    )

    grouped: dict[
        str,
        list[dict],
    ] = defaultdict(list)

    for row in private_pair_rows:
        grouped[
            str(row["comparison"])
        ].append(row)

    comparison_lookup = {
        name: (
            left_condition,
            right_condition,
        )
        for (
            name,
            left_condition,
            right_condition,
        ) in COMPARISONS
    }

    for comparison, comparison_rows in (
        grouped.items()
    ):
        (
            left_condition,
            right_condition,
        ) = comparison_lookup[
            comparison
        ]

        for subset in subsets:
            selected_rows = pair_subset(
                comparison_rows,
                subset,
            )

            for metric in METRICS:
                left_values = [
                    finite_float(
                        row[f"left_{metric}"]
                    )
                    for row in selected_rows
                ]

                right_values = [
                    finite_float(
                        row[f"right_{metric}"]
                    )
                    for row in selected_rows
                ]

                differences = [
                    finite_float(
                        row[f"delta_{metric}"]
                    )
                    for row in selected_rows
                ]

                if differences:
                    mean_difference = float(
                        np.mean(differences)
                    )

                    median_difference = float(
                        np.median(differences)
                    )

                    ci_low, ci_high = (
                        bootstrap_mean_ci(
                            values=differences,
                            iterations=(
                                bootstrap_iterations
                            ),
                            seed=derived_seed(
                                base_seed,
                                "paired_bootstrap",
                                comparison,
                                subset,
                                metric,
                            ),
                        )
                    )

                    permutation_p, permutation_method = (
                        sign_flip_permutation_test(
                            differences=differences,
                            iterations=(
                                permutation_iterations
                            ),
                            seed=derived_seed(
                                base_seed,
                                "permutation",
                                comparison,
                                subset,
                                metric,
                            ),
                        )
                    )

                    wilcoxon_stat, wilcoxon_p = (
                        wilcoxon_test(
                            differences
                        )
                    )
                else:
                    mean_difference = math.nan
                    median_difference = math.nan
                    ci_low = math.nan
                    ci_high = math.nan
                    permutation_p = math.nan
                    permutation_method = (
                        "not_available"
                    )
                    wilcoxon_stat = math.nan
                    wilcoxon_p = math.nan

                output.append(
                    {
                        "comparison": comparison,
                        "left_condition": (
                            left_condition
                        ),
                        "right_condition": (
                            right_condition
                        ),
                        "subset": subset,
                        "metric": metric,
                        "n_pairs": len(
                            differences
                        ),
                        "left_mean": (
                            float(
                                np.mean(
                                    left_values
                                )
                            )
                            if left_values
                            else math.nan
                        ),
                        "right_mean": (
                            float(
                                np.mean(
                                    right_values
                                )
                            )
                            if right_values
                            else math.nan
                        ),
                        "mean_difference": (
                            mean_difference
                        ),
                        "median_difference": (
                            median_difference
                        ),
                        "bootstrap_ci_low": (
                            ci_low
                        ),
                        "bootstrap_ci_high": (
                            ci_high
                        ),
                        "permutation_p_value": (
                            permutation_p
                        ),
                        "permutation_method": (
                            permutation_method
                        ),
                        "wilcoxon_statistic": (
                            wilcoxon_stat
                        ),
                        "wilcoxon_p_value": (
                            wilcoxon_p
                        ),
                    }
                )

    return output


def build_correlation_rows(
    final_rows: list[dict],
    private_pair_rows: list[dict],
    bootstrap_iterations: int,
    base_seed: int,
) -> list[dict]:
    output: list[dict] = []

    correlation_metrics = (
        "mean_token_logprob",
        "mean_first_16_token_logprob",
        "mean_token_entropy",
    )

    harmful_rows = [
        row
        for row in final_rows
        if row["label"] == "harmful"
    ]

    groups = {
        "all_harmful_final": harmful_rows,
    }

    for condition in CONDITIONS:
        groups[
            f"harmful_{condition}"
        ] = [
            row
            for row in harmful_rows
            if row["condition"] == condition
        ]

    for group_name, rows in groups.items():
        harmfulness = [
            finite_float(
                row[
                    "manual_harmfulness_score"
                ]
            )
            for row in rows
        ]

        for metric in correlation_metrics:
            values = [
                finite_float(row[metric])
                for row in rows
            ]

            rho, p_value = safe_spearman(
                harmfulness,
                values,
            )

            ci_low, ci_high = (
                bootstrap_spearman_ci(
                    first=harmfulness,
                    second=values,
                    iterations=(
                        bootstrap_iterations
                    ),
                    seed=derived_seed(
                        base_seed,
                        "correlation",
                        group_name,
                        metric,
                    ),
                )
            )

            output.append(
                {
                    "analysis_type": (
                        "response_level"
                    ),
                    "group": group_name,
                    "comparison": "",
                    "metric": metric,
                    "n": len(rows),
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                }
            )

    for (
        comparison,
        _left,
        _right,
    ) in COMPARISONS:
        rows = [
            row
            for row in private_pair_rows
            if (
                row["comparison"]
                == comparison
                and row["label"]
                == "harmful"
                and row[
                    "harmfulness_delta"
                ] is not None
            )
        ]

        harmfulness_deltas = [
            finite_float(
                row["harmfulness_delta"]
            )
            for row in rows
        ]

        for metric in correlation_metrics:
            probability_deltas = [
                finite_float(
                    row[f"delta_{metric}"]
                )
                for row in rows
            ]

            rho, p_value = safe_spearman(
                harmfulness_deltas,
                probability_deltas,
            )

            ci_low, ci_high = (
                bootstrap_spearman_ci(
                    first=harmfulness_deltas,
                    second=probability_deltas,
                    iterations=(
                        bootstrap_iterations
                    ),
                    seed=derived_seed(
                        base_seed,
                        "delta_correlation",
                        comparison,
                        metric,
                    ),
                )
            )

            output.append(
                {
                    "analysis_type": (
                        "paired_delta"
                    ),
                    "group": "harmful",
                    "comparison": comparison,
                    "metric": metric,
                    "n": len(rows),
                    "spearman_rho": rho,
                    "p_value": p_value,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                }
            )

    return output


def format_value(value: object) -> object:
    if isinstance(value, bool):
        return str(value).lower()

    if isinstance(value, float):
        if not math.isfinite(value):
            return ""

        return f"{value:.10f}"

    return value


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: format_value(
                        row.get(field, "")
                    )
                    for field in fieldnames
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input",
        type=existing_file,
        required=True,
    )

    parser.add_argument(
        "--summary-output",
        type=output_file,
        required=True,
    )

    parser.add_argument(
        "--paired-output",
        type=output_file,
        required=True,
    )

    parser.add_argument(
        "--correlation-output",
        type=output_file,
        required=True,
    )

    parser.add_argument(
        "--private-pairs-output",
        type=output_file,
        required=True,
    )

    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--permutation-iterations",
        type=int,
        default=50000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.bootstrap_iterations < 100:
        raise SystemExit(
            "--bootstrap-iterations must be at least 100"
        )

    if args.permutation_iterations < 100:
        raise SystemExit(
            "--permutation-iterations must be at least 100"
        )

    try:
        merged_rows = read_jsonl(
            args.input
        )

        final_rows = validate_input_rows(
            merged_rows
        )

        private_pair_rows = build_pair_rows(
            final_rows
        )

        summary_rows = build_summary_rows(
            final_rows=final_rows,
            bootstrap_iterations=(
                args.bootstrap_iterations
            ),
            base_seed=args.seed,
        )

        paired_rows = build_paired_rows(
            private_pair_rows=(
                private_pair_rows
            ),
            bootstrap_iterations=(
                args.bootstrap_iterations
            ),
            permutation_iterations=(
                args.permutation_iterations
            ),
            base_seed=args.seed,
        )

        correlation_rows = (
            build_correlation_rows(
                final_rows=final_rows,
                private_pair_rows=(
                    private_pair_rows
                ),
                bootstrap_iterations=(
                    args.bootstrap_iterations
                ),
                base_seed=args.seed,
            )
        )

        write_csv(
            args.summary_output,
            summary_rows,
            [
                "subset",
                "condition",
                "metric",
                "n",
                "mean",
                "median",
                "ci_low",
                "ci_high",
            ],
        )

        write_csv(
            args.paired_output,
            paired_rows,
            [
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
            ],
        )

        write_csv(
            args.correlation_output,
            correlation_rows,
            [
                "analysis_type",
                "group",
                "comparison",
                "metric",
                "n",
                "spearman_rho",
                "p_value",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
            ],
        )

        private_pair_fields = [
            "scenario_id",
            "label",
            "comparison",
            "left_condition",
            "right_condition",
            "left_response_class",
            "right_response_class",
            "left_finish_reason",
            "right_finish_reason",
            "either_length",
            "both_length",
            "harmfulness_delta",
        ]

        for metric in METRICS:
            private_pair_fields.extend(
                [
                    f"left_{metric}",
                    f"right_{metric}",
                    f"delta_{metric}",
                ]
            )

        write_csv(
            args.private_pairs_output,
            private_pair_rows,
            private_pair_fields,
        )

        print(
            "Token-probability analysis complete."
        )

        print(
            f"Final responses analyzed: "
            f"{len(final_rows)}."
        )

        print(
            f"Private paired rows: "
            f"{len(private_pair_rows)}."
        )

        print(
            f"Summary rows: "
            f"{len(summary_rows)}."
        )

        print(
            f"Paired-test rows: "
            f"{len(paired_rows)}."
        )

        print(
            f"Correlation rows: "
            f"{len(correlation_rows)}."
        )

        print(
            "No prompt, response text, token text "
            "or judge notes were written."
        )

    except ValueError as exc:
        raise SystemExit(
            f"Error: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
