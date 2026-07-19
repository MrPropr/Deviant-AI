"""Build public sensitivity summaries from private token metrics."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from src.analyze_token_probabilities import (
    CONDITIONS,
    FORBIDDEN_FIELDS,
    METRICS,
    bootstrap_mean_ci,
    derived_seed,
    existing_file,
    output_file,
    read_jsonl,
    write_csv,
)


SUBSETS = (
    "all",
    "finish_length",
    "finish_non_length",
    "harmful",
    "benign",
    "refusal",
    "unsafe_compliant",
)

OUTPUT_FIELDS = (
    "subset",
    "condition",
    "metric",
    "n",
    "mean",
    "standard_deviation",
    "ci_low",
    "ci_high",
)

REQUIRED_FIELDS = {
    "condition",
    "is_final_turn",
    "label",
    "response_class",
    "finish_reason",
    *METRICS,
}

ALLOWED_LABELS = {
    "harmful",
    "benign",
}

ALLOWED_FINISH_REASONS = {
    "eos",
    "length",
    "other",
    "dry_run",
}


def finite_metric(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Metric must be numeric.") from exc

    if not math.isfinite(parsed):
        raise ValueError("Metric must be finite.")

    return parsed


def validate_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        raise ValueError("Input contains no records.")

    if any(
        set(row) & FORBIDDEN_FIELDS
        for row in rows
    ):
        raise ValueError(
            "Input contains fields that are not allowed for analysis."
        )

    validated: list[dict] = []
    errors: list[str] = []

    for row_number, row in enumerate(rows, start=1):
        missing = REQUIRED_FIELDS - set(row)

        if missing:
            errors.append(
                f"record {row_number}: missing required fields"
            )
            continue

        if row["condition"] not in CONDITIONS:
            errors.append(
                f"record {row_number}: invalid condition"
            )

        if row["label"] not in ALLOWED_LABELS:
            errors.append(
                f"record {row_number}: invalid label"
            )

        if row["finish_reason"] not in ALLOWED_FINISH_REASONS:
            errors.append(
                f"record {row_number}: invalid finish reason"
            )

        if type(row["is_final_turn"]) is not bool:
            errors.append(
                f"record {row_number}: invalid final-turn flag"
            )

        if not isinstance(row["response_class"], str) or not row[
            "response_class"
        ].strip():
            errors.append(
                f"record {row_number}: invalid response class"
            )

        parsed_row = dict(row)

        for metric in METRICS:
            try:
                parsed_row[metric] = finite_metric(row[metric])
            except ValueError:
                errors.append(
                    f"record {row_number}: invalid metric"
                )

        validated.append(parsed_row)

    if errors:
        preview = "\n".join(
            f"- {error}"
            for error in errors[:20]
        )
        raise ValueError(
            f"Validation found {len(errors)} errors:\n{preview}"
        )

    return validated


def select_analysis_rows(
    rows: list[dict],
    include_intermediate_turns: bool = False,
) -> list[dict]:
    if include_intermediate_turns:
        return list(rows)

    return [
        row
        for row in rows
        if row["is_final_turn"] is True
    ]


def select_subset(rows: list[dict], subset: str) -> list[dict]:
    if subset == "all":
        return list(rows)

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

    if subset in ALLOWED_LABELS:
        return [
            row
            for row in rows
            if row["label"] == subset
        ]

    if subset == "refusal":
        return [
            row
            for row in rows
            if row["response_class"] == "refusal"
        ]

    if subset == "unsafe_compliant":
        return [
            row
            for row in rows
            if row["response_class"] in {"unsafe", "compliant"}
        ]

    raise ValueError(f"Unknown subset: {subset}")


def build_sensitivity_rows(
    rows: list[dict],
    bootstrap_iterations: int,
    base_seed: int,
    include_intermediate_turns: bool = False,
) -> list[dict]:
    output: list[dict] = []
    analysis_rows = select_analysis_rows(
        rows,
        include_intermediate_turns=include_intermediate_turns,
    )

    for subset in SUBSETS:
        selected = select_subset(analysis_rows, subset)

        for condition in CONDITIONS:
            condition_rows = [
                row
                for row in selected
                if row["condition"] == condition
            ]

            for metric in METRICS:
                values = [
                    finite_metric(row[metric])
                    for row in condition_rows
                ]

                if values:
                    mean = float(np.mean(values))
                    standard_deviation = (
                        float(np.std(values, ddof=1))
                        if len(values) > 1
                        else 0.0
                    )
                    ci_low, ci_high = bootstrap_mean_ci(
                        values=values,
                        iterations=bootstrap_iterations,
                        seed=derived_seed(
                            base_seed,
                            "sensitivity",
                            subset,
                            condition,
                            metric,
                        ),
                    )
                else:
                    mean = math.nan
                    standard_deviation = math.nan
                    ci_low = math.nan
                    ci_high = math.nan

                output.append(
                    {
                        "subset": subset,
                        "condition": condition,
                        "metric": metric,
                        "n": len(values),
                        "mean": mean,
                        "standard_deviation": standard_deviation,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )

    return output


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=existing_file,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=output_file,
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
    )
    parser.add_argument(
        "--include-intermediate-turns",
        action="store_true",
        help=(
            "Include validated intermediate turns for diagnostic "
            "analysis. The default is final turns only."
        ),
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if args.bootstrap_iterations < 100:
        raise ValueError(
            "--bootstrap-iterations must be at least 100"
        )

    if not args.validate_only and args.output is None:
        raise ValueError(
            "--output is required unless --validate-only is used"
        )

    rows = validate_rows(read_jsonl(args.input))
    final_rows = select_analysis_rows(rows)

    if args.validate_only:
        print(f"Validation passed for {len(rows)} records.")
        print(f"Final records: {len(final_rows)}.")
        return len(rows)

    analysis_rows = select_analysis_rows(
        rows,
        include_intermediate_turns=(
            args.include_intermediate_turns
        ),
    )

    if not analysis_rows:
        raise ValueError(
            "Input contains no records selected for analysis."
        )

    summary_rows = build_sensitivity_rows(
        rows=rows,
        bootstrap_iterations=args.bootstrap_iterations,
        base_seed=args.seed,
        include_intermediate_turns=(
            args.include_intermediate_turns
        ),
    )
    write_csv(
        Path(args.output),
        summary_rows,
        list(OUTPUT_FIELDS),
    )
    print("Sensitivity analysis complete.")
    print(f"Input records: {len(rows)}.")
    print(f"Final records used: {len(final_rows)}.")
    print(f"Records used: {len(analysis_rows)}.")
    print(f"Aggregate rows written: {len(summary_rows)}.")
    return len(summary_rows)


def main() -> None:
    try:
        run(parse_args())
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
