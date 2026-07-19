"""Create public aggregate analyses from private fixed-continuation scores."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.analyze_token_probabilities import (
    COMPARISONS,
    CONDITIONS,
    METRICS,
    bootstrap_mean_ci,
    derived_seed,
    output_file,
    sign_flip_permutation_test,
    wilcoxon_test,
    write_csv,
)
from src.score_fixed_continuations import (
    ensure_private_repository_path,
    read_jsonl,
    validate_detail_record,
)


ANALYSIS_METRICS = (*METRICS, "continuation_token_count")
SUBSETS = ("all", "harmful", "benign")

SUMMARY_FIELDS = (
    "subset",
    "condition",
    "metric",
    "n",
    "mean",
    "standard_deviation",
    "ci_low",
    "ci_high",
)

PAIRED_FIELDS = (
    "subset",
    "comparison",
    "left_condition",
    "right_condition",
    "metric",
    "n_pairs",
    "mean_difference",
    "ci_low",
    "ci_high",
    "permutation_p_value",
    "wilcoxon_p_value",
)


def finite_value(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError("Private detail input contains an invalid metric.") from None
    if not math.isfinite(parsed):
        raise ValueError("Private detail input contains an invalid metric.")
    return parsed


def validate_detail_rows(rows: list[dict]) -> dict[str, dict[str, dict]]:
    if not rows:
        raise ValueError("Private detail input contains no records.")

    scenario_map: dict[str, dict[str, dict]] = defaultdict(dict)
    scenario_metadata: dict[str, tuple[str, str]] = {}
    run_metadata: set[tuple[str, str, str]] = set()

    for row in rows:
        validate_detail_record(row)
        scenario_id = row["scenario_id"]
        condition = row["condition"]
        if condition in scenario_map[scenario_id]:
            raise ValueError("Private detail input contains a duplicate key.")
        scenario_map[scenario_id][condition] = row

        metadata = (row["label"], row["anchor_condition"])
        if scenario_id in scenario_metadata and scenario_metadata[scenario_id] != metadata:
            raise ValueError("Scenario metadata are inconsistent across conditions.")
        scenario_metadata[scenario_id] = metadata
        run_metadata.add((row["model_id"], row["run_id"], row["status"]))

    if len(run_metadata) != 1:
        raise ValueError("Private detail input mixes model runs or statuses.")

    expected = set(CONDITIONS)
    if any(set(condition_rows) != expected for condition_rows in scenario_map.values()):
        raise ValueError("Every scenario must contain all four conditions.")
    if any(
        len(
            {
                row["continuation_token_count"]
                for row in condition_rows.values()
            }
        )
        != 1
        for condition_rows in scenario_map.values()
    ):
        raise ValueError("Continuation token counts differ across conditions.")

    return dict(scenario_map)


def subset_scenarios(
    scenario_map: dict[str, dict[str, dict]],
    subset: str,
) -> list[dict[str, dict]]:
    scenarios = list(scenario_map.values())
    if subset == "all":
        return scenarios
    if subset in {"harmful", "benign"}:
        return [
            conditions
            for conditions in scenarios
            if conditions["direct"]["label"] == subset
        ]
    raise ValueError(f"Unknown subset: {subset}")


def build_summary_rows(
    scenario_map: dict[str, dict[str, dict]],
    bootstrap_iterations: int,
    base_seed: int,
) -> list[dict]:
    output: list[dict] = []
    for subset in SUBSETS:
        scenarios = subset_scenarios(scenario_map, subset)
        for condition in CONDITIONS:
            for metric in ANALYSIS_METRICS:
                values = [
                    finite_value(scenario[condition][metric])
                    for scenario in scenarios
                ]
                if values:
                    mean = float(np.mean(values))
                    standard_deviation = (
                        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    )
                    ci_low, ci_high = bootstrap_mean_ci(
                        values,
                        iterations=bootstrap_iterations,
                        seed=derived_seed(
                            base_seed,
                            "fixed_summary",
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


def build_paired_rows(
    scenario_map: dict[str, dict[str, dict]],
    bootstrap_iterations: int,
    permutation_iterations: int,
    base_seed: int,
) -> list[dict]:
    output: list[dict] = []
    for subset in SUBSETS:
        scenarios = subset_scenarios(scenario_map, subset)
        for comparison, left_condition, right_condition in COMPARISONS:
            for metric in ANALYSIS_METRICS:
                differences = [
                    finite_value(scenario[right_condition][metric])
                    - finite_value(scenario[left_condition][metric])
                    for scenario in scenarios
                ]
                if differences:
                    mean_difference = float(np.mean(differences))
                    ci_low, ci_high = bootstrap_mean_ci(
                        differences,
                        iterations=bootstrap_iterations,
                        seed=derived_seed(
                            base_seed,
                            "fixed_paired_bootstrap",
                            subset,
                            comparison,
                            metric,
                        ),
                    )
                    permutation_p, _method = sign_flip_permutation_test(
                        differences,
                        iterations=permutation_iterations,
                        seed=derived_seed(
                            base_seed,
                            "fixed_permutation",
                            subset,
                            comparison,
                            metric,
                        ),
                    )
                    _statistic, wilcoxon_p = wilcoxon_test(differences)
                else:
                    mean_difference = math.nan
                    ci_low = math.nan
                    ci_high = math.nan
                    permutation_p = math.nan
                    wilcoxon_p = math.nan
                output.append(
                    {
                        "subset": subset,
                        "comparison": comparison,
                        "left_condition": left_condition,
                        "right_condition": right_condition,
                        "metric": metric,
                        "n_pairs": len(differences),
                        "mean_difference": mean_difference,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "permutation_p_value": permutation_p,
                        "wilcoxon_p_value": wilcoxon_p,
                    }
                )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary-output", type=output_file, required=True)
    parser.add_argument("--paired-output", type=output_file, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--permutation-iterations", type=int, default=50000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, int]:
    if not args.input.exists() or not args.input.is_file():
        raise ValueError("Private detail input does not exist or is not a file.")
    if args.bootstrap_iterations < 100:
        raise ValueError("--bootstrap-iterations must be at least 100")
    if args.permutation_iterations < 100:
        raise ValueError("--permutation-iterations must be at least 100")
    if args.summary_output == args.paired_output:
        raise ValueError("Summary and paired outputs must be different files.")
    if not args.overwrite and (
        args.summary_output.exists() or args.paired_output.exists()
    ):
        raise ValueError("A public output already exists; use --overwrite to replace it.")

    repo_root = Path(__file__).resolve().parents[1]
    ensure_private_repository_path(args.input, repo_root)
    rows = read_jsonl(args.input)
    scenario_map = validate_detail_rows(rows)
    summary_rows = build_summary_rows(
        scenario_map,
        bootstrap_iterations=args.bootstrap_iterations,
        base_seed=args.seed,
    )
    paired_rows = build_paired_rows(
        scenario_map,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
        base_seed=args.seed,
    )
    write_csv(args.summary_output, summary_rows, list(SUMMARY_FIELDS))
    write_csv(args.paired_output, paired_rows, list(PAIRED_FIELDS))
    print(
        "Fixed-continuation analysis complete; "
        f"scenarios={len(scenario_map)}, summary_rows={len(summary_rows)}, "
        f"paired_rows={len(paired_rows)}."
    )
    print("Only aggregate public CSV rows were written.")
    return {
        "scenarios": len(scenario_map),
        "summary_rows": len(summary_rows),
        "paired_rows": len(paired_rows),
    }


def main() -> None:
    try:
        run(parse_args())
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
