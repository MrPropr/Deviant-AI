"""Prepare a private JailbreakBench pilot dataset for Deviant-AI."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from .validate_prompt_variants import is_safe_private_output


DATASET_ID = "JailbreakBench/JBB-Behaviors"
DATASET_CONFIG = "behaviors"
DEFAULT_OUTPUT = Path("data/private/pilot_dataset.jsonl")
REQUIRED_JBB_FIELDS = {"Goal", "Target", "Behavior", "Category", "Source"}


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected an integer, got: {value}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be non-negative")
    return parsed


def output_file(value: str) -> Path:
    path = Path(value)
    if path.exists() and path.is_dir():
        raise argparse.ArgumentTypeError(f"Output path is a directory: {path}")
    return path


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def load_jbb_splits() -> tuple[list[dict], list[dict]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the datasets package before loading JBB-Behaviors.") from exc

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG)
    missing_splits = {"harmful", "benign"} - set(dataset.keys())
    if missing_splits:
        available = ", ".join(dataset.keys())
        missing = ", ".join(sorted(missing_splits))
        raise ValueError(f"Dataset splits missing ({missing}). Available splits: {available}")
    harmful_rows = [dict(row) for row in dataset["harmful"]]
    benign_rows = [dict(row) for row in dataset["benign"]]
    validate_jbb_rows(harmful_rows, "harmful")
    validate_jbb_rows(benign_rows, "benign")
    return harmful_rows, benign_rows


def validate_jbb_rows(rows: list[dict], split: str) -> None:
    if not rows:
        raise ValueError(f"JBB split '{split}' is empty")
    missing = REQUIRED_JBB_FIELDS - rows[0].keys()
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"JBB split '{split}' is missing expected fields: {missing_text}")


def sample_rows(rows: list[dict], size: int, rng: random.Random, split: str) -> list[dict]:
    if size > len(rows):
        raise ValueError(f"Requested {size} {split} rows, but only {len(rows)} are available")
    if size == 0:
        return []
    indices = rng.sample(range(len(rows)), size)
    return [rows[index] for index in indices]


def normalize_jbb_row(row: dict, split: str, ordinal: int) -> dict:
    original_index = row.get("Index", ordinal)
    scenario_id = f"jbb_{split}_{original_index}"
    return {
        "scenario_id": scenario_id,
        "behavior_id": scenario_id,
        "label": split,
        "dataset": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "split": split,
        "jbb_index": original_index,
        "source": row.get("Source", "unknown"),
        "risk_area": row.get("Category", "unknown"),
        "behavior_label": row.get("Behavior", "unknown"),
        "behavior": row.get("Goal", ""),
        "goal": row.get("Goal", ""),
        "target": row.get("Target", ""),
        "benign": split == "benign",
        "sanitized": False,
        "private_raw_text": True,
    }


def build_pilot_records(
    harmful_rows: list[dict],
    benign_rows: list[dict],
    harmful_size: int,
    benign_size: int,
    seed: int,
) -> list[dict]:
    if harmful_size + benign_size == 0:
        raise ValueError("At least one of --harmful-size or --benign-size must be greater than zero")

    rng = random.Random(seed)
    sampled_harmful = sample_rows(harmful_rows, harmful_size, rng, "harmful")
    sampled_benign = sample_rows(benign_rows, benign_size, rng, "benign")

    records: list[dict] = []
    records.extend(normalize_jbb_row(row, "harmful", index) for index, row in enumerate(sampled_harmful))
    records.extend(normalize_jbb_row(row, "benign", index) for index, row in enumerate(sampled_benign))
    return records


def make_dry_run_rows(split: str, size: int) -> list[dict]:
    placeholder = "[BENIGN_BEHAVIOR_PLACEHOLDER]" if split == "benign" else "[HARMFUL_BEHAVIOR_PLACEHOLDER]"
    return [
        {
            "Index": index,
            "Goal": placeholder,
            "Target": "[MODEL_RESPONSE_PLACEHOLDER]",
            "Behavior": f"{split}_placeholder",
            "Category": "placeholder",
            "Source": "dry_run",
        }
        for index in range(size)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--harmful-size",
        type=non_negative_int,
        default=10,
        help="Number of harmful JBB examples to include in the private pilot split.",
    )
    parser.add_argument(
        "--benign-size",
        type=non_negative_int,
        default=10,
        help="Number of benign JBB examples to include in the private pilot split.",
    )
    parser.add_argument("--seed", type=int, default=20260709, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--output",
        type=output_file,
        default=DEFAULT_OUTPUT,
        help="Private JSONL output path for the pilot split.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use placeholder rows instead of downloading JBB; intended for tests and CLI checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if not is_safe_private_output(args.output, repo_root):
        raise SystemExit("Private dataset output inside the repository must be ignored by git")
    try:
        if args.dry_run:
            harmful_rows = make_dry_run_rows("harmful", args.harmful_size)
            benign_rows = make_dry_run_rows("benign", args.benign_size)
            source_label = "placeholder dry-run rows"
        else:
            harmful_rows, benign_rows = load_jbb_splits()
            source_label = f"{DATASET_ID}/{DATASET_CONFIG}"

        records = build_pilot_records(
            harmful_rows=harmful_rows,
            benign_rows=benign_rows,
            harmful_size=args.harmful_size,
            benign_size=args.benign_size,
            seed=args.seed,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    write_jsonl(args.output, records)
    harmful_written = sum(1 for record in records if record["split"] == "harmful")
    benign_written = sum(1 for record in records if record["split"] == "benign")
    print(
        "Prepared private pilot dataset: "
        f"{len(records)} records ({harmful_written} harmful, {benign_written} benign), "
        f"seed={args.seed}, source={source_label}, output={args.output}"
    )


if __name__ == "__main__":
    main()
