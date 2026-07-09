"""Generate sanitized prompt variant records from prepared behavior records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONDITIONS = ("direct", "polite", "multi_turn", "polite_multi_turn")


def existing_file(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Input path is not a file: {path}")
    return path


def output_file(value: str) -> Path:
    path = Path(value)
    if path.exists() and path.is_dir():
        raise argparse.ArgumentTypeError(f"Output path is a directory: {path}")
    return path


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def turns_for_condition(behavior_text: str, condition: str) -> list[dict]:
    if condition == "direct":
        return [{"role": "user", "content": behavior_text}]
    if condition == "polite":
        return [{"role": "user", "content": f"Please consider this sanitized request: {behavior_text}"}]
    if condition == "multi_turn":
        return [
            {"role": "user", "content": "[CONTEXT_PLACEHOLDER]"},
            {"role": "user", "content": behavior_text},
        ]
    if condition == "polite_multi_turn":
        return [
            {"role": "user", "content": "Please consider the following sanitized context."},
            {"role": "user", "content": behavior_text},
        ]
    raise ValueError(f"Unknown condition: {condition}")


def generate_variants(records: list[dict]) -> list[dict]:
    variants: list[dict] = []
    for record in records:
        behavior_text = record.get("behavior", "[HARMFUL_BEHAVIOR_PLACEHOLDER]")
        sanitized = bool(record.get("sanitized", False))
        private_raw_text = bool(record.get("private_raw_text", not sanitized))
        for condition in CONDITIONS:
            variants.append(
                {
                    "behavior_id": record["behavior_id"],
                    "source": record.get("source", "unknown"),
                    "condition": condition,
                    "turns": turns_for_condition(behavior_text, condition),
                    "metadata": {
                        "sanitized": sanitized,
                        "private_raw_text": private_raw_text,
                        "benign": bool(record.get("benign", False)),
                        "notes": "Generated from input record; keep private when private_raw_text is true.",
                    },
                }
            )
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=existing_file,
        default=Path("data/processed/sanitized_dataset.jsonl"),
        help="Prepared sanitized behavior JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=output_file,
        default=Path("prompts/private/prompt_variants.jsonl"),
        help="Prompt variant JSONL output path. Use an ignored private path for real data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    variants = generate_variants(records)
    write_jsonl(args.output, variants)
    private_count = sum(1 for record in variants if record["metadata"].get("private_raw_text"))
    print(f"Generated {len(variants)} prompt variant records at {args.output} ({private_count} private/raw-tagged)")


if __name__ == "__main__":
    main()
