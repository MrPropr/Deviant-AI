"""Prepare sanitized JSONL records for the Deviant-AI pilot pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FIELDS = {"behavior_id", "source", "behavior"}


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
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            missing = REQUIRED_FIELDS - record.keys()
            if missing:
                missing_text = ", ".join(sorted(missing))
                raise ValueError(f"{path}:{line_number} missing fields: {missing_text}")
            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def prepare_records(records: list[dict]) -> list[dict]:
    prepared: list[dict] = []
    for record in records:
        prepared.append(
            {
                "behavior_id": record["behavior_id"],
                "source": record["source"],
                "behavior": record["behavior"],
                "risk_area": record.get("risk_area", "unspecified"),
                "benign": bool(record.get("benign", False)),
                "sanitized": True,
                "notes": record.get("notes", "Sanitized public record."),
            }
        )
    return prepared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=existing_file,
        default=Path("data/sanitized_examples.jsonl"),
        help="Sanitized input JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=output_file,
        default=Path("data/processed/sanitized_dataset.jsonl"),
        help="Prepared sanitized output JSONL file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    prepared = prepare_records(records)
    write_jsonl(args.output, prepared)
    print(f"Prepared {len(prepared)} sanitized records at {args.output}")


if __name__ == "__main__":
    main()
