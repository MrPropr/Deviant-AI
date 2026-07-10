"""Convert private raw model outputs to a private CSV annotation sheet."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .validate_prompt_variants import is_git_ignored


FIELDS = [
    "response_id",
    "scenario_id",
    "condition",
    "turn_index",
    "label",
    "model_id",
    "run_id",
    "generation_status",
    "dataset_metadata",
    "response_text",
    "harmfulness_score",
    "refusal",
    "benign_success",
    "judge_notes",
]


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
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def annotation_row(record: dict) -> dict:
    return {
        "response_id": record.get("response_id", ""),
        "scenario_id": record.get("scenario_id", ""),
        "condition": record.get("condition", ""),
        "turn_index": record.get("turn_index", ""),
        "label": record.get("label", ""),
        "model_id": record.get("model_id", ""),
        "run_id": record.get("run_id", ""),
        "generation_status": record.get("generation_status", ""),
        "dataset_metadata": json.dumps(record.get("dataset", {}), ensure_ascii=True, sort_keys=True),
        "response_text": record.get("response_text", ""),
        "harmfulness_score": "",
        "refusal": "",
        "benign_success": "",
        "judge_notes": "",
    }


def write_annotation_sheet(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(annotation_row(record))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=existing_file, required=True, help="Private raw output JSONL file.")
    parser.add_argument(
        "--output",
        type=output_file,
        default=Path("results/judged/pilot_annotation_sheet.csv"),
        help="Ignored private CSV annotation sheet path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if not is_git_ignored(args.output, repo_root):
        raise SystemExit("Private annotation output path must be ignored by git")
    try:
        records = read_jsonl(args.input)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    write_annotation_sheet(args.output, records)
    print(f"Created private annotation sheet with {len(records)} rows at {args.output}; response text was not printed.")


if __name__ == "__main__":
    main()
