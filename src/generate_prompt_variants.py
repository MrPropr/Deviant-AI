"""Create private, non-adaptive prompt templates for the four pilot conditions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .validate_prompt_variants import is_safe_private_output


CONDITIONS = ("direct", "polite", "multi_turn", "polite_multi_turn")
COUNT_METHOD = "regex_v1"
TEMPLATE_CONTENTS = {
    "direct": ["[DIRECT_PROMPT_TEMPLATE]"],
    "polite": ["[POLITE_PROMPT_TEMPLATE]"],
    "multi_turn": [
        "[MULTI_TURN_PROMPT_TEMPLATE_1]",
        "[MULTI_TURN_PROMPT_TEMPLATE_2]",
        "[MULTI_TURN_PROMPT_TEMPLATE_3]",
    ],
    "polite_multi_turn": [
        "[POLITE_MULTI_TURN_PROMPT_TEMPLATE_1]",
        "[POLITE_MULTI_TURN_PROMPT_TEMPLATE_2]",
        "[POLITE_MULTI_TURN_PROMPT_TEMPLATE_3]",
    ],
}


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


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def count_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def build_condition_template(condition: str) -> dict:
    turns = []
    for turn_index, content in enumerate(TEMPLATE_CONTENTS[condition], start=1):
        turns.append(
            {
                "turn_index": turn_index,
                "role": "user",
                "content": content,
                "word_count": count_words(content),
                "token_count": count_tokens(content),
            }
        )
    return {
        "prompt_status": "requires_manual_authoring",
        "count_method": COUNT_METHOD,
        "word_count": sum(turn["word_count"] for turn in turns),
        "token_count": sum(turn["token_count"] for turn in turns),
        "turns": turns,
    }


def dataset_metadata(record: dict) -> dict:
    return {
        "id": str(record.get("dataset") or "JailbreakBench/JBB-Behaviors"),
        "config": str(record.get("dataset_config") or "behaviors"),
        "split": str(record.get("split") or record.get("label") or "unknown"),
        "index": record.get("jbb_index"),
        "source": str(record.get("source") or "unknown"),
        "category": str(record.get("risk_area") or "unknown"),
        "behavior_label": str(record.get("behavior_label") or "unknown"),
    }


def generate_templates(records: list[dict]) -> list[dict]:
    templates: list[dict] = []
    seen_scenario_ids: set[str] = set()
    for record_number, record in enumerate(records, start=1):
        scenario_id = str(record.get("scenario_id") or record.get("behavior_id") or "").strip()
        if not scenario_id:
            raise ValueError("Dataset record is missing scenario_id")
        if scenario_id in seen_scenario_ids:
            raise ValueError(f"Dataset row {record_number} has a duplicate scenario_id")
        seen_scenario_ids.add(scenario_id)

        label = str(record.get("label") or record.get("split") or "").strip().lower()
        if label not in {"harmful", "benign"}:
            raise ValueError(f"Dataset row {record_number} has an invalid label")

        templates.append(
            {
                "scenario_id": scenario_id,
                "label": label,
                "dataset": dataset_metadata(record),
                "private_raw_text": bool(record.get("private_raw_text", True)),
                "conditions": {condition: build_condition_template(condition) for condition in CONDITIONS},
            }
        )
    return templates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=existing_file,
        required=True,
        help="Private prepared dataset JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=output_file,
        default=Path("prompts/private/prompt_variants.jsonl"),
        help="Ignored private prompt-template JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if not is_safe_private_output(args.output, repo_root):
        raise SystemExit("Private prompt output path must be ignored by git")
    try:
        records = read_jsonl(args.input)
        templates = generate_templates(records)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    write_jsonl(args.output, templates)
    print(f"Created {len(templates)} private scenario templates with four fixed conditions at {args.output}")


if __name__ == "__main__":
    main()
