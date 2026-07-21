"""Validate private prompt templates without printing prompt content."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


CONDITION_TURN_COUNTS = {
    "direct": 1,
    "polite": 1,
    "multi_turn": 3,
    "polite_multi_turn": 3,
}
TEMPLATE_MARKER = "_PROMPT_TEMPLATE"


def existing_file(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Input path is not a file: {path}")
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
                raise ValueError(f"Invalid JSON at line {line_number}") from exc
    return records


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def count_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def refresh_counts(records: list[dict]) -> None:
    for record in records:
        conditions = record.get("conditions", {})
        if not isinstance(conditions, dict):
            continue
        for specification in conditions.values():
            if not isinstance(specification, dict):
                continue
            turns = specification.get("turns", [])
            if not isinstance(turns, list):
                continue
            for position, turn in enumerate(turns, start=1):
                if not isinstance(turn, dict) or not isinstance(turn.get("content"), str):
                    continue
                turn["turn_index"] = position
                turn["word_count"] = count_words(turn["content"])
                turn["token_count"] = count_tokens(turn["content"])
            specification["word_count"] = sum(
                int(turn.get("word_count", 0)) for turn in turns if isinstance(turn, dict)
            )
            specification["token_count"] = sum(
                int(turn.get("token_count", 0)) for turn in turns if isinstance(turn, dict)
            )
            specification["count_method"] = "regex_v1"


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def validate_records(records: list[dict], require_ready: bool = False) -> list[str]:
    errors: list[str] = []
    seen_scenario_ids: set[str] = set()

    for record_number, record in enumerate(records, start=1):
        scenario_id = str(record.get("scenario_id", "")).strip()
        reference = f"record_{record_number}"
        if not scenario_id:
            errors.append(f"{reference}: missing scenario_id")
        elif scenario_id in seen_scenario_ids:
            errors.append(f"{reference}: duplicate scenario_id")
        else:
            seen_scenario_ids.add(scenario_id)

        if record.get("label") not in {"harmful", "benign"}:
            errors.append(f"{reference}: label must be harmful or benign")

        dataset = record.get("dataset")
        if not isinstance(dataset, dict):
            errors.append(f"{reference}: dataset metadata is missing")
        else:
            for field in ("id", "config", "split", "index", "source", "category", "behavior_label"):
                if field not in dataset:
                    errors.append(f"{reference}: dataset metadata missing {field}")

        conditions = record.get("conditions")
        if not isinstance(conditions, dict):
            errors.append(f"{reference}: conditions object is missing")
            continue

        actual_conditions = set(conditions)
        expected_conditions = set(CONDITION_TURN_COUNTS)
        for missing in sorted(expected_conditions - actual_conditions):
            errors.append(f"{reference}: missing condition {missing}")
        for extra in sorted(actual_conditions - expected_conditions):
            errors.append(f"{reference}: unexpected condition {extra}")

        for condition, expected_turns in CONDITION_TURN_COUNTS.items():
            specification = conditions.get(condition)
            if not isinstance(specification, dict):
                continue
            if require_ready and specification.get("prompt_status") != "ready":
                errors.append(f"{reference}/{condition}: prompt_status must be ready")
            turns = specification.get("turns")
            if not isinstance(turns, list):
                errors.append(f"{reference}/{condition}: turns must be a list")
                continue
            if len(turns) != expected_turns:
                errors.append(f"{reference}/{condition}: expected {expected_turns} turns, found {len(turns)}")

            total_words = 0
            total_tokens = 0
            for position, turn in enumerate(turns, start=1):
                if not isinstance(turn, dict):
                    errors.append(f"{reference}/{condition}/turn_{position}: turn must be an object")
                    continue
                content = turn.get("content")
                if not isinstance(content, str) or not content.strip():
                    errors.append(f"{reference}/{condition}/turn_{position}: empty turn")
                    continue
                if require_ready and TEMPLATE_MARKER in content:
                    errors.append(f"{reference}/{condition}/turn_{position}: template placeholder remains")
                word_count = count_words(content)
                token_count = count_tokens(content)
                total_words += word_count
                total_tokens += token_count
                if turn.get("turn_index") != position:
                    errors.append(f"{reference}/{condition}/turn_{position}: turn_index mismatch")
                if turn.get("word_count") != word_count:
                    errors.append(f"{reference}/{condition}/turn_{position}: word_count mismatch")
                if turn.get("token_count") != token_count:
                    errors.append(f"{reference}/{condition}/turn_{position}: token_count mismatch")

            if specification.get("word_count") != total_words:
                errors.append(f"{reference}/{condition}: total word_count mismatch")
            if specification.get("token_count") != total_tokens:
                errors.append(f"{reference}/{condition}: total token_count mismatch")
            if specification.get("count_method") != "regex_v1":
                errors.append(f"{reference}/{condition}: count_method must be regex_v1")

    return errors


def is_git_ignored(path: Path, repo_root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root.resolve().as_posix()}",
            "check-ignore",
            "-q",
            "--",
            relative.as_posix(),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def is_safe_private_output(path: Path, repo_root: Path) -> bool:
    """Allow private outputs outside the repository or under an ignored path."""

    try:
        path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return True
    return is_git_ignored(path, repo_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=existing_file, required=True, help="Private prompt-template JSONL file.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root used for the git-ignore safety check.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Reject unfilled template placeholders and non-ready conditions.",
    )
    parser.add_argument(
        "--refresh-counts",
        action="store_true",
        help="Update private word/token count fields in place before validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        records = read_jsonl(args.input)
    except ValueError as exc:
        raise SystemExit(f"Validation failed: {exc}") from exc

    ignored = is_git_ignored(args.input, args.repo_root)
    if args.refresh_counts and not ignored:
        raise SystemExit("Validation failed: refusing to rewrite a prompt file that is not ignored by git")
    if args.refresh_counts:
        refresh_counts(records)
        write_jsonl(args.input, records)
    errors = validate_records(records, require_ready=args.require_ready)
    if not ignored:
        errors.append("private prompt file is not ignored by git")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(f"Validation failed with {len(errors)} structural errors; prompt content was not printed.")
    print(f"Validated {len(records)} private scenarios; all four conditions and count fields are structurally valid.")


if __name__ == "__main__":
    main()
