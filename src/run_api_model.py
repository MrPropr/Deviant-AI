"""Create placeholder API-model outputs without making network requests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PLACEHOLDER_RESPONSE = "[MODEL_RESPONSE_PLACEHOLDER]"


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


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def build_outputs(records: list[dict], provider: str, model_id: str, run_id: str) -> list[dict]:
    outputs: list[dict] = []
    for record in records:
        outputs.append(
            {
                "behavior_id": record["behavior_id"],
                "source": record.get("source", "unknown"),
                "condition": record["condition"],
                "provider": provider,
                "model_id": model_id,
                "run_id": run_id,
                "output_text": PLACEHOLDER_RESPONSE,
                "output_is_placeholder": True,
                "benign": bool(record.get("metadata", {}).get("benign", False)),
            }
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=existing_file, required=True, help="Prompt variant JSONL file.")
    parser.add_argument(
        "--output",
        type=output_file,
        default=Path("results/raw/api_model_outputs.jsonl"),
        help="Raw output JSONL path. Use an ignored path for real outputs.",
    )
    parser.add_argument("--provider", default="placeholder-provider", help="API provider label.")
    parser.add_argument("--model-id", default="placeholder-api-model", help="Model identifier for the run.")
    parser.add_argument("--run-id", default="api_placeholder_run", help="Run identifier.")
    parser.add_argument("--api-key-env", default="DEVIANT_AI_API_KEY", help="Environment variable for future API use.")
    parser.add_argument("--execute", action="store_true", help="Reserved for future real API calls.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of records.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive when provided")
    if args.execute:
        if not os.environ.get(args.api_key_env):
            raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")
        raise SystemExit("Real API execution is not implemented in this lightweight placeholder.")
    records = read_jsonl(args.input, args.limit)
    outputs = build_outputs(records, args.provider, args.model_id, args.run_id)
    write_jsonl(args.output, outputs)
    print(
        f"Wrote {len(outputs)} placeholder API-model outputs at {args.output}. "
        "No network requests were made."
    )


if __name__ == "__main__":
    main()
