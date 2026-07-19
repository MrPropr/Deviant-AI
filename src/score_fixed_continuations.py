"""Score one fixed continuation under four contexts without generation."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from src.analyze_token_probabilities import CONDITIONS, METRICS
from src.run_local_model import DEFAULT_MODEL_ID
from src.score_response_logprobs import (
    RunningTokenMetrics,
    TeacherForcingBackend,
    existing_file,
    output_file,
    read_jsonl,
)
from src.validate_prompt_variants import is_git_ignored


ALLOWED_LABELS = {"harmful", "benign"}
ALLOWED_ROLES = {"system", "user", "assistant"}
ALLOWED_STATUSES = {"ok", "dry_run"}

DETAIL_FIELDS = (
    "scenario_id",
    "label",
    "anchor_condition",
    "condition",
    "model_id",
    "run_id",
    "continuation_token_count",
    *METRICS,
    "status",
)

FORBIDDEN_DETAIL_FIELDS = {
    "fixed_continuation",
    "prompt",
    "prompt_text",
    "messages",
    "contexts",
    "context_text",
    "response_text",
    "judge_notes",
    "token_ids",
    "token_strings",
    "logits",
    "top_tokens",
    "raw_output",
}


def validate_input_records(records: list[dict]) -> list[dict]:
    if not records:
        raise ValueError("Input contains no scenarios.")

    errors: list[str] = []
    seen_ids: set[str] = set()
    required_fields = {
        "scenario_id",
        "label",
        "anchor_condition",
        "fixed_continuation",
        "contexts",
    }
    required_conditions = set(CONDITIONS)

    for row_number, record in enumerate(records, start=1):
        missing = required_fields - set(record)
        if missing:
            errors.append(f"record {row_number}: missing required fields")
            continue

        scenario_id = record["scenario_id"]
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            errors.append(f"record {row_number}: invalid scenario identifier")
        elif scenario_id in seen_ids:
            errors.append(f"record {row_number}: duplicate scenario identifier")
        else:
            seen_ids.add(scenario_id)

        if record["label"] not in ALLOWED_LABELS:
            errors.append(f"record {row_number}: invalid label")

        if record["anchor_condition"] != "direct":
            errors.append(f"record {row_number}: anchor must be direct")

        continuation = record["fixed_continuation"]
        if not isinstance(continuation, str) or not continuation.strip():
            errors.append(f"record {row_number}: empty fixed continuation")

        contexts = record["contexts"]
        if not isinstance(contexts, dict):
            errors.append(f"record {row_number}: contexts must be an object")
            continue

        if set(contexts) != required_conditions:
            errors.append(f"record {row_number}: condition set is incomplete")
            continue

        for condition in CONDITIONS:
            messages = contexts[condition]
            if not isinstance(messages, list) or not messages:
                errors.append(f"record {row_number}: invalid context")
                continue

            user_count = 0
            context_valid = True
            for message in messages:
                if not isinstance(message, dict):
                    context_valid = False
                    break

                role = message.get("role")
                content = message.get("content")
                if role not in ALLOWED_ROLES:
                    context_valid = False
                    break
                if not isinstance(content, str) or not content.strip():
                    context_valid = False
                    break
                if role == "user":
                    user_count += 1

            if not context_valid:
                errors.append(f"record {row_number}: invalid context message")
                continue

            if user_count < 1:
                errors.append(f"record {row_number}: context lacks a user message")

            if messages[-1].get("role") != "user":
                errors.append(
                    f"record {row_number}: context must end with a user message"
                )

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:20])
        raise ValueError(
            f"Input validation found {len(errors)} errors:\n{preview}"
        )

    return records


def _input_ids(encoded: Any) -> list[int]:
    value = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(
        encoded, "input_ids", None
    )
    if value is None:
        raise ValueError("Tokenizer did not return input IDs.")

    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Tokenizer returned an unexpected batch.")
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError("Tokenizer returned invalid input IDs.")
    return value


def tokenize_context(
    tokenizer: Any,
    messages: list[dict],
    fixed_continuation: str,
) -> tuple[list[int], list[int]]:
    try:
        prefix_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(prefix_text, str) or not prefix_text:
            raise ValueError

        prefix_ids = _input_ids(
            tokenizer(prefix_text, add_special_tokens=False)
        )
        full_ids = _input_ids(
            tokenizer(prefix_text + fixed_continuation, add_special_tokens=False)
        )
    except Exception:
        raise ValueError("Tokenization failed for one context.") from None

    if not prefix_ids:
        raise ValueError("A context prefix contains no tokens.")
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "Context prefix token IDs are not an exact prefix of the full input."
        )

    continuation_ids = full_ids[len(prefix_ids) :]
    if not continuation_ids:
        raise ValueError("Fixed continuation contains no tokens.")
    return prefix_ids, continuation_ids


def tokenize_scenario(
    tokenizer: Any,
    scenario: dict,
) -> dict[str, tuple[list[int], list[int]]]:
    tokenized: dict[str, tuple[list[int], list[int]]] = {}
    expected_continuation_ids: list[int] | None = None

    for condition in CONDITIONS:
        prefix_ids, continuation_ids = tokenize_context(
            tokenizer,
            scenario["contexts"][condition],
            scenario["fixed_continuation"],
        )
        if expected_continuation_ids is None:
            expected_continuation_ids = continuation_ids
        elif continuation_ids != expected_continuation_ids:
            raise ValueError(
                "Fixed continuation token IDs differ across conditions."
            )
        tokenized[condition] = (prefix_ids, continuation_ids)

    return tokenized


class SyntheticTokenizer:
    """Deterministic character tokenizer for plumbing checks only."""

    chat_template = "synthetic-fixed-continuation-template-v1"

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        if tokenize or not add_generation_prompt:
            raise ValueError("Unsupported synthetic tokenizer mode.")
        parts = [
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        ]
        return "".join(parts) + "<assistant>"

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        if add_special_tokens:
            raise ValueError("Synthetic tokenizer does not add special tokens.")
        return {"input_ids": [ord(character) + 1 for character in text]}


def _dry_metrics(prefix_ids: list[int], continuation_ids: list[int]) -> dict[str, float]:
    mean_logprob = -0.1 - (
        (sum(continuation_ids) % 97) + (len(prefix_ids) % 29)
    ) / 1000.0
    entropy = 0.4 + (len(prefix_ids) % 17) / 100.0
    margin = 0.2 + (len(prefix_ids) % 11) / 100.0
    return {
        "mean_token_logprob": mean_logprob,
        "geometric_mean_token_probability": math.exp(mean_logprob),
        "perplexity": math.exp(-mean_logprob),
        "mean_first_8_token_logprob": mean_logprob,
        "mean_first_16_token_logprob": mean_logprob,
        "mean_first_32_token_logprob": mean_logprob,
        "mean_token_entropy": entropy,
        "mean_top1_top2_probability_margin": margin,
    }


class SyntheticScoringBackend:
    status = "dry_run"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.tokenizer = SyntheticTokenizer()

    def score(
        self,
        prefix_ids: list[int],
        continuation_ids: list[int],
        chunk_size: int,
    ) -> dict[str, float]:
        del chunk_size
        return _dry_metrics(prefix_ids, continuation_ids)


class FixedContinuationBackend(TeacherForcingBackend):
    status = "ok"

    def __init__(self, model_id: str, seed: int) -> None:
        settings = {
            "model_id": model_id,
            "load_in_4bit": True,
            "quantization_type": "nf4",
            "compute_dtype": "float16",
            "seed": seed,
        }
        super().__init__(settings)
        if hasattr(self.torch, "use_deterministic_algorithms"):
            self.torch.use_deterministic_algorithms(True, warn_only=True)

    def score(
        self,
        prefix_ids: list[int],
        continuation_ids: list[int],
        chunk_size: int,
    ) -> dict[str, float]:
        torch = self.torch
        prefix_tensor = torch.tensor(
            [prefix_ids], dtype=torch.long, device=self.device
        )
        continuation_tensor = torch.tensor(
            [continuation_ids], dtype=torch.long, device=self.device
        )
        past_key_values, next_token_logits = self.process_prefix(
            prompt_ids=prefix_tensor,
            chunk_size=chunk_size,
        )
        metrics = RunningTokenMetrics()

        with torch.inference_mode():
            for start in range(0, len(continuation_ids), chunk_size):
                end = min(start + chunk_size, len(continuation_ids))
                token_chunk = continuation_tensor[:, start:end]
                metrics.update(
                    logits=next_token_logits,
                    target_ids=token_chunk[:, 0],
                    torch_module=torch,
                )
                outputs = self.model(
                    input_ids=token_chunk,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                if token_chunk.shape[-1] > 1:
                    metrics.update(
                        logits=outputs.logits[:, :-1, :],
                        target_ids=token_chunk[:, 1:],
                        torch_module=torch,
                    )
                next_token_logits = outputs.logits[:, -1, :]

        finalized = metrics.finalize()
        if int(finalized["scored_token_count"]) != len(continuation_ids):
            raise ValueError("Continuation token scoring count mismatch.")
        return {metric: float(finalized[metric]) for metric in METRICS}


def validate_detail_record(record: dict) -> None:
    if set(record) != set(DETAIL_FIELDS):
        raise ValueError("Detail output record has an invalid schema.")
    if set(record) & FORBIDDEN_DETAIL_FIELDS:
        raise ValueError("Detail output contains a forbidden field.")
    if not isinstance(record["scenario_id"], str) or not record["scenario_id"].strip():
        raise ValueError("Detail output has an invalid scenario identifier.")
    if record["label"] not in ALLOWED_LABELS:
        raise ValueError("Detail output has an invalid label.")
    if record["anchor_condition"] != "direct":
        raise ValueError("Detail output has an invalid anchor.")
    if record["condition"] not in CONDITIONS:
        raise ValueError("Detail output has an invalid condition.")
    if not isinstance(record["model_id"], str) or not record["model_id"]:
        raise ValueError("Detail output has an invalid model identifier.")
    if not isinstance(record["run_id"], str) or not record["run_id"]:
        raise ValueError("Detail output has an invalid run identifier.")
    count = record["continuation_token_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Detail output has an invalid token count.")
    for metric in METRICS:
        try:
            value = float(record[metric])
        except (TypeError, ValueError):
            raise ValueError("Detail output contains an invalid metric.") from None
        if not math.isfinite(value):
            raise ValueError("Detail output contains an invalid metric.")
    if record["status"] not in ALLOWED_STATUSES:
        raise ValueError("Detail output has an invalid status.")


def atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                validate_detail_record(record)
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=True,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def ensure_private_repository_path(path: Path, repo_root: Path) -> None:
    try:
        path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return
    if not is_git_ignored(path, repo_root):
        raise ValueError("Private JSONL paths inside the repository must be ignored.")


def load_completed_records(
    path: Path,
    model_id: str,
    run_id: str,
    status: str,
) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    if not path.exists():
        return [], {}
    records = read_jsonl(path)
    completed: dict[tuple[str, str], dict] = {}
    for record in records:
        validate_detail_record(record)
        key = (record["scenario_id"], record["condition"])
        if key in completed:
            raise ValueError("Resume output contains a duplicate key.")
        if (
            record["model_id"] != model_id
            or record["run_id"] != run_id
            or record["status"] != status
        ):
            raise ValueError("Resume output metadata does not match this run.")
        completed[key] = record
    return records, completed


def _validated_metrics(metrics: dict) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in METRICS:
        try:
            value = float(metrics[metric])
        except (KeyError, TypeError, ValueError):
            raise ValueError("Scoring backend returned an invalid metric set.") from None
        if not math.isfinite(value):
            raise ValueError("Scoring backend returned a non-finite metric.")
        result[metric] = value
    return result


def score_records(
    records: list[dict],
    backend: Any,
    output_path: Path,
    run_id: str,
    chunk_size: int,
    resume: bool = False,
) -> dict[str, int]:
    if output_path.exists() and not resume:
        raise ValueError("Output already exists; use --resume to continue.")

    status = getattr(backend, "status", "ok")
    model_id = str(getattr(backend, "model_id", ""))
    if status not in ALLOWED_STATUSES or not model_id:
        raise ValueError("Scoring backend metadata is invalid.")

    saved, completed = load_completed_records(
        output_path,
        model_id=model_id,
        run_id=run_id,
        status=status,
    ) if resume else ([], {})
    expected_keys = {
        (scenario["scenario_id"], condition)
        for scenario in records
        for condition in CONDITIONS
    }
    if set(completed) - expected_keys:
        raise ValueError("Resume output contains keys absent from the current input.")
    generated = 0
    resumed = 0

    for scenario in records:
        tokenized = tokenize_scenario(backend.tokenizer, scenario)
        for condition in CONDITIONS:
            prefix_ids, continuation_ids = tokenized[condition]
            key = (scenario["scenario_id"], condition)
            if key in completed:
                prior = completed[key]
                if (
                    prior["label"] != scenario["label"]
                    or prior["anchor_condition"] != scenario["anchor_condition"]
                    or prior["continuation_token_count"] != len(continuation_ids)
                ):
                    raise ValueError("Resume output does not match the current input.")
                resumed += 1
                continue

            metrics = _validated_metrics(
                backend.score(prefix_ids, continuation_ids, chunk_size)
            )
            detail = {
                "scenario_id": scenario["scenario_id"],
                "label": scenario["label"],
                "anchor_condition": scenario["anchor_condition"],
                "condition": condition,
                "model_id": model_id,
                "run_id": run_id,
                "continuation_token_count": len(continuation_ids),
                **metrics,
                "status": status,
            }
            validate_detail_record(detail)
            saved.append(detail)
            completed[key] = detail
            atomic_write_jsonl(output_path, saved)
            generated += 1

    return {
        "input_scenarios": len(records),
        "generated": generated,
        "resumed": resumed,
        "completed": len(completed),
    }


def load_tokenizer(model_id: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise RuntimeError("Install transformers before tokenizer validation.") from None
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("Tokenizer does not provide a chat template.")
    return tokenizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=existing_file, required=True)
    parser.add_argument("--output", type=output_file)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace, backend: Any | None = None) -> dict[str, int]:
    if args.model_id != DEFAULT_MODEL_ID:
        raise ValueError(f"--model-id must be {DEFAULT_MODEL_ID}")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    if not args.validate_only and args.output is None:
        raise ValueError("--output is required unless --validate-only is used")

    repo_root = Path(__file__).resolve().parents[1]
    ensure_private_repository_path(args.input, repo_root)
    if args.output is not None:
        ensure_private_repository_path(args.output, repo_root)

    records = validate_input_records(read_jsonl(args.input))
    print(f"Validated scenarios: {len(records)}.")

    if args.validate_only:
        tokenizer = (
            backend.tokenizer
            if backend is not None
            else SyntheticTokenizer() if args.dry_run else load_tokenizer(args.model_id)
        )
        for scenario in records:
            tokenize_scenario(tokenizer, scenario)
        print(f"Validated context-continuation pairs: {len(records) * len(CONDITIONS)}.")
        print("Validation-only run complete; no model was loaded and no output was written.")
        return {
            "input_scenarios": len(records),
            "generated": 0,
            "resumed": 0,
            "completed": 0,
        }

    active_backend = backend
    if active_backend is None:
        active_backend = (
            SyntheticScoringBackend(args.model_id)
            if args.dry_run
            else FixedContinuationBackend(args.model_id, args.seed)
        )
    run_id = args.run_id or f"qwen_fixed_continuation_seed{args.seed}"
    summary = score_records(
        records=records,
        backend=active_backend,
        output_path=Path(args.output),
        run_id=run_id,
        chunk_size=args.chunk_size,
        resume=args.resume,
    )
    print(
        "Fixed-continuation scoring complete; "
        f"scenarios={summary['input_scenarios']}, "
        f"new={summary['generated']}, resumed={summary['resumed']}, "
        f"completed={summary['completed']}."
    )
    print("No private text or token IDs were printed.")
    return summary


def main() -> None:
    try:
        run(parse_args())
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    except KeyboardInterrupt:
        raise SystemExit(
            "Interrupted; completed atomic checkpoints remain resumable."
        ) from None


if __name__ == "__main__":
    main()
