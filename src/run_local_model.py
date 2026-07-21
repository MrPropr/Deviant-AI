"""Run a private local-model pilot with isolated, resumable conversations."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .validate_prompt_variants import CONDITION_TURN_COUNTS, is_git_ignored, validate_records


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
GEMMA_MODEL_ID = "google/gemma-2-9b-it"
MISTRAL_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

SUPPORTED_MODEL_IDS = (
    DEFAULT_MODEL_ID,
    GEMMA_MODEL_ID,
    MISTRAL_MODEL_ID,
)

DIRECT_CHAT_TEMPLATE_MODEL_IDS = frozenset(
    {
        GEMMA_MODEL_ID,
        MISTRAL_MODEL_ID,
    }
)
PLACEHOLDER_RESPONSE = "[MODEL_RESPONSE_PLACEHOLDER]"
SUPPORTED_COMPUTE_DTYPES = ("float16", "bfloat16", "float32")
FINISH_REASONS = ("eos", "length", "other", "dry_run")


@dataclass(frozen=True)
class GenerationResult:
    response_text: str
    generated_token_count: int
    hit_max_new_tokens: bool
    finish_reason: str

    def __post_init__(self) -> None:
        if isinstance(self.generated_token_count, bool) or not isinstance(self.generated_token_count, int):
            raise ValueError("generated_token_count must be an integer")
        if self.generated_token_count < 0:
            raise ValueError("generated_token_count must not be negative")
        if not isinstance(self.hit_max_new_tokens, bool):
            raise ValueError("hit_max_new_tokens must be boolean")
        if self.finish_reason not in FINISH_REASONS:
            allowed = ", ".join(FINISH_REASONS)
            raise ValueError(f"finish_reason must be one of: {allowed}")


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


def append_jsonl(path: Path, record: dict) -> None:
    """Append one JSONL record through an atomic file replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)

    existing_bytes = (
        path.read_bytes()
        if path.exists()
        else b""
    )

    if (
        existing_bytes
        and not existing_bytes.endswith(b"\n")
    ):
        raise ValueError(
            "Existing JSONL file does not end "
            "with a complete line"
        )

    new_line = (
        json.dumps(
            record,
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    file_descriptor, temporary_name = (
        tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
    )

    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(
            file_descriptor,
            "wb",
        ) as handle:
            handle.write(existing_bytes)
            handle.write(new_line)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(
            temporary_path,
            path,
        )

    finally:
        temporary_path.unlink(
            missing_ok=True
        )


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def count_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def token_id_to_int(token_id: Any) -> int:
    item = getattr(token_id, "item", None)
    return int(item() if callable(item) else token_id)


def generation_termination_metadata(
    generated_token_ids: Any,
    max_new_tokens: int,
    eos_token_ids: int | list[int] | tuple[int, ...] | None,
) -> tuple[int, bool, str]:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    generated_token_count = len(generated_token_ids)
    hit_max_new_tokens = generated_token_count >= max_new_tokens
    if eos_token_ids is None:
        normalized_eos_ids: set[int] = set()
    elif isinstance(eos_token_ids, int):
        normalized_eos_ids = {eos_token_ids}
    else:
        normalized_eos_ids = {token_id_to_int(token_id) for token_id in eos_token_ids}

    ended_with_eos = (
        generated_token_count > 0
        and token_id_to_int(generated_token_ids[-1]) in normalized_eos_ids
    )
    if ended_with_eos:
        finish_reason = "eos"
    elif hit_max_new_tokens:
        finish_reason = "length"
    else:
        finish_reason = "other"
    return generated_token_count, hit_max_new_tokens, finish_reason


def tokenize_chat_messages(
    tokenizer: Any,
    messages: list[dict],
    *,
    tokenize_directly: bool,
) -> Any:
    if tokenize_directly:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(prompt, return_tensors="pt")


def generation_eos_token_ids(tokenizer: Any, model_id: str) -> int | list[int] | None:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    values: list[int] = []
    if isinstance(eos_token_id, int):
        values.append(eos_token_id)
    elif eos_token_id is not None:
        values.extend(token_id_to_int(value) for value in eos_token_id)

    if model_id == GEMMA_MODEL_ID:
        end_of_turn_id = tokenizer.get_vocab().get("<end_of_turn>")
        if not isinstance(end_of_turn_id, int):
            raise RuntimeError("Gemma tokenizer does not define <end_of_turn>")
        values.append(end_of_turn_id)

    unique_values = list(dict.fromkeys(values))
    if not unique_values:
        return None
    if len(unique_values) == 1:
        return unique_values[0]
    return unique_values


def response_key(scenario_id: str, condition: str, turn_index: int) -> tuple[str, str, int]:
    return scenario_id, condition, turn_index


def load_completed(path: Path) -> dict[tuple[str, str, int], dict]:
    completed: dict[tuple[str, str, int], dict] = {}
    if not path.exists():
        return completed
    for record in read_jsonl(path):
        if record.get("generation_status") != "ok":
            continue
        key = response_key(
            str(record.get("scenario_id", "")),
            str(record.get("condition", "")),
            int(record.get("turn_index", 0)),
        )

        if key in completed:
            raise ValueError(
                "Resume output contains a duplicate "
                "completed response key"
            )

        completed[key] = record
    return completed


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install pyyaml before running the local model pipeline.") from exc
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must contain a YAML object: {path}")
    return loaded


def resolve_settings(config: dict, load_in_4bit_override: bool | None = None) -> dict:
    model_config = config.get("model", {})
    inference_config = config.get("inference", {})
    settings = {
        "model_id": model_config.get("id", DEFAULT_MODEL_ID),
        "load_in_4bit": bool(model_config.get("load_in_4bit", False)),
        "quantization_type": model_config.get("quantization_type", "nf4"),
        "compute_dtype": str(model_config.get("compute_dtype", "float16")).lower(),
        "max_new_tokens": int(inference_config.get("max_new_tokens", 512)),
        "do_sample": bool(inference_config.get("do_sample", False)),
        "seed": int(inference_config.get("seed", 42)),
        "max_multi_turn_length": int(inference_config.get("max_multi_turn_length", 3)),
    }
    if load_in_4bit_override is not None:
        settings["load_in_4bit"] = load_in_4bit_override
    if settings["model_id"] not in SUPPORTED_MODEL_IDS:
        raise ValueError(f"Unsupported model id: {settings['model_id']}")
    settings["tokenize_chat_template_directly"] = (
        settings["model_id"] in DIRECT_CHAT_TEMPLATE_MODEL_IDS
    )
    if settings["do_sample"]:
        raise ValueError("Pilot inference requires do_sample=false")
    if settings["max_multi_turn_length"] != 3:
        raise ValueError("Pilot max_multi_turn_length must be 3")
    if settings["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be positive")
    if settings["compute_dtype"] not in SUPPORTED_COMPUTE_DTYPES:
        supported = ", ".join(SUPPORTED_COMPUTE_DTYPES)
        raise ValueError(f"model.compute_dtype must be one of: {supported}")
    return settings


class DryRunBackend:
    """Deterministic placeholder backend used only for tests and plumbing checks."""

    model_id = "dry-run-placeholder-model"

    def generate(self, messages: list[dict], max_new_tokens: int, do_sample: bool) -> GenerationResult:
        del messages, max_new_tokens, do_sample
        return GenerationResult(
            response_text=PLACEHOLDER_RESPONSE,
            generated_token_count=0,
            hit_max_new_tokens=False,
            finish_reason="dry_run",
        )


class TransformersBackend:
    def __init__(self, settings: dict):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
        except ImportError as exc:
            raise RuntimeError("Install transformers, accelerate, torch, and optional bitsandbytes first.") from exc

        self.torch = torch
        self.model_id = settings["model_id"]
        self.tokenize_chat_template_directly = settings[
            "tokenize_chat_template_directly"
        ]
        set_seed(settings["seed"])
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if not getattr(tokenizer, "chat_template", None):
            raise RuntimeError("Tokenizer does not provide the required chat template")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model_kwargs: dict[str, Any] = {"device_map": "auto"}
        if settings["load_in_4bit"]:
            compute_dtype = getattr(torch, settings["compute_dtype"])
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=settings["quantization_type"],
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        self.model.eval()
        self.eos_token_ids = generation_eos_token_ids(tokenizer, self.model_id)

    def generate(self, messages: list[dict], max_new_tokens: int, do_sample: bool) -> GenerationResult:
        inputs = tokenize_chat_messages(
            self.tokenizer,
            messages,
            tokenize_directly=self.tokenize_chat_template_directly,
        )
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_token_ids,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        new_tokens = generated[0][prompt_length:]
        generated_token_count, hit_max_new_tokens, finish_reason = generation_termination_metadata(
            new_tokens,
            max_new_tokens=max_new_tokens,
            eos_token_ids=self.eos_token_ids,
        )
        return GenerationResult(
            response_text=self.tokenizer.decode(new_tokens, skip_special_tokens=True),
            generated_token_count=generated_token_count,
            hit_max_new_tokens=hit_max_new_tokens,
            finish_reason=finish_reason,
        )


def is_cuda_oom(exc: BaseException) -> bool:
    return exc.__class__.__name__ == "OutOfMemoryError" or "cuda out of memory" in str(exc).lower()


def clear_cuda_cache(backend: object) -> None:
    torch_module = getattr(backend, "torch", None)
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def build_response_record(
    scenario: dict,
    condition: str,
    turn: dict,
    generation_result: GenerationResult,
    model_id: str,
    run_id: str,
    context_message_count: int,
) -> dict:
    scenario_id = scenario["scenario_id"]
    turn_index = int(turn["turn_index"])
    return {
        "response_id": f"{scenario_id}::{condition}::turn_{turn_index}",
        "scenario_id": scenario_id,
        "condition": condition,
        "turn_index": turn_index,
        "label": scenario["label"],
        "dataset": scenario.get("dataset", {}),
        "model_id": model_id,
        "run_id": run_id,
        "generation_status": "ok",
        "response_text": generation_result.response_text,
        "response_word_count": count_words(generation_result.response_text),
        "response_token_count": count_tokens(generation_result.response_text),
        "generated_token_count": generation_result.generated_token_count,
        "hit_max_new_tokens": generation_result.hit_max_new_tokens,
        "finish_reason": generation_result.finish_reason,
        "input_word_count": turn["word_count"],
        "input_token_count": turn["token_count"],
        "context_message_count": context_message_count,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_prompt_records(
    records: list[dict],
    backend: object,
    output_path: Path,
    run_id: str,
    max_new_tokens: int,
    do_sample: bool,
    resume: bool = False,
    limit_pairs: int | None = None,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        output_path.write_text("", encoding="utf-8")
    completed = load_completed(output_path) if resume else {}
    expected_model_id = getattr(backend, "model_id", "unknown")
    for completed_record in completed.values():
        if completed_record.get("run_id") != run_id or completed_record.get("model_id") != expected_model_id:
            raise ValueError("Resume output contains a different run_id or model_id")
    generated_count = 0
    resumed_count = 0
    pair_count = 0

    for scenario in records:
        for condition in CONDITION_TURN_COUNTS:
            if limit_pairs is not None and pair_count >= limit_pairs:
                return {
                    "generated": generated_count,
                    "resumed": resumed_count,
                    "pairs": pair_count,
                    "stopped_reason": "limit",
                }
            pair_count += 1
            specification = scenario["conditions"][condition]
            messages: list[dict] = []
            for turn in specification["turns"]:
                messages.append({"role": "user", "content": turn["content"]})
                key = response_key(scenario["scenario_id"], condition, int(turn["turn_index"]))
                if key in completed:
                    prior_response = str(completed[key]["response_text"])
                    messages.append({"role": "assistant", "content": prior_response})
                    resumed_count += 1
                    continue
                try:
                    generation_result = backend.generate(
                        [dict(message) for message in messages],
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                    )
                except Exception as exc:
                    if not is_cuda_oom(exc):
                        raise
                    clear_cuda_cache(backend)
                    return {
                        "generated": generated_count,
                        "resumed": resumed_count,
                        "pairs": pair_count,
                        "stopped_reason": "cuda_oom",
                    }

                response_record = build_response_record(
                    scenario=scenario,
                    condition=condition,
                    turn=turn,
                    generation_result=generation_result,
                    model_id=expected_model_id,
                    run_id=run_id,
                    context_message_count=len(messages),
                )
                append_jsonl(output_path, response_record)
                completed[key] = response_record
                generated_count += 1
                messages.append({"role": "assistant", "content": generation_result.response_text})

    return {
        "generated": generated_count,
        "resumed": resumed_count,
        "pairs": pair_count,
        "stopped_reason": "complete",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=existing_file, default=Path("configs/pilot_qwen.yaml"))
    parser.add_argument("--input", type=existing_file, required=True, help="Validated private prompt JSONL file.")
    parser.add_argument(
        "--output",
        type=output_file,
        default=Path("results/raw/pilot_qwen_outputs.jsonl"),
        help="Ignored private raw-output JSONL path.",
    )
    parser.add_argument("--run-id", default="pilot_qwen_seed42", help="Identifier stored with every response.")
    parser.add_argument("--resume", action="store_true", help="Resume without duplicating completed turn responses.")
    parser.add_argument("--limit", type=int, default=None, help="Optional scenario-condition pair limit.")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic placeholder responses; no model load.")
    parser.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true")
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive when provided")
    repo_root = Path(__file__).resolve().parents[1]
    if not is_git_ignored(args.output, repo_root):
        raise SystemExit("Raw output path must be ignored by git")

    try:
        records = read_jsonl(args.input)
        validation_errors = validate_records(records, require_ready=not args.dry_run)
        if validation_errors:
            raise ValueError(f"prompt validation found {len(validation_errors)} structural errors")
        settings = resolve_settings(load_yaml(args.config), args.load_in_4bit)
        backend = DryRunBackend() if args.dry_run else TransformersBackend(settings)
        result = run_prompt_records(
            records=records,
            backend=backend,
            output_path=args.output,
            run_id=args.run_id,
            max_new_tokens=settings["max_new_tokens"],
            do_sample=settings["do_sample"],
            resume=args.resume,
            limit_pairs=args.limit,
        )
    except KeyboardInterrupt as exc:
        print(f"Interrupted safely; completed responses remain saved at {args.output}. No response text was printed.")
        raise SystemExit(130) from exc
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    if result["stopped_reason"] == "cuda_oom":
        print(
            f"Stopped safely after CUDA out-of-memory; {result['generated']} new responses remain saved. "
            "Reduce memory use and rerun with --resume."
        )
        return
    print(
        f"Local run status={result['stopped_reason']}; generated={result['generated']}, "
        f"resumed={result['resumed']}, pairs={result['pairs']}, output={args.output}. "
        "No prompt or response text was printed."
    )


if __name__ == "__main__":
    main()
