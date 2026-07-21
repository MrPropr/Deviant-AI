"""Score saved model responses with teacher forcing without exposing private text."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .run_local_model import load_yaml, resolve_settings, tokenize_chat_messages
from .validate_prompt_variants import (
    CONDITION_TURN_COUNTS,
    is_git_ignored,
    validate_records,
)


SCRIPT_VERSION = "1.0.0"

ALLOWED_FINISH_REASONS = {
    "eos",
    "length",
    "other",
}

FORBIDDEN_OUTPUT_FIELDS = {
    "prompt",
    "prompt_text",
    "response",
    "response_text",
    "judge_notes",
    "raw_output",
    "conversation_text",
    "harmful_prompt",
    "messages",
    "logits",
    "token_logprobs",
    "top_tokens",
    "top_k_tokens",
}

REQUIRED_RESPONSE_FIELDS = {
    "response_id",
    "scenario_id",
    "condition",
    "turn_index",
    "label",
    "model_id",
    "run_id",
    "generation_status",
    "response_text",
    "generated_token_count",
    "hit_max_new_tokens",
    "finish_reason",
}


@dataclass
class ExperimentData:
    prompt_rows: list[dict]
    response_rows: list[dict]
    prompts_by_id: dict[str, dict]
    responses_by_key: dict[tuple[str, str, int], dict]
    summary: dict[str, Any]


@dataclass
class RunningTokenMetrics:
    count: int = 0
    sum_logprob: float = 0.0
    sum_entropy: float = 0.0
    sum_margin: float = 0.0
    minimum_logprob: float = math.inf
    maximum_logprob: float = -math.inf
    first_logprobs: list[float] = field(default_factory=list)

    def update(
        self,
        logits: Any,
        target_ids: Any,
        torch_module: Any,
    ) -> None:
        """Add metrics for one or more next-token predictions."""

        torch = torch_module

        if logits.ndim == 3:
            logits = logits.reshape(-1, logits.shape[-1])

        target_ids = target_ids.reshape(-1)

        if logits.shape[0] != target_ids.shape[0]:
            raise ValueError(
                "Logit and target lengths do not match: "
                f"{logits.shape[0]} != {target_ids.shape[0]}"
            )

        float_logits = logits.float()
        log_probs = torch.log_softmax(float_logits, dim=-1)

        selected = log_probs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)

        probabilities = log_probs.exp()

        entropy = -(
            probabilities * log_probs
        ).sum(dim=-1)

        top_two_logprobs = torch.topk(
            log_probs,
            k=2,
            dim=-1,
        ).values

        top_two_probabilities = top_two_logprobs.exp()

        margins = (
            top_two_probabilities[:, 0]
            - top_two_probabilities[:, 1]
        )

        selected_values = (
            selected.detach()
            .cpu()
            .double()
            .tolist()
        )

        entropy_sum = float(
            entropy.detach()
            .double()
            .sum()
            .item()
        )

        margin_sum = float(
            margins.detach()
            .double()
            .sum()
            .item()
        )

        for value in selected_values:
            numeric_value = float(value)

            if not math.isfinite(numeric_value):
                raise ValueError(
                    "Encountered non-finite token log-probability"
                )

            self.count += 1
            self.sum_logprob += numeric_value
            self.minimum_logprob = min(
                self.minimum_logprob,
                numeric_value,
            )
            self.maximum_logprob = max(
                self.maximum_logprob,
                numeric_value,
            )

            if len(self.first_logprobs) < 32:
                self.first_logprobs.append(
                    numeric_value
                )

        self.sum_entropy += entropy_sum
        self.sum_margin += margin_sum

        del float_logits
        del log_probs
        del probabilities
        del entropy
        del top_two_logprobs
        del top_two_probabilities
        del margins
        del selected

    def mean_first(self, token_count: int) -> float:
        values = self.first_logprobs[:token_count]

        if not values:
            raise ValueError(
                "No token log-probabilities are available"
            )

        return sum(values) / len(values)

    def finalize(self) -> dict[str, float | int]:
        if self.count < 1:
            raise ValueError(
                "Cannot finalize empty token metrics"
            )

        mean_logprob = (
            self.sum_logprob / self.count
        )

        geometric_mean_probability = math.exp(
            mean_logprob
        )

        perplexity = math.exp(
            -mean_logprob
        )

        result = {
            "scored_token_count": self.count,
            "sum_token_logprob": self.sum_logprob,
            "mean_token_logprob": mean_logprob,
            "geometric_mean_token_probability": (
                geometric_mean_probability
            ),
            "perplexity": perplexity,
            "mean_first_8_token_logprob": (
                self.mean_first(8)
            ),
            "mean_first_16_token_logprob": (
                self.mean_first(16)
            ),
            "mean_first_32_token_logprob": (
                self.mean_first(32)
            ),
            "mean_token_entropy": (
                self.sum_entropy / self.count
            ),
            "mean_top1_top2_probability_margin": (
                self.sum_margin / self.count
            ),
            "minimum_token_logprob": (
                self.minimum_logprob
            ),
            "maximum_token_logprob": (
                self.maximum_logprob
            ),
        }

        for key, value in result.items():
            if isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError(
                        f"Metric {key} is not finite"
                    )

        return result


def existing_file(value: str) -> Path:
    path = Path(value)

    if not path.exists():
        raise argparse.ArgumentTypeError(
            f"Input file does not exist: {path}"
        )

    if not path.is_file():
        raise argparse.ArgumentTypeError(
            f"Input path is not a file: {path}"
        )

    return path


def output_file(value: str) -> Path:
    path = Path(value)

    if path.exists() and path.is_dir():
        raise argparse.ArgumentTypeError(
            f"Output path is a directory: {path}"
        )

    return path


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at "
                    f"{path.name}:{line_number}"
                ) from exc

            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL row is not an object at "
                    f"{path.name}:{line_number}"
                )

            records.append(value)

    return records


def append_jsonl(
    path: Path,
    record: dict,
) -> None:
    existing = (
        read_jsonl(path)
        if path.exists()
        else []
    )
    existing.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            for row in existing:
                handle.write(
                    json.dumps(
                        row,
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
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def is_inside(
    child: Path,
    parent: Path,
) -> bool:
    try:
        child.resolve().relative_to(
            parent.resolve()
        )
        return True
    except ValueError:
        return False


def validate_private_paths(
    prompt_path: Path,
    response_path: Path,
    output_path: Path,
    private_root: Path,
    repo_root: Path,
) -> None:
    if not private_root.exists():
        raise ValueError(
            f"Private root does not exist: "
            f"{private_root}"
        )

    for path, description in (
        (prompt_path, "prompt input"),
        (response_path, "response input"),
    ):
        if not is_inside(path, private_root):
            raise ValueError(
                f"{description} must be inside "
                f"the private root"
            )

    output_is_private = is_inside(
        output_path,
        private_root,
    )

    output_is_ignored = is_git_ignored(
        output_path,
        repo_root,
    )

    if not output_is_private and not output_is_ignored:
        raise ValueError(
            "Output must be inside --private-root "
            "or inside a git-ignored repository path"
        )


def response_key(
    scenario_id: str,
    condition: str,
    turn_index: int,
) -> tuple[str, str, int]:
    return (
        scenario_id,
        condition,
        turn_index,
    )


def validate_experiment(
    prompt_rows: list[dict],
    response_rows: list[dict],
    expected_model_id: str,
    expected_run_id: str,
) -> ExperimentData:
    errors = validate_records(
        prompt_rows,
        require_ready=True,
    )

    prompts_by_id: dict[str, dict] = {}

    for prompt_row_number, prompt_row in enumerate(prompt_rows, start=1):
        scenario_id = str(
            prompt_row.get(
                "scenario_id",
                "",
            )
        ).strip()

        if not scenario_id:
            continue

        if scenario_id in prompts_by_id:
            errors.append(
                f"prompt row {prompt_row_number}: "
                "duplicate scenario_id"
            )
        else:
            prompts_by_id[scenario_id] = (
                prompt_row
            )

    expected_keys: set[
        tuple[str, str, int]
    ] = set()

    for scenario_id, prompt_row in (
        prompts_by_id.items()
    ):
        conditions = prompt_row["conditions"]

        for condition, expected_turns in (
            CONDITION_TURN_COUNTS.items()
        ):
            specification = conditions.get(
                condition,
                {},
            )

            turns = specification.get(
                "turns",
                [],
            )

            if len(turns) != expected_turns:
                continue

            for turn_index in range(
                1,
                expected_turns + 1,
            ):
                expected_keys.add(
                    response_key(
                        scenario_id,
                        condition,
                        turn_index,
                    )
                )

    responses_by_key: dict[
        tuple[str, str, int],
        dict,
    ] = {}

    response_ids: set[str] = set()

    for row_number, row in enumerate(
        response_rows,
        start=1,
    ):
        row_reference = f"response row {row_number}"
        missing_fields = sorted(
            REQUIRED_RESPONSE_FIELDS - set(row)
        )

        if missing_fields:
            errors.append(
                f"response row {row_number}: "
                f"missing fields {missing_fields}"
            )

        scenario_id = str(
            row.get("scenario_id", "")
        ).strip()

        condition = str(
            row.get("condition", "")
        ).strip()

        try:
            turn_index = int(
                row.get("turn_index", 0)
            )
        except (TypeError, ValueError):
            turn_index = 0

        key = response_key(
            scenario_id,
            condition,
            turn_index,
        )

        if key in responses_by_key:
            errors.append(
                f"{row_reference}: duplicate response key"
            )
        else:
            responses_by_key[key] = row

        response_id = str(
            row.get("response_id", "")
        ).strip()

        if not response_id:
            errors.append(
                f"response row {row_number}: "
                "missing response_id"
            )
        elif response_id in response_ids:
            errors.append(
                f"{row_reference}: duplicate response_id"
            )
        else:
            response_ids.add(response_id)

        expected_response_id = (
            f"{scenario_id}::{condition}::"
            f"turn_{turn_index}"
        )

        if (
            response_id
            and response_id
            != expected_response_id
        ):
            errors.append(
                f"{row_reference}: response_id "
                "does not match its key"
            )

        if condition not in CONDITION_TURN_COUNTS:
            errors.append(
                f"{row_reference}: invalid condition"
            )

        if key not in expected_keys:
            errors.append(
                f"{row_reference}: response key "
                "does not exist in prompts"
            )

        prompt_row = prompts_by_id.get(
            scenario_id
        )

        if prompt_row is not None:
            expected_label = str(
                prompt_row.get("label", "")
            )

            if (
                str(row.get("label", ""))
                != expected_label
            ):
                errors.append(
                    f"{row_reference}: label mismatch"
                )

        if row.get("model_id") != expected_model_id:
            errors.append(
                f"{row_reference}: model_id mismatch"
            )

        if row.get("run_id") != expected_run_id:
            errors.append(
                f"{row_reference}: run_id mismatch"
            )

        if row.get("generation_status") != "ok":
            errors.append(
                f"{row_reference}: generation_status "
                "must be ok"
            )

        response_text = row.get(
            "response_text"
        )

        if (
            not isinstance(response_text, str)
            or not response_text
        ):
            errors.append(
                f"{row_reference}: empty response_text"
            )

        finish_reason = row.get(
            "finish_reason"
        )

        if finish_reason not in (
            ALLOWED_FINISH_REASONS
        ):
            errors.append(
                f"{row_reference}: invalid "
                "finish_reason"
            )

        generated_token_count = row.get(
            "generated_token_count"
        )

        if (
            isinstance(
                generated_token_count,
                bool,
            )
            or not isinstance(
                generated_token_count,
                int,
            )
            or generated_token_count < 1
        ):
            errors.append(
                f"{row_reference}: invalid "
                "generated_token_count"
            )

        hit_max = row.get(
            "hit_max_new_tokens"
        )

        if not isinstance(hit_max, bool):
            errors.append(
                f"{row_reference}: "
                "hit_max_new_tokens must be boolean"
            )

        if (
            finish_reason == "length"
            and hit_max is not True
        ):
            errors.append(
                f"{row_reference}: length response "
                "must have hit_max_new_tokens=true"
            )

        if (
            finish_reason == "eos"
            and hit_max is True
        ):
            errors.append(
                f"{row_reference}: eos response "
                "must not hit max_new_tokens"
            )

        expected_context_count = None

        if condition in {
            "direct",
            "polite",
        }:
            expected_context_count = 1
        elif condition in {
            "multi_turn",
            "polite_multi_turn",
        }:
            expected_context_count = (
                2 * turn_index - 1
            )

        saved_context_count = row.get(
            "context_message_count"
        )

        if (
            saved_context_count is not None
            and expected_context_count is not None
            and saved_context_count
            != expected_context_count
        ):
            errors.append(
                f"{row_reference}: "
                "context_message_count mismatch"
            )

    actual_keys = set(responses_by_key)

    missing_keys = (
        expected_keys - actual_keys
    )

    extra_keys = (
        actual_keys - expected_keys
    )

    if missing_keys:
        errors.append(
            f"missing {len(missing_keys)} "
            "response keys"
        )

    if extra_keys:
        errors.append(
            f"found {len(extra_keys)} "
            "extra response keys"
        )

    for key_number, (
        scenario_id,
        condition,
        turn_index,
    ) in enumerate(sorted(actual_keys), start=1):
        if condition not in {
            "multi_turn",
            "polite_multi_turn",
        }:
            continue

        for previous_turn in range(
            1,
            turn_index,
        ):
            previous_key = response_key(
                scenario_id,
                condition,
                previous_turn,
            )

            if previous_key not in actual_keys:
                errors.append(
                    f"response key {key_number}: missing "
                    f"previous turn {previous_turn}"
                )

    if errors:
        preview = "\n".join(
            f"- {error}"
            for error in errors[:20]
        )

        raise ValueError(
            f"Structural validation found "
            f"{len(errors)} errors.\n{preview}"
        )

    condition_counts = Counter(
        str(row["condition"])
        for row in response_rows
    )

    label_counts = Counter(
        str(row["label"])
        for row in response_rows
    )

    finish_counts = Counter(
        str(row["finish_reason"])
        for row in response_rows
    )

    summary = {
        "prompt_scenarios": len(
            prompt_rows
        ),
        "response_rows": len(
            response_rows
        ),
        "unique_response_keys": len(
            actual_keys
        ),
        "condition_counts": dict(
            condition_counts
        ),
        "label_counts": dict(
            label_counts
        ),
        "finish_reason_counts": dict(
            finish_counts
        ),
    }

    return ExperimentData(
        prompt_rows=prompt_rows,
        response_rows=response_rows,
        prompts_by_id=prompts_by_id,
        responses_by_key=responses_by_key,
        summary=summary,
    )


def build_context_messages(
    scenario_id: str,
    condition: str,
    turn_index: int,
    prompts_by_id: dict[str, dict],
    responses_by_key: dict[
        tuple[str, str, int],
        dict,
    ],
) -> list[dict]:
    prompt_row = prompts_by_id[
        scenario_id
    ]

    specification = prompt_row[
        "conditions"
    ][condition]

    turns = specification["turns"]

    if turn_index < 1:
        raise ValueError(
            "turn_index must be positive"
        )

    if turn_index > len(turns):
        raise ValueError(
            "turn_index exceeds prompt turns"
        )

    messages: list[dict] = []

    for current_turn in range(
        1,
        turn_index + 1,
    ):
        user_text = turns[
            current_turn - 1
        ]["content"]

        messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        if current_turn < turn_index:
            prior_key = response_key(
                scenario_id,
                condition,
                current_turn,
            )

            prior_response = (
                responses_by_key[
                    prior_key
                ]["response_text"]
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": prior_response,
                }
            )

    expected_message_count = (
        2 * turn_index - 1
    )

    if len(messages) != expected_message_count:
        raise ValueError(
            "Context reconstruction produced "
            "an unexpected message count"
        )

    return messages


def package_version(
    package_name: str,
) -> str | None:
    try:
        return importlib.metadata.version(
            package_name
        )
    except importlib.metadata.PackageNotFoundError:
        return None


class TeacherForcingBackend:
    def __init__(
        self,
        settings: dict,
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                set_seed,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install torch, transformers, "
                "accelerate and bitsandbytes"
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required for scoring"
            )

        self.torch = torch
        self.model_id = settings["model_id"]
        self.settings = dict(settings)

        set_seed(settings["seed"])

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_id
            )
        )

        if not getattr(
            self.tokenizer,
            "chat_template",
            None,
        ):
            raise RuntimeError(
                "Tokenizer does not provide "
                "the required chat template"
            )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id
            )

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }

        if settings["load_in_4bit"]:
            compute_dtype = getattr(
                torch,
                settings["compute_dtype"],
            )

            model_kwargs[
                "quantization_config"
            ] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=settings[
                    "quantization_type"
                ],
                bnb_4bit_compute_dtype=(
                    compute_dtype
                ),
                bnb_4bit_use_double_quant=True,
            )
        else:
            model_kwargs["torch_dtype"] = (
                getattr(
                    torch,
                    settings["compute_dtype"],
                )
            )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        )

        self.model.eval()
        self.model.config.use_cache = True

        self.device = (
            self.model
            .get_input_embeddings()
            .weight
            .device
        )

        self.chat_template_sha256 = (
            sha256_text(
                str(
                    self.tokenizer.chat_template
                )
            )
        )

    def process_prefix(
        self,
        prompt_ids: Any,
        chunk_size: int,
    ) -> tuple[Any, Any]:
        torch = self.torch

        past_key_values = None
        next_token_logits = None

        with torch.inference_mode():
            for start in range(
                0,
                prompt_ids.shape[-1],
                chunk_size,
            ):
                end = min(
                    start + chunk_size,
                    prompt_ids.shape[-1],
                )

                input_chunk = (
                    prompt_ids[:, start:end]
                )

                outputs = self.model(
                    input_ids=input_chunk,
                    past_key_values=(
                        past_key_values
                    ),
                    use_cache=True,
                    return_dict=True,
                )

                past_key_values = (
                    outputs.past_key_values
                )

                next_token_logits = (
                    outputs.logits[:, -1, :]
                )

                del outputs
                del input_chunk

        if next_token_logits is None:
            raise ValueError(
                "Prompt tokenization produced "
                "no tokens"
            )

        return (
            past_key_values,
            next_token_logits,
        )

    def score_response(
        self,
        messages: list[dict],
        response_text: str,
        finish_reason: str,
        chunk_size: int,
    ) -> dict[str, Any]:
        torch = self.torch

        prompt_inputs = tokenize_chat_messages(
            self.tokenizer,
            messages,
            tokenize_directly=self.settings.get(
                "tokenize_chat_template_directly",
                False,
            ),
        )

        prompt_ids = (
            prompt_inputs["input_ids"]
            .to(self.device)
        )

        response_inputs = self.tokenizer(
            response_text,
            add_special_tokens=False,
            return_tensors="pt",
        )

        response_ids = (
            response_inputs["input_ids"]
            .to(self.device)
        )

        prompt_token_count = int(
            prompt_ids.shape[-1]
        )

        response_token_count = int(
            response_ids.shape[-1]
        )

        del prompt_inputs
        del response_inputs

        if response_token_count < 1:
            raise ValueError(
                "Retokenized response is empty"
            )

        (
            past_key_values,
            next_token_logits,
        ) = self.process_prefix(
            prompt_ids=prompt_ids,
            chunk_size=chunk_size,
        )

        del prompt_ids

        metrics = RunningTokenMetrics()

        with torch.inference_mode():
            for start in range(
                0,
                response_token_count,
                chunk_size,
            ):
                end = min(
                    start + chunk_size,
                    response_token_count,
                )

                response_chunk = (
                    response_ids[:, start:end]
                )

                metrics.update(
                    logits=next_token_logits,
                    target_ids=(
                        response_chunk[:, 0]
                    ),
                    torch_module=torch,
                )

                outputs = self.model(
                    input_ids=response_chunk,
                    past_key_values=(
                        past_key_values
                    ),
                    use_cache=True,
                    return_dict=True,
                )

                past_key_values = (
                    outputs.past_key_values
                )

                chunk_length = int(
                    response_chunk.shape[-1]
                )

                if chunk_length > 1:
                    continuation_logits = (
                        outputs.logits[:, :-1, :]
                    )

                    continuation_targets = (
                        response_chunk[:, 1:]
                    )

                    metrics.update(
                        logits=(
                            continuation_logits
                        ),
                        target_ids=(
                            continuation_targets
                        ),
                        torch_module=torch,
                    )

                    del continuation_logits
                    del continuation_targets

                next_token_logits = (
                    outputs.logits[:, -1, :]
                )

                del outputs
                del response_chunk

        terminal_eos_logprob = None

        if finish_reason == "eos":
            eos_token_id = (
                self.tokenizer.eos_token_id
            )

            if eos_token_id is None:
                raise ValueError(
                    "Tokenizer has no eos_token_id"
                )

            terminal_log_probs = (
                torch.log_softmax(
                    next_token_logits.float(),
                    dim=-1,
                )
            )

            terminal_eos_logprob = float(
                terminal_log_probs[
                    0,
                    eos_token_id,
                ]
                .detach()
                .double()
                .item()
            )

            if not math.isfinite(
                terminal_eos_logprob
            ):
                raise ValueError(
                    "EOS log-probability "
                    "is not finite"
                )

            del terminal_log_probs

        result = metrics.finalize()

        result.update(
            {
                "prompt_token_count": (
                    prompt_token_count
                ),
                "retokenized_response_token_count": (
                    response_token_count
                ),
                "terminal_eos_logprob": (
                    terminal_eos_logprob
                ),
            }
        )

        del response_ids
        del next_token_logits
        del past_key_values
        del metrics

        return result

    def metadata(
        self,
        prompt_path: Path,
        response_path: Path,
        config_path: Path,
        run_id: str,
        chunk_size: int,
    ) -> dict[str, Any]:
        torch = self.torch

        model_revision = getattr(
            self.model.config,
            "_commit_hash",
            None,
        )

        tokenizer_revision = (
            getattr(
                self.tokenizer,
                "init_kwargs",
                {},
            ).get("_commit_hash")
        )

        return {
            "script_version": SCRIPT_VERSION,
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "model_id": self.model_id,
            "run_id": run_id,
            "model_revision": model_revision,
            "tokenizer_revision": (
                tokenizer_revision
            ),
            "chat_template_sha256": (
                self.chat_template_sha256
            ),
            "config_file_name": (
                config_path.name
            ),
            "config_sha256": sha256_file(
                config_path
            ),
            "prompt_file_name": (
                prompt_path.name
            ),
            "prompt_sha256": sha256_file(
                prompt_path
            ),
            "response_file_name": (
                response_path.name
            ),
            "response_sha256": sha256_file(
                response_path
            ),
            "load_in_4bit": self.settings[
                "load_in_4bit"
            ],
            "quantization_type": (
                self.settings[
                    "quantization_type"
                ]
            ),
            "compute_dtype": self.settings[
                "compute_dtype"
            ],
            "double_quantization": True,
            "chunk_size": chunk_size,
            "scoring_method": (
                "teacher_forcing_on_"
                "retokenized_saved_response_v1"
            ),
            "python_version": (
                platform.python_version()
            ),
            "torch_version": (
                torch.__version__
            ),
            "transformers_version": (
                package_version(
                    "transformers"
                )
            ),
            "accelerate_version": (
                package_version(
                    "accelerate"
                )
            ),
            "bitsandbytes_version": (
                package_version(
                    "bitsandbytes"
                )
            ),
            "cuda_available": (
                torch.cuda.is_available()
            ),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        }


def validate_safe_output_record(
    record: dict,
) -> None:
    forbidden = sorted(
        set(record) & FORBIDDEN_OUTPUT_FIELDS
    )

    if forbidden:
        raise ValueError(
            "Output record contains forbidden "
            f"fields: {forbidden}"
        )

    for key, value in record.items():
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(
                    f"Output field {key} "
                    "is not finite"
                )


def build_output_record(
    source_row: dict,
    scored_metrics: dict[str, Any],
    chat_template_sha256: str,
) -> dict[str, Any]:
    original_token_count = int(
        source_row[
            "generated_token_count"
        ]
    )

    finish_reason = str(
        source_row["finish_reason"]
    )

    expected_text_token_count = (
        original_token_count - 1
        if finish_reason == "eos"
        else original_token_count
    )

    scored_token_count = int(
        scored_metrics[
            "scored_token_count"
        ]
    )

    record = {
        "response_id": source_row[
            "response_id"
        ],
        "scenario_id": source_row[
            "scenario_id"
        ],
        "condition": source_row[
            "condition"
        ],
        "turn_index": int(
            source_row["turn_index"]
        ),
        "label": source_row["label"],
        "model_id": source_row[
            "model_id"
        ],
        "run_id": source_row["run_id"],
        "scoring_method": (
            "teacher_forcing_on_"
            "retokenized_saved_response_v1"
        ),
        "chat_template_sha256": (
            chat_template_sha256
        ),
        "scoring_status": "ok",
        "original_generated_token_count": (
            original_token_count
        ),
        "expected_text_token_count": (
            expected_text_token_count
        ),
        "token_count_delta_from_expected": (
            scored_token_count
            - expected_text_token_count
        ),
        "finish_reason": finish_reason,
        "hit_max_new_tokens": bool(
            source_row[
                "hit_max_new_tokens"
            ]
        ),
        "context_message_count": int(
            source_row.get(
                "context_message_count",
                2 * int(
                    source_row["turn_index"]
                ) - 1,
            )
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        **scored_metrics,
    }

    validate_safe_output_record(record)

    return record


def load_completed_ids(
    output_path: Path,
) -> set[str]:
    if not output_path.exists():
        return set()

    completed: set[str] = set()

    for row in read_jsonl(output_path):
        validate_safe_output_record(row)

        response_id = str(
            row.get("response_id", "")
        )

        if not response_id:
            raise ValueError(
                "Resume file contains a row "
                "without response_id"
            )

        if response_id in completed:
            raise ValueError(
                "Resume file contains a duplicate response_id"
            )

        if row.get("scoring_status") != "ok":
            raise ValueError(
                "Resume file contains a row "
                "without scoring_status=ok"
            )

        completed.add(response_id)

    return completed


def metadata_path_for(
    output_path: Path,
) -> Path:
    return Path(
        str(output_path) + ".meta.json"
    )


def write_or_validate_metadata(
    metadata_path: Path,
    metadata: dict[str, Any],
    resume: bool,
) -> None:
    comparison_fields = {
        "model_id",
        "run_id",
        "model_revision",
        "tokenizer_revision",
        "chat_template_sha256",
        "config_sha256",
        "prompt_sha256",
        "response_sha256",
        "load_in_4bit",
        "quantization_type",
        "compute_dtype",
        "double_quantization",
        "scoring_method",
    }

    if resume and metadata_path.exists():
        existing = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        mismatches = [
            field
            for field in comparison_fields
            if existing.get(field)
            != metadata.get(field)
        ]

        if mismatches:
            raise ValueError(
                "Resume metadata mismatch for: "
                + ", ".join(
                    sorted(mismatches)
                )
            )

        return

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--config",
        type=existing_file,
        required=True,
    )

    parser.add_argument(
        "--prompts",
        type=existing_file,
        required=True,
    )

    parser.add_argument(
        "--responses",
        type=existing_file,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=output_file,
        required=True,
    )

    parser.add_argument(
        "--private-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--run-id",
        required=True,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of new response "
            "rows to score"
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    parser.add_argument(
        "--load-in-4bit",
        dest="load_in_4bit",
        action="store_true",
    )

    parser.add_argument(
        "--no-load-in-4bit",
        dest="load_in_4bit",
        action="store_false",
    )

    parser.set_defaults(
        load_in_4bit=None
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.chunk_size < 1:
        raise SystemExit(
            "--chunk-size must be positive"
        )

    if (
        args.limit is not None
        and args.limit < 1
    ):
        raise SystemExit(
            "--limit must be positive"
        )

    if args.resume and args.overwrite:
        raise SystemExit(
            "--resume and --overwrite "
            "cannot be combined"
        )

    repo_root = Path(
        __file__
    ).resolve().parents[1]

    try:
        validate_private_paths(
            prompt_path=args.prompts,
            response_path=args.responses,
            output_path=args.output,
            private_root=args.private_root,
            repo_root=repo_root,
        )

        settings = resolve_settings(
            load_yaml(args.config),
            args.load_in_4bit,
        )

        prompt_rows = read_jsonl(
            args.prompts
        )

        response_rows = read_jsonl(
            args.responses
        )

        experiment = validate_experiment(
            prompt_rows=prompt_rows,
            response_rows=response_rows,
            expected_model_id=settings[
                "model_id"
            ],
            expected_run_id=args.run_id,
        )

        summary = experiment.summary

        print(
            "Structural validation passed; "
            f"scenarios="
            f"{summary['prompt_scenarios']}, "
            f"responses="
            f"{summary['response_rows']}, "
            f"unique_keys="
            f"{summary['unique_response_keys']}."
        )

        print(
            "Condition counts:",
            summary["condition_counts"],
        )

        print(
            "Finish reason counts:",
            summary[
                "finish_reason_counts"
            ],
        )

        print(
            "No prompt or response text "
            "was printed."
        )

        if args.validate_only:
            print(
                "Validation-only run complete; "
                "the model was not loaded."
            )
            return

        metadata_path = (
            metadata_path_for(
                args.output
            )
        )

        if args.overwrite:
            if args.output.exists():
                args.output.unlink()

            if metadata_path.exists():
                metadata_path.unlink()

        if (
            args.output.exists()
            and not args.resume
            and not args.overwrite
        ):
            raise ValueError(
                "Output already exists. Use "
                "--resume or --overwrite."
            )

        completed_ids = (
            load_completed_ids(
                args.output
            )
            if args.resume
            else set()
        )

        print(
            "Loading the scoring model. "
            "Private text will not be printed."
        )

        backend = TeacherForcingBackend(
            settings
        )

        metadata = backend.metadata(
            prompt_path=args.prompts,
            response_path=args.responses,
            config_path=args.config,
            run_id=args.run_id,
            chunk_size=args.chunk_size,
        )

        write_or_validate_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            resume=args.resume,
        )

        new_scored = 0
        resumed = 0

        for source_row in (
            experiment.response_rows
        ):
            response_id = str(
                source_row["response_id"]
            )

            if response_id in completed_ids:
                resumed += 1
                continue

            if (
                args.limit is not None
                and new_scored >= args.limit
            ):
                break

            scenario_id = str(
                source_row["scenario_id"]
            )

            condition = str(
                source_row["condition"]
            )

            turn_index = int(
                source_row["turn_index"]
            )

            messages = (
                build_context_messages(
                    scenario_id=scenario_id,
                    condition=condition,
                    turn_index=turn_index,
                    prompts_by_id=(
                        experiment.prompts_by_id
                    ),
                    responses_by_key=(
                        experiment.responses_by_key
                    ),
                )
            )

            scored_metrics = (
                backend.score_response(
                    messages=messages,
                    response_text=source_row[
                        "response_text"
                    ],
                    finish_reason=source_row[
                        "finish_reason"
                    ],
                    chunk_size=args.chunk_size,
                )
            )

            output_record = (
                build_output_record(
                    source_row=source_row,
                    scored_metrics=(
                        scored_metrics
                    ),
                    chat_template_sha256=(
                        backend
                        .chat_template_sha256
                    ),
                )
            )

            append_jsonl(
                args.output,
                output_record,
            )

            completed_ids.add(
                response_id
            )

            new_scored += 1

            print(
                f"Saved scoring row "
                f"{new_scored}; "
                f"total_completed="
                f"{len(completed_ids)}."
            )

            del messages
            del scored_metrics
            del output_record

            if (
                new_scored % 10 == 0
                and backend.torch.cuda.is_available()
            ):
                backend.torch.cuda.empty_cache()

        print(
            "Scoring finished; "
            f"new_scored={new_scored}, "
            f"resumed={resumed}, "
            f"total_completed="
            f"{len(completed_ids)}, "
            f"output={args.output}."
        )

        print(
            "No prompt, response, token text "
            "or logits were saved."
        )

    except KeyboardInterrupt as exc:
        print(
            "Interrupted safely. Completed "
            "numeric rows remain saved."
        )
        raise SystemExit(130) from exc

    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(
                "CUDA out of memory. Completed "
                "numeric rows remain saved. "
                "Restart with --resume and a "
                "smaller --chunk-size."
            )
            return

        raise SystemExit(
            f"Error: {exc}"
        ) from exc

    except ValueError as exc:
        raise SystemExit(
            f"Error: {exc}"
        ) from exc


if __name__ == "__main__":
    main()
