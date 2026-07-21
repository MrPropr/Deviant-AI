from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.generate_prompt_variants import generate_templates
from src.run_local_model import (
    DryRunBackend,
    GEMMA_MODEL_ID,
    MISTRAL_MODEL_ID,
    GenerationResult,
    append_jsonl,
    generation_eos_token_ids,
    generation_termination_metadata,
    load_completed,
    load_yaml,
    resolve_settings,
    run_prompt_records,
    tokenize_chat_messages,
)


class RecordingBackend:
    model_id = "recording-test-model"

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def generate(self, messages: list[dict], max_new_tokens: int, do_sample: bool) -> GenerationResult:
        assert max_new_tokens == 8
        assert do_sample is False
        self.calls.append([dict(message) for message in messages])
        return GenerationResult(
            response_text="[MODEL_RESPONSE_PLACEHOLDER]",
            generated_token_count=3,
            hit_max_new_tokens=False,
            finish_reason="other",
        )


def scenario_record(scenario_id: str) -> dict:
    return {
        "scenario_id": scenario_id,
        "behavior_id": scenario_id,
        "label": "harmful",
        "split": "harmful",
        "dataset": "JailbreakBench/JBB-Behaviors",
        "dataset_config": "behaviors",
        "jbb_index": 1,
        "source": "sanitized_test",
        "risk_area": "placeholder",
        "behavior_label": "placeholder",
        "behavior": "[HARMFUL_BEHAVIOR_PLACEHOLDER]",
        "private_raw_text": True,
    }


def read_output(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pilot_config(compute_dtype: str = "float16") -> dict:
    return {
        "model": {
            "id": "Qwen/Qwen2.5-7B-Instruct",
            "load_in_4bit": True,
            "quantization_type": "nf4",
            "compute_dtype": compute_dtype,
        },
        "inference": {
            "max_new_tokens": 512,
            "do_sample": False,
            "seed": 42,
            "max_multi_turn_length": 3,
        },
    }


class RecordingTokenizer:
    eos_token_id = 1

    def __init__(self) -> None:
        self.apply_calls: list[dict] = []
        self.tokenizer_calls: list[tuple[str, str]] = []

    def apply_chat_template(self, messages: list[dict], **kwargs):
        del messages
        self.apply_calls.append(dict(kwargs))
        if kwargs["tokenize"]:
            return {"input_ids": "direct-token-ids"}
        return "[SANITIZED_CHAT_TEMPLATE]"

    def __call__(self, text: str, return_tensors: str):
        self.tokenizer_calls.append((text, return_tensors))
        return {"input_ids": "retokenized-input-ids"}

    def get_vocab(self) -> dict[str, int]:
        return {"<end_of_turn>": 107}


def test_compute_dtype_is_loaded_from_config() -> None:
    assert resolve_settings(pilot_config())["compute_dtype"] == "float16"


def test_gemma_expanded_config_is_supported() -> None:
    settings = resolve_settings(
        load_yaml(Path("configs/gemma_expanded_pilot.yaml"))
    )
    assert settings == {
        "model_id": GEMMA_MODEL_ID,
        "load_in_4bit": True,
        "quantization_type": "nf4",
        "compute_dtype": "float16",
        "max_new_tokens": 1024,
        "do_sample": False,
        "seed": 42,
        "max_multi_turn_length": 3,
        "tokenize_chat_template_directly": True,
    }


def test_mistral_expanded_config_is_supported() -> None:
    settings = resolve_settings(
        load_yaml(
            Path(
                "configs/"
                "mistral7b_v03_expanded_pilot.yaml"
            )
        )
    )

    assert settings == {
        "model_id": MISTRAL_MODEL_ID,
        "load_in_4bit": True,
        "quantization_type": "nf4",
        "compute_dtype": "float16",
        "max_new_tokens": 1024,
        "do_sample": False,
        "seed": 42,
        "max_multi_turn_length": 3,
        "tokenize_chat_template_directly": True,
    }


def test_unsupported_model_is_rejected() -> None:
    config = pilot_config()
    config["model"]["id"] = "unsupported/model"
    with pytest.raises(ValueError, match="Unsupported model id"):
        resolve_settings(config)


def test_gemma_chat_template_is_tokenized_directly() -> None:
    tokenizer = RecordingTokenizer()
    output = tokenize_chat_messages(
        tokenizer,
        [{"role": "user", "content": "[BENIGN_BEHAVIOR_PLACEHOLDER]"}],
        tokenize_directly=True,
    )
    assert output == {"input_ids": "direct-token-ids"}
    assert tokenizer.apply_calls == [
        {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
    ]
    assert tokenizer.tokenizer_calls == []


def test_qwen_chat_template_path_remains_unchanged() -> None:
    tokenizer = RecordingTokenizer()
    output = tokenize_chat_messages(
        tokenizer,
        [{"role": "user", "content": "[BENIGN_BEHAVIOR_PLACEHOLDER]"}],
        tokenize_directly=False,
    )
    assert output == {"input_ids": "retokenized-input-ids"}
    assert tokenizer.apply_calls == [
        {"tokenize": False, "add_generation_prompt": True}
    ]
    assert tokenizer.tokenizer_calls == [
        ("[SANITIZED_CHAT_TEMPLATE]", "pt")
    ]


def test_gemma_termination_ids_include_eos_and_end_of_turn() -> None:
    assert generation_eos_token_ids(
        RecordingTokenizer(), GEMMA_MODEL_ID
    ) == [1, 107]


def test_unsupported_compute_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="model.compute_dtype"):
        resolve_settings(pilot_config("int8"))


def test_atomic_append_preserves_original_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "raw" / "outputs.jsonl"

    append_jsonl(
        output,
        {"safe_test_record": 1},
    )

    original_bytes = output.read_bytes()

    def fail_replace(
        source: Path,
        destination: Path,
    ) -> None:
        del source, destination
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(
        "src.run_local_model.os.replace",
        fail_replace,
    )

    with pytest.raises(
        OSError,
        match="simulated atomic replace failure",
    ):
        append_jsonl(
            output,
            {"safe_test_record": 2},
        )

    assert output.read_bytes() == original_bytes

    temporary_files = list(
        output.parent.glob(
            f".{output.name}.*.tmp"
        )
    )

    assert temporary_files == []


def test_load_completed_rejects_duplicate_keys(
    tmp_path: Path,
) -> None:
    output = tmp_path / "raw" / "outputs.jsonl"
    output.parent.mkdir(parents=True)

    row = {
        "scenario_id": "sanitized_scenario",
        "condition": "direct",
        "turn_index": 1,
        "generation_status": "ok",
        "model_id": "sanitized-model",
        "run_id": "sanitized-run",
        "response_text": (
            "[MODEL_RESPONSE_PLACEHOLDER]"
        ),
    }

    output.write_text(
        json.dumps(row)
        + "\n"
        + json.dumps(row)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="duplicate completed response key",
    ):
        load_completed(output)


def test_dry_run_generation_metadata(tmp_path: Path) -> None:
    records = generate_templates([scenario_record("scenario_dry_run")])
    output = tmp_path / "raw" / "outputs.jsonl"
    result = run_prompt_records(
        records,
        DryRunBackend(),
        output,
        run_id="dry-run-test",
        max_new_tokens=8,
        do_sample=False,
        limit_pairs=1,
    )

    row = read_output(output)[0]
    assert result["generated"] == 1
    assert row["response_text"] == "[MODEL_RESPONSE_PLACEHOLDER]"
    assert row["generated_token_count"] == 0
    assert row["hit_max_new_tokens"] is False
    assert row["finish_reason"] == "dry_run"


def test_generated_response_text_is_preserved_exactly(tmp_path: Path) -> None:
    exact_response = "[MODEL_RESPONSE_PLACEHOLDER]\n  preserved spacing  "

    class ExactTextBackend:
        model_id = "exact-text-test-model"

        def generate(self, messages: list[dict], max_new_tokens: int, do_sample: bool) -> GenerationResult:
            del messages, max_new_tokens, do_sample
            return GenerationResult(exact_response, 4, False, "other")

    records = generate_templates([scenario_record("scenario_exact_text")])
    output = tmp_path / "raw" / "outputs.jsonl"
    run_prompt_records(
        records,
        ExactTextBackend(),
        output,
        run_id="exact-text-test",
        max_new_tokens=8,
        do_sample=False,
        limit_pairs=1,
    )

    assert read_output(output)[0]["response_text"] == exact_response


def test_length_finish_reason_uses_generated_token_ids() -> None:
    metadata = generation_termination_metadata([10, 11, 12], max_new_tokens=3, eos_token_ids=99)
    assert metadata == (3, True, "length")


def test_eos_finish_reason_uses_generated_token_ids() -> None:
    metadata = generation_termination_metadata([10, 99], max_new_tokens=3, eos_token_ids=[98, 99])
    assert metadata == (2, False, "eos")


def test_resume_does_not_duplicate_completed_responses(tmp_path: Path) -> None:
    records = generate_templates([scenario_record("scenario_resume")])
    output = tmp_path / "raw" / "outputs.jsonl"

    first_backend = RecordingBackend()
    first = run_prompt_records(
        records,
        first_backend,
        output,
        run_id="resume-test",
        max_new_tokens=8,
        do_sample=False,
        resume=False,
        limit_pairs=1,
    )
    assert first["generated"] == 1

    second_backend = RecordingBackend()
    second = run_prompt_records(
        records,
        second_backend,
        output,
        run_id="resume-test",
        max_new_tokens=8,
        do_sample=False,
        resume=True,
    )
    rows = read_output(output)
    assert second["resumed"] == 1
    assert len(rows) == 8
    assert len({row["response_id"] for row in rows}) == 8


def test_resume_accepts_legacy_records_without_generation_metadata(tmp_path: Path) -> None:
    records = generate_templates([scenario_record("scenario_legacy")])
    output = tmp_path / "raw" / "outputs.jsonl"
    output.parent.mkdir(parents=True)
    legacy_record = {
        "response_id": "scenario_legacy::direct::turn_1",
        "scenario_id": "scenario_legacy",
        "condition": "direct",
        "turn_index": 1,
        "label": "harmful",
        "model_id": RecordingBackend.model_id,
        "run_id": "legacy-test",
        "generation_status": "ok",
        "response_text": "[MODEL_RESPONSE_PLACEHOLDER]",
    }
    output.write_text(json.dumps(legacy_record) + "\n", encoding="utf-8")

    result = run_prompt_records(
        records,
        RecordingBackend(),
        output,
        run_id="legacy-test",
        max_new_tokens=8,
        do_sample=False,
        resume=True,
    )

    rows = read_output(output)
    assert result["resumed"] == 1
    assert result["generated"] == 7
    assert len(rows) == 8
    assert "finish_reason" not in rows[0]
    assert all("finish_reason" in row for row in rows[1:])


def test_context_is_fresh_between_conditions_and_preserved_within_multiturn(tmp_path: Path) -> None:
    records = generate_templates([scenario_record("scenario_context")])
    output = tmp_path / "raw" / "outputs.jsonl"
    backend = RecordingBackend()
    run_prompt_records(
        records,
        backend,
        output,
        run_id="context-test",
        max_new_tokens=8,
        do_sample=False,
    )

    assert len(backend.calls[0]) == 1
    assert len(backend.calls[1]) == 1
    assert backend.calls[0][0]["content"] != backend.calls[1][0]["content"]
    assert [len(messages) for messages in backend.calls[2:5]] == [1, 3, 5]
    assert backend.calls[2][0]["content"].startswith("[MULTI_TURN_PROMPT_TEMPLATE")
    assert all("DIRECT_PROMPT_TEMPLATE" not in message["content"] for message in backend.calls[2])
