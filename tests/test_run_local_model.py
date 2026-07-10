from __future__ import annotations

import json
from pathlib import Path

from src.generate_prompt_variants import generate_templates
from src.run_local_model import run_prompt_records


class RecordingBackend:
    model_id = "recording-test-model"

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def generate(self, messages: list[dict], max_new_tokens: int, do_sample: bool) -> str:
        assert max_new_tokens == 8
        assert do_sample is False
        self.calls.append([dict(message) for message in messages])
        return "[MODEL_RESPONSE_PLACEHOLDER]"


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
