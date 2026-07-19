from __future__ import annotations

import json

import pytest

from src.run_local_model import DEFAULT_MODEL_ID
from src.score_fixed_continuations import (
    CONDITIONS,
    DETAIL_FIELDS,
    FORBIDDEN_DETAIL_FIELDS,
    SyntheticScoringBackend,
    SyntheticTokenizer,
    parse_args,
    read_jsonl,
    run,
    score_records,
    tokenize_scenario,
    validate_input_records,
)


def make_scenario(
    scenario_id: str = "synthetic_scenario_1",
    label: str = "harmful",
) -> dict:
    return {
        "scenario_id": scenario_id,
        "label": label,
        "anchor_condition": "direct",
        "fixed_continuation": "[FIXED_CONTINUATION_PLACEHOLDER]",
        "contexts": {
            "direct": [
                {"role": "user", "content": "[DIRECT_CONTEXT]"},
            ],
            "polite": [
                {"role": "user", "content": "[POLITE_CONTEXT]"},
            ],
            "multi_turn": [
                {"role": "user", "content": "[MULTI_USER_1]"},
                {"role": "assistant", "content": "[SAFE_PRIOR_TURN]"},
                {"role": "user", "content": "[MULTI_USER_2]"},
            ],
            "polite_multi_turn": [
                {"role": "user", "content": "[POLITE_MULTI_USER_1]"},
                {"role": "assistant", "content": "[SAFE_PRIOR_TURN]"},
                {"role": "user", "content": "[POLITE_MULTI_USER_2]"},
            ],
        },
    }


def write_input(path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


class RecordingBackend(SyntheticScoringBackend):
    def __init__(self) -> None:
        super().__init__(DEFAULT_MODEL_ID)
        self.calls: list[tuple[list[int], list[int]]] = []

    def score(
        self,
        prefix_ids: list[int],
        continuation_ids: list[int],
        chunk_size: int,
    ) -> dict[str, float]:
        self.calls.append((list(prefix_ids), list(continuation_ids)))
        return super().score(prefix_ids, continuation_ids, chunk_size)


class InterruptingBackend(RecordingBackend):
    def score(
        self,
        prefix_ids: list[int],
        continuation_ids: list[int],
        chunk_size: int,
    ) -> dict[str, float]:
        if len(self.calls) >= 2:
            raise RuntimeError("synthetic interruption")
        return super().score(prefix_ids, continuation_ids, chunk_size)


class MismatchTokenizer(SyntheticTokenizer):
    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        encoded = super().__call__(text, add_special_tokens=add_special_tokens)
        if (
            "[FIXED_CONTINUATION_PLACEHOLDER]" in text
            and "[POLITE_CONTEXT]" in text
        ):
            encoded["input_ids"][-1] += 1
        return encoded


def test_validates_exact_four_condition_schema() -> None:
    assert validate_input_records([make_scenario()]) == [make_scenario()]


def test_rejects_missing_condition() -> None:
    scenario = make_scenario()
    del scenario["contexts"]["polite"]
    with pytest.raises(ValueError, match="condition set is incomplete"):
        validate_input_records([scenario])


def test_rejects_duplicate_scenario_identifier() -> None:
    with pytest.raises(ValueError, match="duplicate scenario identifier"):
        validate_input_records([make_scenario(), make_scenario()])


def test_rejects_empty_fixed_continuation() -> None:
    scenario = make_scenario()
    scenario["fixed_continuation"] = "  "
    with pytest.raises(ValueError, match="empty fixed continuation"):
        validate_input_records([scenario])


def test_rejects_non_direct_anchor() -> None:
    scenario = make_scenario()
    scenario["anchor_condition"] = "polite"
    with pytest.raises(ValueError, match="anchor must be direct"):
        validate_input_records([scenario])


def test_rejects_invalid_message_role() -> None:
    scenario = make_scenario()
    scenario["contexts"]["direct"][0]["role"] = "tool"
    with pytest.raises(ValueError, match="invalid context message"):
        validate_input_records([scenario])


def test_rejects_context_ending_with_assistant() -> None:
    scenario = make_scenario()
    scenario["contexts"]["direct"].append(
        {"role": "assistant", "content": "[MODEL_RESPONSE_PLACEHOLDER]"}
    )
    with pytest.raises(ValueError, match="must end with a user message"):
        validate_input_records([scenario])


def test_validate_only_writes_no_output(tmp_path, capsys) -> None:
    input_path = tmp_path / "private_input.jsonl"
    output_path = tmp_path / "unused.jsonl"
    write_input(input_path, [make_scenario()])
    args = parse_args(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--validate-only",
            "--dry-run",
        ]
    )
    summary = run(args)
    assert summary["generated"] == 0
    assert not output_path.exists()
    console = capsys.readouterr().out
    assert "Validated scenarios: 1." in console
    assert "Validated context-continuation pairs: 4." in console
    assert "[DIRECT_CONTEXT]" not in console
    assert "[FIXED_CONTINUATION_PLACEHOLDER]" not in console


def test_continuation_token_ids_are_identical() -> None:
    tokenized = tokenize_scenario(SyntheticTokenizer(), make_scenario())
    continuation_ids = [
        tokenized[condition][1]
        for condition in CONDITIONS
    ]
    assert all(value == continuation_ids[0] for value in continuation_ids)
    assert len(continuation_ids[0]) > 0


def test_mismatched_continuation_token_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="differ across conditions") as exc_info:
        tokenize_scenario(MismatchTokenizer(), make_scenario())
    assert "[FIXED_CONTINUATION_PLACEHOLDER]" not in str(exc_info.value)


def test_scoring_receives_only_continuation_tokens(tmp_path) -> None:
    backend = RecordingBackend()
    output_path = tmp_path / "detail.jsonl"
    records = validate_input_records([make_scenario()])
    summary = score_records(
        records,
        backend,
        output_path,
        run_id="synthetic_run",
        chunk_size=8,
    )
    assert summary["generated"] == 4
    assert len(backend.calls) == 4
    continuation_ids = [call[1] for call in backend.calls]
    assert all(value == continuation_ids[0] for value in continuation_ids)
    assert all(len(prefix) > len(continuation) for prefix, continuation in backend.calls)

    detail_rows = read_jsonl(output_path)
    assert all(
        row["continuation_token_count"] == len(continuation_ids[0])
        for row in detail_rows
    )


def test_atomic_checkpoint_resume_avoids_duplicates(tmp_path) -> None:
    output_path = tmp_path / "detail.jsonl"
    records = validate_input_records([make_scenario()])
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        score_records(
            records,
            InterruptingBackend(),
            output_path,
            run_id="synthetic_run",
            chunk_size=8,
        )
    assert len(read_jsonl(output_path)) == 2

    summary = score_records(
        records,
        RecordingBackend(),
        output_path,
        run_id="synthetic_run",
        chunk_size=8,
        resume=True,
    )
    final_rows = read_jsonl(output_path)
    assert summary == {
        "input_scenarios": 1,
        "generated": 2,
        "resumed": 2,
        "completed": 4,
    }
    assert len(final_rows) == 4
    assert len(
        {(row["scenario_id"], row["condition"]) for row in final_rows}
    ) == 4
    assert not list(tmp_path.glob(".*.tmp"))


def test_resume_rejects_keys_absent_from_current_input(tmp_path) -> None:
    output_path = tmp_path / "detail.jsonl"
    score_records(
        validate_input_records([make_scenario("synthetic_old")]),
        RecordingBackend(),
        output_path,
        run_id="synthetic_run",
        chunk_size=8,
    )
    with pytest.raises(ValueError, match="keys absent from the current input"):
        score_records(
            validate_input_records([make_scenario("synthetic_current")]),
            RecordingBackend(),
            output_path,
            run_id="synthetic_run",
            chunk_size=8,
            resume=True,
        )


def test_detail_output_contains_only_allowed_fields(tmp_path) -> None:
    output_path = tmp_path / "detail.jsonl"
    score_records(
        validate_input_records([make_scenario()]),
        RecordingBackend(),
        output_path,
        run_id="synthetic_run",
        chunk_size=8,
    )
    for row in read_jsonl(output_path):
        assert tuple(sorted(row)) == tuple(sorted(DETAIL_FIELDS))
        assert not (set(row) & FORBIDDEN_DETAIL_FIELDS)
        assert row["status"] == "dry_run"


def test_dry_run_is_deterministic(tmp_path) -> None:
    records = validate_input_records([make_scenario()])
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    score_records(
        records,
        RecordingBackend(),
        first_path,
        run_id="synthetic_run",
        chunk_size=8,
    )
    score_records(
        records,
        RecordingBackend(),
        second_path,
        run_id="synthetic_run",
        chunk_size=8,
    )
    assert first_path.read_bytes() == second_path.read_bytes()
