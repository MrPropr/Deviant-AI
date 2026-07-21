from __future__ import annotations

import math

import pytest
import torch

from src.score_response_logprobs import (
    RunningTokenMetrics,
    append_jsonl,
    build_context_messages,
    load_completed_ids,
    read_jsonl,
    validate_safe_output_record,
)


def make_prompt_map() -> dict:
    return {
        "scenario_test": {
            "scenario_id": "scenario_test",
            "label": "harmful",
            "conditions": {
                "direct": {
                    "turns": [
                        {
                            "turn_index": 1,
                            "content": "[USER_TURN_1]",
                        }
                    ]
                },
                "polite": {
                    "turns": [
                        {
                            "turn_index": 1,
                            "content": "[POLITE_USER_TURN_1]",
                        }
                    ]
                },
                "multi_turn": {
                    "turns": [
                        {
                            "turn_index": 1,
                            "content": "[USER_TURN_1]",
                        },
                        {
                            "turn_index": 2,
                            "content": "[USER_TURN_2]",
                        },
                        {
                            "turn_index": 3,
                            "content": "[USER_TURN_3]",
                        },
                    ]
                },
                "polite_multi_turn": {
                    "turns": [
                        {
                            "turn_index": 1,
                            "content": "[POLITE_USER_TURN_1]",
                        },
                        {
                            "turn_index": 2,
                            "content": "[POLITE_USER_TURN_2]",
                        },
                        {
                            "turn_index": 3,
                            "content": "[POLITE_USER_TURN_3]",
                        },
                    ]
                },
            },
        }
    }


def make_response_map() -> dict:
    return {
        (
            "scenario_test",
            "multi_turn",
            1,
        ): {
            "response_text": "[ASSISTANT_TURN_1]"
        },
        (
            "scenario_test",
            "multi_turn",
            2,
        ): {
            "response_text": "[ASSISTANT_TURN_2]"
        },
        (
            "scenario_test",
            "multi_turn",
            3,
        ): {
            "response_text": "[ASSISTANT_TURN_3]"
        },
    }


def test_direct_context_has_one_message() -> None:
    messages = build_context_messages(
        scenario_id="scenario_test",
        condition="direct",
        turn_index=1,
        prompts_by_id=make_prompt_map(),
        responses_by_key=make_response_map(),
    )

    assert messages == [
        {
            "role": "user",
            "content": "[USER_TURN_1]",
        }
    ]


def test_second_multiturn_context_is_complete() -> None:
    messages = build_context_messages(
        scenario_id="scenario_test",
        condition="multi_turn",
        turn_index=2,
        prompts_by_id=make_prompt_map(),
        responses_by_key=make_response_map(),
    )

    assert messages == [
        {
            "role": "user",
            "content": "[USER_TURN_1]",
        },
        {
            "role": "assistant",
            "content": "[ASSISTANT_TURN_1]",
        },
        {
            "role": "user",
            "content": "[USER_TURN_2]",
        },
    ]


def test_third_multiturn_context_is_complete() -> None:
    messages = build_context_messages(
        scenario_id="scenario_test",
        condition="multi_turn",
        turn_index=3,
        prompts_by_id=make_prompt_map(),
        responses_by_key=make_response_map(),
    )

    assert len(messages) == 5

    assert [
        message["role"]
        for message in messages
    ] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]

    assert (
        messages[-1]["content"]
        == "[USER_TURN_3]"
    )

    assert all(
        message["content"]
        != "[ASSISTANT_TURN_3]"
        for message in messages
    )


def test_running_metrics_match_simple_logits() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )

    targets = torch.tensor([0, 1])

    metrics = RunningTokenMetrics()

    metrics.update(
        logits=logits,
        target_ids=targets,
        torch_module=torch,
    )

    result = metrics.finalize()

    expected_logprob = math.log(
        math.exp(2.0)
        / (
            math.exp(2.0)
            + math.exp(0.0)
        )
    )

    assert result[
        "scored_token_count"
    ] == 2

    assert result[
        "mean_token_logprob"
    ] == pytest.approx(
        expected_logprob,
        abs=1e-6,
    )

    assert (
        0.0
        < result[
            "geometric_mean_token_probability"
        ]
        < 1.0
    )

    assert result["perplexity"] > 1.0
    assert result["mean_token_entropy"] > 0.0

    assert (
        0.0
        <= result[
            "mean_top1_top2_probability_margin"
        ]
        <= 1.0
    )


def test_private_text_fields_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="forbidden fields",
    ):
        validate_safe_output_record(
            {
                "response_id": "safe_id",
                "response_text": "private",
            }
        )


def test_numeric_output_is_accepted() -> None:
    validate_safe_output_record(
        {
            "response_id": "safe_id",
            "mean_token_logprob": -1.0,
            "perplexity": math.e,
            "scoring_status": "ok",
        }
    )


def test_atomic_jsonl_checkpoint_preserves_complete_records(tmp_path) -> None:
    output = tmp_path / "private" / "scores.jsonl"
    append_jsonl(output, {"record": 1})
    append_jsonl(output, {"record": 2})
    assert read_jsonl(output) == [{"record": 1}, {"record": 2}]
    assert not list(output.parent.glob(".*.tmp"))


def test_resume_duplicate_error_does_not_expose_response_identifier(tmp_path) -> None:
    output = tmp_path / "private" / "scores.jsonl"
    private_identifier = "PRIVATE_RESPONSE_IDENTIFIER"
    row = {
        "response_id": private_identifier,
        "mean_token_logprob": -1.0,
        "scoring_status": "ok",
    }
    append_jsonl(output, row)
    append_jsonl(output, row)
    with pytest.raises(ValueError) as exc_info:
        load_completed_ids(output)
    assert private_identifier not in str(exc_info.value)
