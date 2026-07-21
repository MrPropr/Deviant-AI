from __future__ import annotations

import math

import pytest

from src.analyze_token_probabilities import (
    METRICS,
    bootstrap_mean_ci,
    build_pair_rows,
    sign_flip_permutation_test,
    safe_spearman,
    validate_input_rows,
)


def make_row(
    scenario_id: str,
    condition: str,
    label: str,
    value: float,
    harmfulness: float | None,
) -> dict:
    row = {
        "scenario_id": scenario_id,
        "condition": condition,
        "label": label,
        "response_class": (
            "unsafe"
            if harmfulness is not None
            and harmfulness >= 0.75
            else "refusal"
        ),
        "finish_reason": "eos",
        "manual_harmfulness_score": harmfulness,
    }

    for metric in METRICS:
        row[metric] = value

    return row


def test_pair_rows_use_right_minus_left() -> None:
    rows = [
        make_row(
            "scenario_1",
            "direct",
            "harmful",
            -2.0,
            0.25,
        ),
        make_row(
            "scenario_1",
            "polite",
            "harmful",
            -1.5,
            0.50,
        ),
        make_row(
            "scenario_1",
            "multi_turn",
            "harmful",
            -1.0,
            0.75,
        ),
        make_row(
            "scenario_1",
            "polite_multi_turn",
            "harmful",
            -0.5,
            1.00,
        ),
    ]

    pair_rows = build_pair_rows(rows)

    polite_pair = next(
        row
        for row in pair_rows
        if row["comparison"]
        == "polite_minus_direct"
    )

    assert polite_pair[
        "delta_mean_token_logprob"
    ] == 0.5

    assert polite_pair[
        "harmfulness_delta"
    ] == 0.25


def test_bootstrap_constant_values() -> None:
    low, high = bootstrap_mean_ci(
        values=[2.0, 2.0, 2.0],
        iterations=1000,
        seed=42,
    )

    assert low == 2.0
    assert high == 2.0


def test_sign_flip_all_zero() -> None:
    p_value, method = (
        sign_flip_permutation_test(
            differences=[0.0, 0.0],
            iterations=1000,
            seed=42,
        )
    )

    assert p_value == 1.0
    assert method == "all_zero"


def test_positive_spearman() -> None:
    rho, p_value = safe_spearman(
        [0.0, 0.25, 0.5, 1.0],
        [-4.0, -3.0, -2.0, -1.0],
    )

    assert math.isclose(
        rho,
        1.0,
        abs_tol=1e-12,
    )

    assert 0.0 <= p_value <= 1.0


def test_validation_errors_do_not_expose_response_identifiers() -> None:
    private_identifier = "PRIVATE_RESPONSE_IDENTIFIER"
    row = make_row(private_identifier, "invalid", "harmful", -1.0, 0.25)
    row.update(
        {
            "response_id": private_identifier,
            "turn_index": 1,
            "model_id": "synthetic-model",
            "run_id": "synthetic-run",
            "is_final_turn": True,
            "hit_max_new_tokens": False,
        }
    )
    with pytest.raises(ValueError) as exc_info:
        validate_input_rows([row])
    assert private_identifier not in str(exc_info.value)
