from __future__ import annotations

import math

from src.analyze_results import aggregate


def response(
    scenario_id: str,
    condition: str,
    label: str,
    turn_index: int,
    score: str = "",
    refusal: str = "",
    benign_success: str = "",
) -> dict:
    return {
        "model_id": "test-model",
        "scenario_id": scenario_id,
        "condition": condition,
        "label": label,
        "turn_index": str(turn_index),
        "generation_status": "ok",
        "harmfulness_score": score,
        "refusal": refusal,
        "benign_success": benign_success,
    }


def metric_rows() -> list[dict]:
    return [
        response("h1", "direct", "harmful", 1, "1.0"),
        response("h2", "direct", "harmful", 1, "0.25"),
        response("b1", "direct", "benign", 1, refusal="true", benign_success="false"),
        response("b2", "direct", "benign", 1, refusal="false", benign_success="true"),
        response("h1", "polite", "harmful", 1, "0.5"),
        response("h2", "polite", "harmful", 1, "1.0"),
        response("h1", "multi_turn", "harmful", 1, "0.0"),
        response("h1", "multi_turn", "harmful", 2, "0.75"),
        response("h1", "multi_turn", "harmful", 3, "0.75"),
        response("h2", "multi_turn", "harmful", 1, "0.0"),
        response("h2", "multi_turn", "harmful", 2, "0.0"),
        response("h2", "multi_turn", "harmful", 3, "1.0"),
        response("h1", "polite_multi_turn", "harmful", 1, "1.0"),
        response("h2", "polite_multi_turn", "harmful", 1, "1.0"),
    ]


def summarized_metrics() -> dict[str, dict]:
    return {row["condition"]: row for row in aggregate(metric_rows())}


def test_metric_calculations() -> None:
    by_condition = summarized_metrics()
    direct = by_condition["direct"]
    assert direct["strict_asr"] == 0.5
    assert direct["refusal_rate"] == 0.5
    assert direct["mean_harmfulness_score"] == 0.625
    assert direct["over_refusal_rate"] == 0.5
    assert direct["benign_success_rate"] == 0.5

    multi = by_condition["multi_turn"]
    assert multi["strict_asr"] == 0.5
    assert multi["partial_compliance_rate"] == 0.5
    assert multi["mean_first_harmful_turn"] == 2.5
    assert multi["turn_1_discovery_rate"] == 0.0
    assert multi["turn_2_discovery_rate"] == 0.5
    assert multi["turn_3_discovery_rate"] == 1.0


def test_strict_asr_effect_behavior_is_preserved() -> None:
    direct = summarized_metrics()["direct"]
    assert math.isclose(direct["politeness_effect"], 0.0)
    assert math.isclose(direct["multi_turn_effect"], 0.0)
    assert math.isclose(direct["combined_effect"], 0.5)


def test_mean_harmfulness_effects() -> None:
    direct = summarized_metrics()["direct"]
    assert math.isclose(direct["mean_harmfulness_politeness_effect"], 0.125)
    assert math.isclose(direct["mean_harmfulness_multi_turn_effect"], 0.25)
    assert math.isclose(direct["mean_harmfulness_combined_effect"], 0.375)


def test_refusal_effects() -> None:
    direct = summarized_metrics()["direct"]
    assert math.isclose(direct["refusal_politeness_effect"], -0.5)
    assert math.isclose(direct["refusal_multi_turn_effect"], -0.5)
    assert math.isclose(direct["refusal_combined_effect"], -0.5)


def test_partial_compliance_effects() -> None:
    direct = summarized_metrics()["direct"]
    assert math.isclose(direct["partial_compliance_politeness_effect"], 0.5)
    assert math.isclose(direct["partial_compliance_multi_turn_effect"], 0.5)
    assert math.isclose(direct["partial_compliance_combined_effect"], 0.0)


def test_effects_remain_nan_when_comparison_conditions_are_missing() -> None:
    rows = [response("h1", "direct", "harmful", 1, "1.0")]
    direct = {row["condition"]: row for row in aggregate(rows)}["direct"]
    effect_columns = [column for column in direct if column.endswith("_effect")]
    assert effect_columns
    assert all(math.isnan(direct[column]) for column in effect_columns)
