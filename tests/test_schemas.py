from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_prompt_variant_schema_conditions() -> None:
    schema = json.loads((ROOT / "prompts" / "prompt_variants_schema.json").read_text(encoding="utf-8"))
    condition_enum = schema["properties"]["condition"]["enum"]
    assert condition_enum == ["direct", "polite", "multi_turn", "polite_multi_turn"]


def test_sanitized_examples_use_placeholders() -> None:
    lines = (ROOT / "data" / "sanitized_examples.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines
    records = [json.loads(line) for line in lines if line.strip()]
    placeholders = {"[HARMFUL_BEHAVIOR_PLACEHOLDER]", "[BENIGN_BEHAVIOR_PLACEHOLDER]"}
    for record in records:
        assert record["behavior"] in placeholders
        assert "behavior_id" in record
        assert "source" in record


def test_configs_load_as_yaml() -> None:
    for path in [ROOT / "configs" / "models.yaml", ROOT / "configs" / "experiment.yaml"]:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        assert loaded
