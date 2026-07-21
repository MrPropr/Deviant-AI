from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml
import pytest

from src.generate_prompt_variants import CONDITIONS, generate_templates
from src.validate_prompt_variants import is_git_ignored, is_safe_private_output, validate_records


ROOT = Path(__file__).resolve().parents[1]


def sanitized_dataset_record(scenario_id: str = "scenario_001", label: str = "harmful") -> dict:
    placeholder = "[HARMFUL_BEHAVIOR_PLACEHOLDER]" if label == "harmful" else "[BENIGN_BEHAVIOR_PLACEHOLDER]"
    return {
        "scenario_id": scenario_id,
        "behavior_id": scenario_id,
        "label": label,
        "split": label,
        "dataset": "JailbreakBench/JBB-Behaviors",
        "dataset_config": "behaviors",
        "jbb_index": 1,
        "source": "sanitized_test",
        "risk_area": "placeholder",
        "behavior_label": "placeholder",
        "behavior": placeholder,
        "private_raw_text": True,
    }


def test_prompt_variant_schema_has_nested_conditions() -> None:
    schema = json.loads((ROOT / "prompts" / "prompt_variants_schema.json").read_text(encoding="utf-8"))
    assert schema["properties"]["label"]["enum"] == ["harmful", "benign"]
    assert schema["properties"]["conditions"]["required"] == list(CONDITIONS)


def test_generated_templates_are_structurally_valid_and_do_not_copy_behavior_text() -> None:
    templates = generate_templates([sanitized_dataset_record()])
    assert validate_records(templates) == []
    assert set(templates[0]["conditions"]) == set(CONDITIONS)
    serialized = json.dumps(templates)
    assert "[HARMFUL_BEHAVIOR_PLACEHOLDER]" not in serialized
    assert len(templates[0]["conditions"]["direct"]["turns"]) == 1
    assert len(templates[0]["conditions"]["multi_turn"]["turns"]) == 3


def test_validator_detects_duplicate_scenario_and_bad_counts() -> None:
    templates = generate_templates([sanitized_dataset_record()])
    duplicate = json.loads(json.dumps(templates[0]))
    templates[0]["conditions"]["direct"]["turns"][0]["word_count"] = 99
    errors = validate_records([templates[0], duplicate])
    assert any("word_count mismatch" in error for error in errors)
    assert any("duplicate scenario_id" in error for error in errors)
    assert all("scenario_001" not in error for error in errors)


def test_template_generation_errors_do_not_expose_scenario_identifiers() -> None:
    private_identifier = "PRIVATE_SCENARIO_IDENTIFIER"
    record = sanitized_dataset_record(private_identifier)
    with pytest.raises(ValueError) as exc_info:
        generate_templates([record, record])
    assert private_identifier not in str(exc_info.value)


def test_private_prompt_path_is_git_ignored() -> None:
    assert is_git_ignored(ROOT / "prompts" / "private" / "prompt_variants.jsonl", ROOT)


def test_private_output_paths_are_ignored_or_outside_the_repository(tmp_path: Path) -> None:
    assert is_safe_private_output(ROOT / "results" / "raw" / "output.jsonl", ROOT)
    assert is_safe_private_output(tmp_path / "output.jsonl", ROOT)
    assert not is_safe_private_output(ROOT / "tables" / "private_output.jsonl", ROOT)


def test_sanitized_examples_use_placeholders() -> None:
    lines = (ROOT / "data" / "sanitized_examples.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    placeholders = {"[HARMFUL_BEHAVIOR_PLACEHOLDER]", "[BENIGN_BEHAVIOR_PLACEHOLDER]"}
    assert records
    assert all(record["behavior"] in placeholders for record in records)


def test_configs_load_as_yaml() -> None:
    paths = [
        ROOT / "configs" / "models.yaml",
        ROOT / "configs" / "experiment.yaml",
        ROOT / "configs" / "pilot_qwen.yaml",
        ROOT / "configs" / "qwen_expanded_pilot.yaml",
    ]
    for path in paths:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        assert loaded

    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    assert [model["id"] for model in models["completed_models"]] == [
        "Qwen/Qwen2.5-7B-Instruct",
        "google/gemma-2-9b-it",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]

    pilot = yaml.safe_load((ROOT / "configs" / "pilot_qwen.yaml").read_text(encoding="utf-8"))
    assert pilot["model"] == {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "load_in_4bit": True,
        "quantization_type": "nf4",
        "compute_dtype": "float16",
    }
    assert pilot["inference"] == {
        "max_new_tokens": 512,
        "do_sample": False,
        "seed": 42,
        "max_multi_turn_length": 3,
    }

    expanded = yaml.safe_load((ROOT / "configs" / "qwen_expanded_pilot.yaml").read_text(encoding="utf-8"))
    assert expanded["model"] == {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "load_in_4bit": True,
        "quantization_type": "nf4",
        "compute_dtype": "float16",
    }
    assert expanded["inference"] == {
        "max_new_tokens": 1024,
        "do_sample": False,
        "seed": 42,
        "max_multi_turn_length": 3,
    }


def test_prepare_dataset_dry_run_writes_private_pilot(tmp_path: Path) -> None:
    output_path = tmp_path / "private" / "pilot_dataset.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.prepare_dataset",
            "--dry-run",
            "--harmful-size",
            "2",
            "--benign-size",
            "2",
            "--seed",
            "123",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "[HARMFUL_BEHAVIOR_PLACEHOLDER]" not in result.stdout
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert len({record["scenario_id"] for record in records}) == 4
    assert {record["label"] for record in records} == {"harmful", "benign"}
