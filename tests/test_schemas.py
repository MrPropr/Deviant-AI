from __future__ import annotations

import json
import subprocess
import sys
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
    assert output_path.exists()
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert sum(1 for record in records if record["split"] == "harmful") == 2
    assert sum(1 for record in records if record["split"] == "benign") == 2
    assert all(record["private_raw_text"] for record in records)
