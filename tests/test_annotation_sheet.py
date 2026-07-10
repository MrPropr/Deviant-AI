from __future__ import annotations

import csv
from pathlib import Path

from src.create_annotation_sheet import write_annotation_sheet


def test_annotation_sheet_preserves_response_text(tmp_path: Path) -> None:
    response_text = "[MODEL_RESPONSE_PLACEHOLDER]\nsecond sanitized line"
    records = [
        {
            "response_id": "scenario::direct::turn_1",
            "scenario_id": "scenario",
            "condition": "direct",
            "turn_index": 1,
            "label": "harmful",
            "model_id": "test-model",
            "run_id": "test-run",
            "generation_status": "ok",
            "dataset": {"id": "sanitized-test"},
            "response_text": response_text,
        }
    ]
    output = tmp_path / "judged" / "annotations.csv"
    write_annotation_sheet(output, records)
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["response_text"] == response_text
    assert rows[0]["harmfulness_score"] == ""
