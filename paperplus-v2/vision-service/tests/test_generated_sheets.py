"""Sheets printed by api-service's worksheet generator scan correctly: each image in
fixtures/generated/ (kept identical to the generator's current output by api-service's
test_generation_scan_fixtures.py) is a clean 150-dpi rasterization of a generated PDF. Running
them through the real /process pipeline must decode the printed worksheet id/page/template and
find every question's bubbles, all unmarked.

Clean renders are the best case -- they prove the printed tags and layout match what the scanner
expects, not robustness to real photos (test_accuracy.py covers real scans).
"""

import base64
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

GENERATED_DIR = Path(__file__).parent / "fixtures" / "generated"
MANIFEST = json.loads((GENERATED_DIR / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("entry", MANIFEST, ids=[e["file"] for e in MANIFEST])
def test_generated_sheet_scans_back(client: TestClient, entry: dict):
    response = client.post(
        "/process",
        json={
            "correlation_id": f"generated-{entry['file']}",
            "image_b64": base64.b64encode((GENERATED_DIR / entry["file"]).read_bytes()).decode("ascii"),
        },
        headers={"x-service-secret": settings.shared_secret},
    )
    assert response.status_code == 200, response.text
    result = response.json()

    assert result["worksheet_id"] == entry["worksheet_id"]
    assert result["page_no"] == entry["page_no"]
    assert result["template_name"] == entry["template_name"]

    first = entry["first_question_index"]
    expected_indices = list(range(first, first + entry["question_count"]))
    assert sorted(m["question_index"] for m in result["question_marks"]) == expected_indices
    marked = {m["question_index"]: m["marked_option"] for m in result["question_marks"] if m["marked_option"]}
    assert not marked, f"blank sheet read as marked: {marked}"
