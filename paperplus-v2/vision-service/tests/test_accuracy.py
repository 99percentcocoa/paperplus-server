"""Accuracy regression suite: runs real worksheet scans through the real /process endpoint
(real AprilTag detection, real PaddleOCR, real TFLite bubble classifier -- no fakes) and checks
output against tests/fixtures/ground_truth.json.

Provenance of ground_truth.json (2026-09-09): one entry (0151_1.jpeg) was hand-verified earlier
this project against the old system's testing/results.json. The other nine were bootstrapped by
running the OLD system's independent implementation of the same pipeline (its own AprilTag/OCR/
bubble-detection code, not vision-service's) against real worksheet scans in the old repo's
testing/images/ -- NOT hand-verified per-question, but cross-validates that this port's ROI
geometry, row-tag decoding, and OCR variant-fallback logic all agree with a second, separately
written implementation. That bootstrap deliberately never touched Postgres (not even read-only)
since the old system's DB is the live production database (paperpluslive) -- the one incidental
DB lookup inside its scan_image() (an optional template-name fallback) was forced to fail so it
degrades to the same row-tag-only inference vision-service uses. At bootstrap time this port
matched the old system exactly: 314/314 questions and all decoded fields (worksheet_id, page_no,
template_name, roll_number, question_paper_code) across all 10 images.

Every image here is a real scan (not synthetic), so this genuinely exercises PaddleOCR + the real
TFLite model, unlike tests/test_pipeline.py's fake-provider unit tests -- it's the slowest test
file in this suite as a result (real model inference per image).
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GROUND_TRUTH = json.loads((FIXTURES_DIR / "ground_truth.json").read_text())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("entry", GROUND_TRUTH, ids=[e["input_file"] for e in GROUND_TRUTH])
def test_scan_matches_ground_truth(client: TestClient, entry: dict):
    image_path = FIXTURES_DIR / entry["input_file"]

    response = client.post(
        "/process",
        json={"correlation_id": f"accuracy-{entry['input_file']}", "image_path": str(image_path)},
        headers={"x-service-secret": settings.shared_secret},
    )
    assert response.status_code == 200, response.text
    result = response.json()

    assert result["worksheet_id"] == entry["worksheet_id"]
    assert result["page_no"] == entry["page_no"]
    assert result["template_name"] == entry["template_name"]
    assert result["roll_number"] == entry["roll_number"]
    assert (result["question_paper_code"] or "") == (entry["question_paper_code"] or "")

    marks_by_index = {m["question_index"]: (m["marked_option"] or "") for m in result["question_marks"]}
    expected_by_index = {
        entry["first_question_index"] + offset: expected
        for offset, expected in enumerate(entry["question_marks"])
    }

    mismatches = {
        q_index: (expected, marks_by_index.get(q_index, "<missing>"))
        for q_index, expected in expected_by_index.items()
        if marks_by_index.get(q_index, "<missing>") != expected
    }
    assert not mismatches, f"question mark mismatches for {entry['input_file']}: {mismatches}"
