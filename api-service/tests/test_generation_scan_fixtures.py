"""Keeps vision-service/tests/fixtures/generated/*.png identical to what the generator prints
today. vision-service's test_generated_sheets.py scans those images with the real pipeline, so
together the two tests check generate -> print -> scan end to end (the services live in separate
venvs, so one test can't import both).

After an intentional change to the printed layout, regenerate the images with
    UPDATE_SCAN_FIXTURES=1 pytest tests/test_generation_scan_fixtures.py
and re-run vision-service's tests.
"""

import json
import os
import random
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image

from app.generation.composition import compose_omr_worksheet, compose_worksheet
from app.generation.render import PageSpec, render_worksheet_pdfs
from tests.test_generation_render import _differing_pixel_share

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "vision-service" / "tests" / "fixtures" / "generated"
MANIFEST_PATH = FIXTURES_DIR / "manifest.json"
# Absent inside the api-service docker image (only api-service/ is copied in).
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else []
pytestmark = pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="vision-service fixtures not available")
SCAN_DPI = 150  # A4 at 150 dpi = 1240x1754, vision-service's dewarp target size


def render_fixture(entry: dict) -> Image.Image:
    if entry["worksheet_type"] == "omr":
        sheet = compose_omr_worksheet(entry["first_question_index"] + entry["question_count"] - 1, language=entry["language"])
    else:
        sheet = compose_worksheet(entry["worksheet_type"], entry["level"], entry["language"], random.Random(entry["seed"]))
    page = PageSpec(entry["page_no"], entry["first_question_index"], entry["question_count"])
    (pdf,) = render_worksheet_pdfs(entry["worksheet_id"], sheet, entry["template_name"], [page])
    return pdfium.PdfDocument(pdf)[0].render(scale=SCAN_DPI / 72).to_pil().convert("RGB")


@pytest.mark.parametrize("entry", MANIFEST, ids=[e["file"] for e in MANIFEST])
def test_scan_fixture_matches_current_generator_output(entry):
    image = render_fixture(entry)
    path = FIXTURES_DIR / entry["file"]
    if os.environ.get("UPDATE_SCAN_FIXTURES") == "1":
        image.save(path, optimize=True)
    assert path.exists(), f"missing fixture {path.name}; run with UPDATE_SCAN_FIXTURES=1"
    stored = Image.open(path).convert("RGB")
    assert stored.size == image.size
    assert _differing_pixel_share(stored, image) == 0, f"{path.name} is out of date; run with UPDATE_SCAN_FIXTURES=1"
