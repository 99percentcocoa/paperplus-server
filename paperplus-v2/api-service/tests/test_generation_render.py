"""Worksheet PDF rendering: structure checks (always) and visual parity with the old repo's
renderer (when the old repo + its venv are present). Parity rasterizes both PDFs and requires
them to be pixel-identical apart from a tiny tolerance for anti-aliasing."""

import json
import os
import random
import subprocess
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image, ImageChops

from app.generation.composition import compose_omr_worksheet, compose_worksheet
from app.generation.render import PageSpec, render_page_html, render_worksheet_pdfs
from shared.row_tags import decode_row_tag_metadata

OLD_REPO = Path(__file__).resolve().parents[3]
OLD_PYTHON = OLD_REPO / ".venv" / "bin" / "python"
REFERENCE_SCRIPT = Path(__file__).with_name("_old_renderer_reference.py")
RENDER_DPI_SCALE = 100 / 72
MAX_DIFFERING_PIXEL_SHARE = 0.001


def _row_tag_ids(html: str) -> list[int]:
    import re

    return [int(m) for m in re.findall(r"tag25_09_(\d{5})\.svg", html)]


def test_regular_sheet_html_carries_decodable_row_tags_and_every_question():
    sheet = compose_worksheet("homework", "B", "en", random.Random(1))
    html = render_page_html(4810, sheet, "regular", PageSpec(1, 1, 20))
    assert decode_row_tag_metadata(_row_tag_ids(html))["worksheet_id"] == 4810
    for q in sheet["questions"]:
        assert f"{q['index']}. {q['question_text']}" in html
    assert "{{" not in html.split("<!--")[0]


def test_omr_page_two_html_encodes_page_metadata():
    html = render_page_html(990001, compose_omr_worksheet(78), "basic_omr", PageSpec(2, 40, 39))
    tags = _row_tag_ids(html)
    assert len(tags) == 13
    assert decode_row_tag_metadata(tags) == {
        "worksheet_id": 990001, "page_no": 2, "first_question_index": 40, "format": "omr_v2",
    }
    assert "<p>40.</p>" in html and "<p>78.</p>" in html and "<p>39.</p>" not in html


def test_regular_sheet_needs_exactly_20_questions():
    sheet = compose_worksheet("homework", "A", "en", random.Random(1))
    sheet["questions"] = sheet["questions"][:19]
    with pytest.raises(ValueError):
        render_page_html(1, sheet, "regular", PageSpec(1, 1, 19))


def test_renders_a_one_page_pdf_per_page():
    pdfs = render_worksheet_pdfs(990001, compose_omr_worksheet(78), "basic_omr", [PageSpec(1, 1, 39), PageSpec(2, 40, 39)])
    assert len(pdfs) == 2
    for pdf_bytes in pdfs:
        assert pdf_bytes.startswith(b"%PDF") and len(pdfium.PdfDocument(pdf_bytes)) == 1


def _rasterize(pdf) -> Image.Image:
    return pdfium.PdfDocument(pdf)[0].render(scale=RENDER_DPI_SCALE).to_pil().convert("RGB")


def _differing_pixel_share(a: Image.Image, b: Image.Image, threshold: int = 32) -> float:
    """Share of pixels where any channel differs by more than `threshold` (0-255)."""
    red, green, blue = ImageChops.difference(a, b).split()
    worst = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    over = worst.point(lambda v: 255 if v > threshold else 0).histogram()[255]
    return over / (a.width * a.height)


@pytest.mark.skipif(
    not (OLD_PYTHON.exists() and (OLD_REPO / "worksheet_pdf_generator.py").exists()),
    reason="old repo / its venv not available",
)
def test_pdfs_match_the_old_renderer_pixel_for_pixel(tmp_path):
    homework_en = compose_worksheet("homework", "D", "en", random.Random(11))
    homework_mr = compose_worksheet("homework", "F", "mr", random.Random(12))
    practice_mr = compose_worksheet("practice", "M3", "mr", random.Random(13))
    omr_page = compose_omr_worksheet(39)  # the old generator prints each OMR page from a 39-question JSON
    cases = [
        ("homework_en", 4810, homework_en, "regular", PageSpec(1, 1, 20), {}),
        ("homework_mr", 31337, homework_mr, "regular", PageSpec(1, 1, 20), {}),
        ("practice_mr", 7, practice_mr, "regular", PageSpec(1, 1, 20), {}),
        ("omr_p1", 990001, omr_page, "basic_omr", PageSpec(1, 1, 39), {"page_no": 1, "first_question_index": 1}),
        ("omr_p2", 990001, omr_page, "basic_omr", PageSpec(2, 40, 39), {"page_no": 2, "first_question_index": 40}),
    ]
    old_cases = []
    for name, worksheet_id, sheet, template_name, _, extra in cases:
        json_path = tmp_path / f"{name}.json"
        json_path.write_text(json.dumps(sheet, ensure_ascii=False), encoding="utf-8")
        old_cases.append({"name": name, "worksheet_id": worksheet_id, "json_path": str(json_path),
                          "template_name": template_name, **extra})
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(old_cases), encoding="utf-8")

    env = {**os.environ, "DATABASE_URL": "postgresql://render-parity-must-not-connect.invalid/none"}
    result = subprocess.run(
        [str(OLD_PYTHON), str(REFERENCE_SCRIPT), str(cases_path), str(tmp_path)],
        cwd=OLD_REPO, env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"old renderer failed:\n{result.stderr[-3000:]}"

    for name, worksheet_id, sheet, template_name, page, _ in cases:
        (new_pdf,) = render_worksheet_pdfs(worksheet_id, sheet, template_name, [page])
        old_pixels = _rasterize(str(tmp_path / f"{name}.pdf"))
        new_pixels = _rasterize(new_pdf)
        assert old_pixels.size == new_pixels.size, name
        differing = _differing_pixel_share(old_pixels, new_pixels)
        assert differing <= MAX_DIFFERING_PIXEL_SHARE, f"{name}: {differing:.4%} of pixels differ"
