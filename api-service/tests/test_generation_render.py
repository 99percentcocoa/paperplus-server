"""Worksheet PDF rendering: page structure, row tags and page metadata.

The renderer was validated as pixel-identical to the pre-v2 system's renderer, and to sheets it
actually printed, before that code was removed. The parity tests are in git history
(commit 49dfad7, tests/test_generation_parity.py and test_generation_render.py); see
docs/PHASE9_GENERATION.md."""

import random

import pypdfium2 as pdfium
import pytest
from PIL import Image, ImageChops

from app.generation.composition import compose_omr_worksheet, compose_worksheet
from app.generation.render import PageSpec, render_page_html, render_worksheet_pdfs
from shared.row_tags import decode_row_tag_metadata

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


def _differing_pixel_share(a: Image.Image, b: Image.Image, threshold: int = 32) -> float:
    """Share of pixels where any channel differs by more than `threshold` (0-255)."""
    red, green, blue = ImageChops.difference(a, b).split()
    worst = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    over = worst.point(lambda v: 255 if v > threshold else 0).histogram()[255]
    return over / (a.width * a.height)
