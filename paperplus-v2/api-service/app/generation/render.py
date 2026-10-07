"""Worksheet HTML + PDF rendering -- ported from the old repo's worksheet_pdf_generator.py and
services/pdf_generator_service.py.

The HTML is built exactly as before, from the same templates (copied to
assets/worksheets/templates/). Row tags come from shared.row_tags, the same encoder/decoder
vision-service scans with. Tag/image URLs are relative to assets/worksheets/ (WeasyPrint's
base_url) instead of absolute host paths from the old .env.

Like the old generator, every sheet uses the template_en.html of its family -- Marathi sheets get
Devanagari numerals in the questions but the same printed instructions (the old template_mr.html
files were never selected by any code path, so they weren't copied).
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from shared.row_tags import worksheet_id_to_rows

logger = logging.getLogger(__name__)

# WeasyPrint's font subsetting logs every table it touches at DEBUG/INFO.
for _noisy in ("fontTools", "weasyprint"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "worksheets"
TEMPLATE_NAMES = ("regular", "basic_omr")
REGULAR_QUESTION_COUNT = 20
CORNER_TAG_IDS = (0, 1, 2, 3)


@dataclass(frozen=True)
class PageSpec:
    page_no: int
    first_question_index: int
    question_count: int


def _corner_tags_html() -> str:
    """The four fixed 36h11 corner tags (ids 0-3) used for page detection + dewarp."""
    urls = [f"tags/36h11/tag36_11_{tag_id:05d}.svg" for tag_id in CORNER_TAG_IDS]
    return (
        f'<img class="marker top-left" src="{urls[0]}" alt="tag 0" />\n'
        f'<img class="marker top-right" src="{urls[1]}" alt="tag 1" />\n'
        f'<img class="marker bottom-left" src="{urls[2]}" alt="tag 2" />\n'
        f'<img class="marker bottom-right" src="{urls[3]}" alt="tag 3" />\n'
    )


def _row_tag_url(tag_id: int) -> str:
    return f"tags/25h9/tag25_09_{tag_id:05d}.svg"


def _question_box_html(question: dict, q_no: int) -> str:
    options = question["options"]
    return (
        "<td class='question_td'>\n <div class='question'>\n"
        f"<p>{q_no}. {question['question_text']}</p>"
        "<table class='options-table'>\n <tr>"
        f"<td><div class='circle'></div>A. {options[0]}</td>"
        f"<td><div class='circle'></div>B. {options[1]}</td>"
        f"<td><div class='circle'></div>C. {options[2]}</td>"
        f"<td><div class='circle'></div>D. {options[3]}</td>"
        "</tr>\n </table>"
        "</div>\n </td>"
    )


def regular_questions_html(worksheet_id: int, questions: list[dict]) -> str:
    """Two questions per row, one 25h9 row tag at the start of each of the 10 rows."""
    if len(questions) != REGULAR_QUESTION_COUNT:
        raise ValueError(f"Expected exactly {REGULAR_QUESTION_COUNT} questions, got {len(questions)}")

    row_tags = worksheet_id_to_rows(worksheet_id)
    rows_html = ""
    for i in range(0, len(questions), 2):
        q1 = questions[i]
        q2 = questions[i + 1] if i + 1 < len(questions) else None
        row_tag_id = row_tags[(i // 2) % len(row_tags)]
        rows_html += "<tr>\n"
        rows_html += "<td class='row-marker'>\n"
        rows_html += f"<div class='marker' style='background-image: url({_row_tag_url(row_tag_id)})'></div>\n"
        rows_html += "</td>\n"
        rows_html += f"{_question_box_html(q1, i + 1)}\n"
        rows_html += f"{_question_box_html(q2, i + 2) if q2 else ''}\n"
        rows_html += "</tr>\n"
    return rows_html


def basic_omr_questions_html(worksheet_id: int, page: PageSpec) -> str:
    """Blank A-D bubble grid, three questions per row, numbered from page.first_question_index.
    Page 2+ row tags carry the page metadata (13-tag omr_v2 packet)."""
    if page.question_count <= 0:
        return ""

    row_tags = worksheet_id_to_rows(worksheet_id, page_no=page.page_no, first_question_index=page.first_question_index)
    rows_html = ""
    for row_index in range(0, page.question_count, 3):
        row_tag_id = row_tags[(row_index // 3) % len(row_tags)]
        rows_html += "<tr>\n"
        rows_html += "<td class='row-marker'>\n"
        rows_html += f"<img class='marker' src='{_row_tag_url(row_tag_id)}' alt='row tag' />\n"
        rows_html += "</td>\n"

        for offset in range(3):
            if row_index + offset >= page.question_count:
                break
            q_no = page.first_question_index + row_index + offset
            rows_html += "<td class='question_td'>\n <div class='question'>\n"
            rows_html += f"<p>{q_no}.</p>"
            rows_html += "<table class='options-table'>\n <tr>"
            rows_html += "<td><div class='circle'></div>A</td>"
            rows_html += "<td><div class='circle'></div>B</td>"
            rows_html += "<td><div class='circle'></div>C</td>"
            rows_html += "<td><div class='circle'></div>D</td>"
            rows_html += "</tr>\n </table>"
            rows_html += "</div>\n </td>\n"

        rows_html += "</tr>\n"
    return rows_html


def render_page_html(worksheet_id: int, worksheet_json: dict, template_name: str, page: PageSpec) -> str:
    """Fill a template family's HTML for one printed page."""
    if template_name not in TEMPLATE_NAMES:
        raise ValueError(f"Unsupported template name: {template_name!r}; expected one of {TEMPLATE_NAMES}")

    template_html = (ASSETS_DIR / "templates" / template_name / "template_en.html").read_text(encoding="utf-8")
    if template_name == "basic_omr":
        questions_html = basic_omr_questions_html(worksheet_id, page)
    else:
        questions_html = regular_questions_html(worksheet_id, worksheet_json.get("questions") or [])

    return (
        template_html.replace("{{template_name}}", template_name)
        .replace("{{tags_html}}", _corner_tags_html())
        .replace("{{questions}}", questions_html)
        .replace("{{worksheet_id}}", str(worksheet_id))
        .replace("{{level}}", worksheet_json.get("level") or "")
        .replace("{{worksheet_category}}", worksheet_json.get("worksheet_category") or "practice")
        .replace("{{assessment_code}}", str(worksheet_json.get("assessment_code", "")))
        .replace("{{roll_number}}", str(worksheet_json.get("roll_number", "")))
        .replace("{{question_count}}", str(page.question_count))
    )


def html_to_pdf(html: str) -> bytes:
    # Imported lazily: WeasyPrint pulls in Pango/cairo, which nothing else in api-service needs.
    from weasyprint import HTML

    return HTML(string=html, base_url=str(ASSETS_DIR)).write_pdf()


def page_specs(template_name: str, total_question_count: int, worksheet_pages: list | None = None) -> list[PageSpec]:
    """The printed pages of a worksheet. Multi-page worksheets use their worksheet_pages rows
    (anything with first_question_index/last_question_index/page_no); single-page ones print
    every question on page 1."""
    if worksheet_pages:
        return [
            PageSpec(p.page_no, p.first_question_index, p.last_question_index - p.first_question_index + 1)
            for p in sorted(worksheet_pages, key=lambda p: p.page_no)
        ]
    return [PageSpec(1, 1, total_question_count)]


def render_worksheet_pdfs(
    worksheet_id: int, worksheet_json: dict, template_name: str, pages: list[PageSpec]
) -> list[bytes]:
    """One PDF per printed page, in page order."""
    pdfs = []
    for page in pages:
        logger.info(
            "Rendering worksheet PDF: worksheet_id=%s template=%s page_no=%s questions=%s-%s",
            worksheet_id, template_name, page.page_no, page.first_question_index,
            page.first_question_index + page.question_count - 1,
        )
        pdfs.append(html_to_pdf(render_page_html(worksheet_id, worksheet_json, template_name, page)))
    return pdfs
