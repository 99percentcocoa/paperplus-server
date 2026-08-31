import logging
import json
import math
import argparse
from pathlib import Path
from typing import Optional, Any
from services.pdf_generator_service import (
    generate_tags_html,
    generate_questions_html,
    generate_basic_omr_questions_html,
)
from config import SETTINGS
from template_layouts import get_template_layout

TAGS_PATH = SETTINGS.TAGS_PATH
TEMPLATES_PATH = SETTINGS.TEMPLATES_PATH
PDF_WRITE_PATH = SETTINGS.PDF_WRITE_PATH
WORKSHEET_JSON_PATH = SETTINGS.WORKSHEET_JSON_PATH
HTML_BASE_DIR = SETTINGS.HTML_BASE_DIR

logger = logging.getLogger(__name__)

# CLI entrypoints need a configured root logger to emit output in the terminal.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
for logger_name in [
    "fontTools",
    "fontTools.ttLib",
    "fontTools.ttLib.ttFont",
    "fontTools.subset",
    "fontTools.subset.timer",
    "fontTools.misc",
    "weasyprint",
]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# python3 worksheet_pdf_generator.py --worksheet-id 1 --worksheet-json-filename en_level_A.json

def _load_worksheet_data(worksheet_json_filename: str, worksheet_json_path: Optional[str] = None) -> dict:
    """Load a worksheet JSON object from a direct path, a custom folder, or the configured workspace folders."""
    candidate = Path(worksheet_json_filename)
    search_roots: list[Path] = []

    if candidate.is_absolute() and candidate.exists():
        path = candidate
    else:
        if worksheet_json_path:
            search_roots.append(Path(worksheet_json_path))
        if WORKSHEET_JSON_PATH:
            search_roots.append(Path(WORKSHEET_JSON_PATH))
        search_roots.append(Path.cwd())
        search_roots.append(Path(__file__).resolve().parent)

        path = None
        for root in search_roots:
            possible = root / worksheet_json_filename if not candidate.is_absolute() else candidate
            if possible.exists():
                path = possible
                break

        if path is None:
            raise FileNotFoundError(
                f"Worksheet JSON not found: {worksheet_json_filename}. "
                f"Checked: {', '.join(str(p) for p in search_roots)}"
            )

    with open(path, "r", encoding="utf-8") as f:
        worksheet_data = json.load(f)

    if isinstance(worksheet_data, list):
        if not worksheet_data:
            raise ValueError("worksheet JSON list is empty")
        worksheet = worksheet_data[0]
    elif isinstance(worksheet_data, dict):
        worksheet = worksheet_data
    else:
        raise ValueError("worksheet JSON must be an object or a non-empty list")

    return worksheet


def _resolve_template_name(worksheet: dict, template_name: Optional[str] = None) -> str:
    """Resolve which template family to use. Defaults to the legacy regular folder."""
    selected = template_name or worksheet.get("template_name") or worksheet.get("template_type") or "regular"
    normalized = str(selected).strip().lower()

    if normalized in {"regular", "basic_omr"}:
        return normalized
    if normalized in {"plain_omr_assessment", "omr_assessment", "basic-omr"}:
        return "basic_omr"
    if normalized in {"practice", "worksheet"}:
        return "regular"

    raise ValueError(f"Unsupported template name: {selected!r}")


def _resolve_question_count(
    worksheet: dict,
    questions: Optional[list] = None,
    template_name: Optional[str] = None,
) -> int:
    """Prefer the actual question list length, then a template-specific default."""
    actual_count = len(questions) if questions is not None else len(worksheet.get("questions") or [])
    if actual_count > 0:
        return actual_count

    raw_count = worksheet.get("question_count")
    if raw_count is not None:
        try:
            return max(0, int(raw_count))
        except (TypeError, ValueError):
            pass

    layout = get_template_layout(template_name)
    return max(0, int(layout.num_questions))


def _resolve_template_filename(
    worksheet: dict,
    template_name: str,
    template_filename: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve the template folder and file name.

    Returns (template_name, filename).
    """
    if template_filename:
        return template_name, template_filename

    if template_name in {"basic_omr", "regular"}:
        # Legacy worksheets often omit language metadata. Keep the default template
        # behavior consistent for both the regular and OMR families.
        return template_name, "template_en.html"

    language = worksheet.get("language")
    if language == "en":
        return template_name, "template_en.html"
    if language == "mr":
        return template_name, "template_mr.html"

    raise ValueError("worksheet must specify a supported language or explicit template filename")


def _infer_page_plan(
    question_count: int,
    template_name: str,
    page_no: Optional[int] = None,
    first_question_index: Optional[int] = None,
) -> list[tuple[int, int]]:
    """Infer the page list for a worksheet.

    basic_omr sheets use 39 questions per page. If the page metadata is omitted,
    page numbers and question starts are inferred automatically.
    """
    if template_name != "basic_omr":
        return [(page_no or 1, first_question_index or 1)]

    total_pages = max(1, math.ceil(max(0, question_count) / 39))
    if question_count >= 39:
        total_pages = max(2, total_pages)

    if page_no is not None and first_question_index is not None:
        return [(page_no, first_question_index)]
    if page_no is not None:
        return [(page_no, 1 + (page_no - 1) * 39)]
    if first_question_index is not None:
        inferred_page = 1 + ((first_question_index - 1) // 39)
        return [(inferred_page, first_question_index)]

    return [(page_index, 1 + (page_index - 1) * 39) for page_index in range(1, total_pages + 1)]


def generate_worksheet_pdf(
    worksheet_id: int,
    worksheet_json_filename: str,
    tags_folder_path: Optional[str] = None,
    output_path: Optional[str] = PDF_WRITE_PATH,
    base_dir: Optional[str] = HTML_BASE_DIR,
    template_name: Optional[str] = "regular",
    template_filename: Optional[str] = None,
    worksheet_json_path: Optional[str] = None,
    page_no: Optional[int] = None,
    first_question_index: Optional[int] = None,
) -> bytes:
    """
    Generate a worksheet PDF and return its bytes.

    Args:
      worksheet_id: numeric id used by the tag functions.
      worksheet_json_filename: name of the worksheet json file within WORKSHEET_JSON_PATH.
    tags_folder_path: optional override for tags folder path.
      output_path: if provided, the PDF will be written to this path or directory (default: PDF_WRITE_PATH).
    base_dir: base directory used to derive tags/ and templates/ paths (default: configured HTML_BASE_DIR).

    Returns:
      The generated PDF as bytes.
    """

    effective_base_dir = Path(base_dir or HTML_BASE_DIR)
    effective_tags_folder_path = Path(tags_folder_path) if tags_folder_path else (effective_base_dir / "tags")
    effective_templates_path = effective_base_dir / "templates"

    worksheet = _load_worksheet_data(worksheet_json_filename, worksheet_json_path)
    questions = worksheet.get("questions") or []
    resolved_template_name = _resolve_template_name(worksheet, template_name)
    template_layout = get_template_layout(resolved_template_name)
    question_count = _resolve_question_count(worksheet, questions, resolved_template_name)
    logger.info("Generating PDF: worksheet_id=%s template=%s question_count=%s page_no=%s first_question_index=%s",
                worksheet_id, resolved_template_name, question_count, page_no, first_question_index)
    if question_count == 0:
        question_count = max(0, int(template_layout.num_questions))
    resolved_template_name, resolved_template_filename = _resolve_template_filename(
        worksheet, resolved_template_name, template_filename
    )

    template_dir = effective_templates_path / resolved_template_name
    template_path = template_dir / resolved_template_filename
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
        template_html = f.read()

    if resolved_template_name == "basic_omr":
        logger.info("Building basic_omr question grid")
        questions_html = generate_basic_omr_questions_html(
            worksheet_id=worksheet_id,
            question_count=question_count,
            tags_folder_path=str(effective_tags_folder_path),
            page_no=page_no,
            first_question_index=first_question_index,
        )
    else:
        logger.info("Building legacy question grid")
        questions_html = generate_questions_html(worksheet_id, questions, str(effective_tags_folder_path))

    # corner tags are fixed: 0, 1, 2, 3
    logger.info("Building corner tags")
    tags_html = generate_tags_html([0, 1, 2, 3], str(effective_tags_folder_path))
    # tags_html = generate_tags_html(getTagNumbers(worksheet_id), str(effective_tags_folder_path))
    # tags_html = generate_cctag_html([0,1,2,3], str(effective_tags_folder_path))

    # fill template placeholders
    final_html = (
        template_html
        .replace("{{template_name}}", resolved_template_name)
        .replace("{{tags_html}}", tags_html)
        .replace("{{questions}}", questions_html)
        .replace("{{worksheet_id}}", str(worksheet_id))
        .replace("{{level}}", worksheet.get("level", ""))
        .replace("{{worksheet_category}}", worksheet.get("worksheet_category", "practice"))
        .replace("{{assessment_code}}", str(worksheet.get("assessment_code", "")))
        .replace("{{roll_number}}", str(worksheet.get("roll_number", "")))
        .replace("{{question_count}}", str(question_count))
    )

    # Import weasyprint lazily so errors are raised only when generating.
    # Loading this C-backed library at import time can trigger allocator issues
    # when it is combined with other modules in the same interpreter lifecycle.
    from weasyprint import HTML

    # weasyprint needs a base_url so relative asset URLs resolve properly
    pdf_bytes = HTML(string=final_html, base_url=str(effective_base_dir)).write_pdf()

    # optionally write to disk
    if output_path:
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes


if __name__ == "__main__":
    # Usage:
    # python3 worksheet_pdf_generator.py --worksheet-id 1 --student-name "John Doe" \
    #   --student-iyatta 5 --worksheet-date 2023-01-01 --worksheet-json-filename en_level_A.json \
    #   --output-path /home/saarang/paperplus_server/files/pdf/worksheet_2.pdf \
    #   --base-dir /home/saarang/paperplus_server/assets
    parser = argparse.ArgumentParser(description="Generate a worksheet PDF from a worksheet JSON file.")
    parser.add_argument("--worksheet-id", type=int, required=True, help="Numeric worksheet id used by the tag functions.")
    parser.add_argument(
        "--worksheet-json-filename",
        required=True,
        help="Worksheet JSON filename within the configured worksheet JSON folder.",
    )
    parser.add_argument("--tags-folder-path", default=None, help="Optional override for tags folder path.")
    parser.add_argument(
        "--output-path",
        default=PDF_WRITE_PATH,
        help="Path to write the generated PDF file.",
    )
    parser.add_argument(
        "--base-dir",
        default=HTML_BASE_DIR,
        help="Base directory used to resolve tags/ and templates/ paths.",
    )
    parser.add_argument(
        "--template-name",
        default="regular",
        help="Template family folder under assets/templates: regular or basic_omr. Default: regular.",
    )
    parser.add_argument(
        "--template-file",
        default=None,
        help="Optional template filename inside the selected template folder.",
    )
    parser.add_argument(
        "--worksheet-json-path",
        default=None,
        help="Optional custom directory that contains the worksheet JSON file.",
    )
    parser.add_argument(
        "--page-no",
        type=int,
        default=None,
        help="OMR page number. If omitted, basic_omr pages are inferred automatically.",
    )
    parser.add_argument(
        "--first-question-index",
        type=int,
        default=None,
        help="First question number on the current page. If omitted, it is inferred from the page number.",
    )

    args = parser.parse_args()
    worksheet = _load_worksheet_data(args.worksheet_json_filename, args.worksheet_json_path)
    resolved_template_name = _resolve_template_name(worksheet, args.template_name)
    question_count = _resolve_question_count(worksheet, worksheet.get("questions") or [], resolved_template_name)
    page_plan = _infer_page_plan(question_count, resolved_template_name, args.page_no, args.first_question_index)

    if len(page_plan) == 1:
        page_no, first_question_index = page_plan[0]
        output_path = args.output_path
        if output_path.endswith(".pdf"):
            final_output = output_path
        else:
            final_output = f"{output_path}/{args.worksheet_id}.pdf"
        generate_worksheet_pdf(
            worksheet_id=args.worksheet_id,
            worksheet_json_filename=args.worksheet_json_filename,
            tags_folder_path=args.tags_folder_path,
            output_path=final_output,
            base_dir=args.base_dir,
            template_name=args.template_name,
            template_filename=args.template_file,
            worksheet_json_path=args.worksheet_json_path,
            page_no=page_no,
            first_question_index=first_question_index,
        )
    else:
        output_dir = args.output_path if not args.output_path.endswith(".pdf") else str(Path(args.output_path).parent)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for page_no, first_question_index in page_plan:
            page_output = f"{output_dir}/{args.worksheet_id}_page{page_no}.pdf"
            generate_worksheet_pdf(
                worksheet_id=args.worksheet_id,
                worksheet_json_filename=args.worksheet_json_filename,
                tags_folder_path=args.tags_folder_path,
                output_path=page_output,
                base_dir=args.base_dir,
                template_name=args.template_name,
                template_filename=args.template_file,
                worksheet_json_path=args.worksheet_json_path,
                page_no=page_no,
                first_question_index=first_question_index,
            )


__all__ = ["generate_worksheet_pdf"]
