import logging
import json
import argparse
from pathlib import Path
from typing import Optional, Any
from services.pdf_generator_service import generate_tags_html, generate_questions_html
from config import SETTINGS

TAGS_PATH = SETTINGS.TAGS_PATH
TEMPLATES_PATH = SETTINGS.TEMPLATES_PATH
PDF_WRITE_PATH = SETTINGS.PDF_WRITE_PATH
WORKSHEET_JSON_PATH = SETTINGS.WORKSHEET_JSON_PATH
HTML_BASE_DIR = SETTINGS.HTML_BASE_DIR

logger = logging.getLogger(__name__)

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

    if template_name == "basic_omr":
        # OMR sheets are intentionally language-independent and may omit language metadata.
        return template_name, "template_en.html"

    language = worksheet.get("language")
    if language == "en":
        return template_name, "template_en.html"
    if language == "mr":
        return template_name, "template_mr.html"

    raise ValueError("worksheet must specify a supported language or explicit template filename")


def generate_worksheet_pdf(
    worksheet_id: int,
    worksheet_json_filename: str,
    tags_folder_path: Optional[str] = None,
    output_path: Optional[str] = PDF_WRITE_PATH,
    base_dir: Optional[str] = HTML_BASE_DIR,
    template_name: Optional[str] = "regular",
    template_filename: Optional[str] = None,
    worksheet_json_path: Optional[str] = None,
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
        questions_html = worksheet.get("questions_html", "")
    else:
        questions_html = generate_questions_html(worksheet_id, questions, str(effective_tags_folder_path))

    # corner tags are fixed: 0, 1, 2, 3
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
        .replace("{{question_count}}", str(worksheet.get("question_count", len(questions))))
    )

    # Import weasyprint lazily so errors are raised only when generating
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

    args = parser.parse_args()
    generate_worksheet_pdf(
        worksheet_id=args.worksheet_id,
        worksheet_json_filename=args.worksheet_json_filename,
        tags_folder_path=args.tags_folder_path,
        output_path=f"{args.output_path}/{args.worksheet_id}.pdf",
        base_dir=args.base_dir,
        template_name=args.template_name,
        template_filename=args.template_file,
        worksheet_json_path=args.worksheet_json_path,
    )


__all__ = ["generate_worksheet_pdf"]
