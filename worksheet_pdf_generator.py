import json
import argparse
from pathlib import Path
from typing import Optional
from services.pdf_generator_service import generate_tags_html, generate_questions_html
from config import SETTINGS

TAGS_PATH = SETTINGS.TAGS_PATH
TEMPLATES_PATH = SETTINGS.TEMPLATES_PATH
PDF_WRITE_PATH = SETTINGS.PDF_WRITE_PATH
WORKSHEET_JSON_PATH = SETTINGS.WORKSHEET_JSON_PATH
HTML_BASE_DIR = SETTINGS.HTML_BASE_DIR

# python3 worksheet_pdf_generator.py --worksheet-id 1 --worksheet-json-filename en_level_A.json

def generate_worksheet_pdf(
    worksheet_id: int,
    worksheet_json_filename: str,
    tags_folder_path: Optional[str] = None,
    output_path: Optional[str] = PDF_WRITE_PATH,
    base_dir: Optional[str] = HTML_BASE_DIR
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

    # open worksheet json once and select language-specific template
    with open(Path(WORKSHEET_JSON_PATH) / worksheet_json_filename, "r", encoding="utf-8") as f:
        worksheet_data = json.load(f)

    # Support both formats:
    # 1) schema object: {"language": "mr", "questions": [...]}
    # 2) legacy list: [{"language": "mr", "questions": [...]}]
    if isinstance(worksheet_data, list):
        if not worksheet_data:
            raise ValueError("worksheet JSON list is empty")
        worksheet = worksheet_data[0]
    elif isinstance(worksheet_data, dict):
        worksheet = worksheet_data
    else:
        raise ValueError("worksheet JSON must be an object or a non-empty list")

    language = worksheet.get("language")
    questions = worksheet["questions"]

    if language == "en":
        template_filename = "template_en.html"
    elif language == "mr":
        template_filename = "template_mr.html"
    else:
        raise ValueError("worksheet 'language' must be 'en' or 'mr'")

    # read template from package directory
    template_path = effective_templates_path / template_filename
    with open(template_path, "r", encoding="utf-8") as f:
        template_html = f.read()

    # generate questions HTML and tags HTML
    questions_html = generate_questions_html(worksheet_id, questions, str(effective_tags_folder_path))

    # corner tags are fixed: 0, 1, 2, 3
    tags_html = generate_tags_html([0, 1, 2, 3], str(effective_tags_folder_path))
    # tags_html = generate_tags_html(getTagNumbers(worksheet_id), str(effective_tags_folder_path))
    # tags_html = generate_cctag_html([0,1,2,3], str(effective_tags_folder_path))

    # fill template placeholders
    final_html = (
        template_html
        .replace("{{tags_html}}", tags_html)
        .replace("{{questions}}", questions_html)
        .replace("{{worksheet_id}}", str(worksheet_id))
    )

    # create PDF
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

    args = parser.parse_args()
    generate_worksheet_pdf(
        worksheet_id=args.worksheet_id,
        worksheet_json_filename=args.worksheet_json_filename,
        tags_folder_path=args.tags_folder_path,
        output_path=f"{args.output_path}/{args.worksheet_id}.pdf",
        base_dir=args.base_dir,
    )


__all__ = ["generate_worksheet_pdf"]
