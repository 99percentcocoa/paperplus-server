"""Generate a single worksheet at a time, for either homework or practice.

Typical usage:
  python3 batch_generate_worksheets.py --type homework --level A --language en --count 5
  python3 batch_generate_worksheets.py --type practice --level D5 --language mr --count 3

Generated files are saved as:
  files/json/<worksheet_id>_<lang>.json
  files/pdf/student_test/<worksheet_id>_<lang>.pdf
"""

import argparse
import sys
from pathlib import Path

# Run from the repo root so imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SETTINGS, resolve_project_path
from worksheet_json_generator import (
    create_worksheet_json,
    create_practice_worksheet_json,
    parse_practice_level,
    save_worksheet,
)
from worksheet_pdf_generator import generate_worksheet_pdf

JSON_DIR = Path(SETTINGS.WORKSHEET_JSON_PATH)
PDF_DIR = Path(SETTINGS.PDF_WRITE_PATH)
VALID_HOMEWORK_LEVELS = "ABCDEFG"
PLACEHOLDER_WORKSHEET_ID = 0


def ensure_project_directories() -> tuple[Path, Path]:
    """Return a consistent repo-relative output layout compatible with Colab and local runs."""
    json_dir = Path(SETTINGS.WORKSHEET_JSON_PATH) if SETTINGS.WORKSHEET_JSON_PATH else Path(resolve_project_path("files", "json"))
    pdf_dir = Path(SETTINGS.PDF_WRITE_PATH) if SETTINGS.PDF_WRITE_PATH else Path(resolve_project_path("files", "pdf"))
    json_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    return json_dir, pdf_dir


def resolve_level_spec(level: str, worksheet_type: str) -> dict:
    """Normalize a single user-provided level for the selected worksheet type."""
    clean_level = str(level).strip()
    if not clean_level:
        raise ValueError("Level cannot be blank.")

    if worksheet_type == "practice":
        theme, level_num = parse_practice_level(clean_level)
        return {
            "theme": theme,
            "level": level_num,
            "worksheet_category": "practice",
        }

    normalized = clean_level.upper()
    if normalized not in VALID_HOMEWORK_LEVELS:
        raise ValueError(f"Homework level must be one of {VALID_HOMEWORK_LEVELS}; got {level!r}")
    return {
        "level": normalized,
        "worksheet_category": "homework",
    }


def build_worksheet_json(level: str, language: str, worksheet_type: str, title: str | None = None) -> dict:
    """Create the worksheet JSON for either practice or homework."""
    spec = resolve_level_spec(level, worksheet_type)
    if worksheet_type == "practice":
        title = title or f"Practice Worksheet {spec['theme']}{spec['level']}"
        return create_practice_worksheet_json(
            title=title,
            theme=spec["theme"],
            level=spec["level"],
            language=language,
        )

    title = title or f"Worksheet Level {spec['level']}"
    return create_worksheet_json(
        title=title,
        level=spec["level"],
        language=language,
        worksheet_category="homework",
    )


def build_output_filename(language: str, worksheet_type: str, level: str, worksheet_id: int | None = None, index: int | None = None) -> str:
    """Return the JSON filename used for a generated sheet."""
    if worksheet_id is not None:
        return f"{worksheet_id}_{language}.json"
    level_label = str(level).strip().upper()
    if index is None:
        return f"{language}_{worksheet_type}_{level_label}.json"
    return f"{language}_{worksheet_type}_{level_label}_{index}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one or many worksheets for practice or homework.")
    parser.add_argument("--type", dest="worksheet_type", choices=["practice", "homework"], default="homework",
                        help="Worksheet category to generate.")
    parser.add_argument("--level", required=True,
                        help="Homework level: A-G. Practice level: A1, D5, etc.")
    parser.add_argument("--language", choices=["en", "mr"], default="en",
                        help="Worksheet language.")
    parser.add_argument("--worksheet-id", type=int, default=None,
                        help="Optional numeric worksheet ID for the first generated sheet. If omitted, the database assigns the next available ID.")
    parser.add_argument("--start-id", type=int, default=None,
                        help="Optional starting worksheet ID. If set, generated sheets use this as the first ID and increment by one for each sheet.")
    parser.add_argument("--count", type=int, default=1,
                        help="How many worksheets to generate for this level. Default: 1.")
    parser.add_argument("--title", default=None,
                        help="Optional title override for the worksheet JSON.")
    parser.add_argument("--subfolder", default=None,
                        help="Optional subfolder under files/json and files/pdf/student_test to save the generated worksheets.")
    args = parser.parse_args()

    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if args.worksheet_id is not None and args.start_id is not None:
        raise ValueError("Use either --worksheet-id or --start-id, not both.")

    json_dir, pdf_output_dir = ensure_project_directories()
    JSON_DIR = json_dir
    PDF_DIR = pdf_output_dir
    if args.subfolder:
        pdf_output_dir = pdf_output_dir / args.subfolder
    pdf_output_dir.mkdir(parents=True, exist_ok=True)

    current_id = args.start_id if args.start_id is not None else args.worksheet_id
    for index in range(args.count):
        worksheet_json = build_worksheet_json(
            level=args.level,
            language=args.language,
            worksheet_type=args.worksheet_type,
            title=args.title,
        )

        output_worksheet_id = current_id if current_id is not None else PLACEHOLDER_WORKSHEET_ID

        json_filename = build_output_filename(
            language=args.language,
            worksheet_type=args.worksheet_type,
            level=worksheet_json.get("level", args.level),
            worksheet_id=current_id,
            index=index if current_id is None else None,
        )
        json_filepath = JSON_DIR / json_filename
        level_label = str(worksheet_json.get("level", args.level)).strip().upper()
        pdf_filename = (
            f"{output_worksheet_id}_{args.language}_{args.worksheet_type}_{level_label}.pdf"
            if current_id is not None
            else f"{args.language}_{args.worksheet_type}_{level_label}_{index}.pdf"
        )
        pdf_filepath = pdf_output_dir / pdf_filename

        save_worksheet(worksheet_json, str(json_filepath))
        generate_worksheet_pdf(
            worksheet_id=output_worksheet_id,
            worksheet_json_filename=json_filename,
            output_path=str(pdf_filepath),
        )

        display_id = current_id if current_id is not None else "auto"
        print(f"Generated {args.worksheet_type} worksheet #{index + 1}/{args.count} id={display_id} level={worksheet_json.get('level')} lang={args.language}")
        print(f"JSON: {json_filepath}")
        print(f"PDF:  {pdf_filepath}")

        if current_id is not None:
            current_id += 1


if __name__ == "__main__":
    main()
