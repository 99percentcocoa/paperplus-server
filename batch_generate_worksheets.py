"""Generate a single worksheet at a time, for either homework or practice.

Typical usage:
  python3 batch_generate_worksheets.py --type homework --level A --language en --worksheet-id 8001
  python3 batch_generate_worksheets.py --type practice --level D5 --language mr --worksheet-id 8002

Generated files are saved as:
  files/json/<worksheet_id>_<lang>.json
  files/pdf/student_test/<worksheet_id>_<lang>.pdf
"""

import argparse
import sys
from pathlib import Path

# Run from the repo root so imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SETTINGS
from worksheet_json_generator import (
    create_worksheet_json,
    create_practice_worksheet_json,
    parse_practice_level,
    save_worksheet,
)
from worksheet_pdf_generator import generate_worksheet_pdf

JSON_DIR = Path(SETTINGS.WORKSHEET_JSON_PATH)
PDF_DIR = Path(SETTINGS.PDF_WRITE_PATH, "student_test")
DEFAULT_START_ID = 8001
VALID_HOMEWORK_LEVELS = "ABCDEFG"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a single worksheet for practice or homework.")
    parser.add_argument("--type", dest="worksheet_type", choices=["practice", "homework"], default="homework",
                        help="Worksheet category to generate.")
    parser.add_argument("--level", required=True,
                        help="Homework level: A-G. Practice level: A1, D5, etc.")
    parser.add_argument("--language", choices=["en", "mr"], default="en",
                        help="Worksheet language.")
    parser.add_argument("--worksheet-id", type=int, default=DEFAULT_START_ID,
                        help="Numeric worksheet ID to assign to the generated file.")
    parser.add_argument("--title", default=None,
                        help="Optional title override for the worksheet JSON.")
    args = parser.parse_args()

    JSON_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    worksheet_json = build_worksheet_json(
        level=args.level,
        language=args.language,
        worksheet_type=args.worksheet_type,
        title=args.title,
    )

    json_filename = f"{args.worksheet_id}_{args.language}.json"
    json_filepath = JSON_DIR / json_filename
    pdf_filepath = PDF_DIR / f"{args.worksheet_id}_{args.language}.pdf"

    save_worksheet(worksheet_json, str(json_filepath))
    generate_worksheet_pdf(
        worksheet_id=args.worksheet_id,
        worksheet_json_filename=json_filename,
        output_path=str(pdf_filepath),
    )

    print(f"Generated {args.worksheet_type} worksheet id={args.worksheet_id} level={worksheet_json.get('level')} lang={args.language}")
    print(f"JSON: {json_filepath}")
    print(f"PDF:  {pdf_filepath}")


if __name__ == "__main__":
    main()
