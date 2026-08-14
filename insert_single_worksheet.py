#!/usr/bin/env python3
"""Insert one worksheet JSON file into the database.

Examples:
    python3 insert_single_worksheet.py --json-file 8001_en.json --id 999 --type homework
    python3 insert_single_worksheet.py --json-file files/json/8001_en.json --id 123 --type practice
    python3 insert_single_worksheet.py --json-file 8001_en.json --id 999 --type homework --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from db.flows import add_worksheet_to_db


def resolve_json_path(value: str) -> Path:
    """Resolve a user-provided JSON filename or path to a real file."""
    raw = Path(value)
    candidates = [
        raw,
        REPO_ROOT / raw,
        REPO_ROOT / "files" / "json" / raw,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    if raw.name and raw.name != str(raw):
        alt = REPO_ROOT / "files" / "json" / raw.name
        if alt.exists() and alt.is_file():
            return alt

    raise FileNotFoundError(f"Worksheet JSON not found: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Insert a single worksheet JSON file into the database.",
    )
    parser.add_argument(
        "--json-file",
        required=True,
        help="Worksheet JSON filename or path. If just a name, it is resolved inside files/json/.",
    )
    parser.add_argument(
        "--id",
        type=int,
        default=None,
        help="Optional worksheet id to assign. If omitted, the database assigns one automatically.",
    )
    parser.add_argument(
        "--type",
        dest="worksheet_type",
        choices=["practice", "homework"],
        default="practice",
        help="Worksheet category: practice or homework.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be inserted without writing to the database.",
    )
    args = parser.parse_args()

    try:
        json_path = resolve_json_path(args.json_file)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        worksheet_json = json.load(f)

    print(f"JSON file: {json_path}")
    print(f"worksheet_id: {args.id if args.id is not None else 'auto'}")
    print(f"worksheet_category: {args.worksheet_type}")

    if args.dry_run:
        print("Dry run: nothing inserted.")
        return

    result = add_worksheet_to_db(
        worksheet_json,
        worksheet_id=args.id,
        worksheet_category=args.worksheet_type,
    )

    print(f"Inserted worksheet_id={result['worksheet_id']} with {len(result['question_ids'])} questions")


if __name__ == "__main__":
    main()
