"""
Bulk-insert pre-generated worksheet JSONs into the database.

Reads files/csv/worksheet_id_mapping.csv, loads each JSON from files/json/,
and calls add_worksheet_to_db() with the pinned worksheet_id.

Usage
-----
  # Dry run – print what would be inserted without touching the DB
  python3 bulk_insert_worksheets.py --dry-run

  # Insert only missing worksheets (safe to re-run)
  python3 bulk_insert_worksheets.py

  # Insert a single student
  python3 bulk_insert_worksheets.py --student gauri

  # Filter by student and language
  python3 bulk_insert_worksheets.py --student amod --lang mr

  # Force re-insert (skip duplicate check – will error if IDs already exist)
  python3 bulk_insert_worksheets.py --force

Remote usage
------------
  1. Copy this repo (or at least files/json/ and files/csv/) to the remote machine.
  2. Set the DATABASE_URL env variable (or populate a .env / config.py there).
  3. Run: python3 bulk_insert_worksheets.py

The script resolves all paths relative to its own location, so it works
regardless of the current working directory.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Resolve repo root relative to this script so it works from any cwd
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from db.flows import add_worksheet_to_db
from db.worksheets import get_worksheet

MAPPING_CSV = REPO_ROOT / "files" / "csv" / "student_exp.csv"
JSON_DIR    = REPO_ROOT / "files" / "json"


def load_mapping(student_filter: str | None, lang_filter: str | None) -> list[dict]:
    """Return rows from the mapping CSV, optionally filtered."""
    rows = []
    with open(MAPPING_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if student_filter and row["student"] != student_filter:
                continue
            if lang_filter and row["lang"] != lang_filter:
                continue
            rows.append(row)
    return rows


def worksheet_exists(worksheet_id: int) -> bool:
    """Return True if the worksheet is already in the database."""
    try:
        return get_worksheet(worksheet_id) is not None
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Bulk-insert worksheets into the DB.")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print what would be inserted; do not modify the DB.")
    parser.add_argument("--force",    action="store_true",
                        help="Attempt insert even if the worksheet_id already exists.")
    parser.add_argument("--student",  metavar="NAME",
                        help="Only process worksheets for this student (gauri/amod/pallavi/saarang).")
    parser.add_argument("--lang",     metavar="LANG",
                        help="Only process worksheets for this language (en/mr).")
    args = parser.parse_args()

    rows = load_mapping(args.student, args.lang)
    if not rows:
        print("No rows matched the given filters.")
        sys.exit(0)

    inserted = skipped = errors = 0

    for row in rows:
        filename     = row["filename"]
        worksheet_id = int(row["worksheet_id"])
        json_path    = JSON_DIR / filename

        if not json_path.exists():
            print(f"[MISSING]  {filename}  (expected at {json_path})")
            errors += 1
            continue

        if not args.force and not args.dry_run:
            if worksheet_exists(worksheet_id):
                print(f"[SKIP]     {filename}  →  worksheet_id={worksheet_id}  (already in DB)")
                skipped += 1
                continue

        if args.dry_run:
            print(f"[DRY-RUN]  {filename}  →  worksheet_id={worksheet_id}")
            inserted += 1
            continue

        try:
            with open(json_path) as f:
                worksheet_json = json.load(f)
            result = add_worksheet_to_db(worksheet_json, worksheet_id=worksheet_id)
            print(f"[OK]       {filename}  →  worksheet_id={result['worksheet_id']}  "
                  f"({len(result['question_ids'])} questions)")
            inserted += 1
        except Exception as exc:
            print(f"[ERROR]    {filename}  →  worksheet_id={worksheet_id}  :  {exc}")
            errors += 1

    label = "would insert" if args.dry_run else "inserted"
    print(f"\nDone: {label}={inserted}, skipped={skipped}, errors={errors}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
