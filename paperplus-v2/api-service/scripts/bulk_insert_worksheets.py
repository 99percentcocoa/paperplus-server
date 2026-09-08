#!/usr/bin/env python3
"""Bulk-insert already-generated worksheet JSON files into the database.

Reads generated worksheet JSON files from <storage_root>/json/ (default ./files/json) and
inserts each one by its worksheet_id, without requiring a separate CSV mapping file.
Worksheet generation itself is not ported here — point this at JSON already produced by the
old worksheet_json_generator.py.

Typical usage:
    python3 scripts/bulk_insert_worksheets.py --dry-run
    python3 scripts/bulk_insert_worksheets.py
    python3 scripts/bulk_insert_worksheets.py --lang mr
    python3 scripts/bulk_insert_worksheets.py --force
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import Session

from app.core.config import settings
from app.db.session import engine
from app.domain.worksheet_import import infer_worksheet_category, insert_worksheet, parse_generated_filename, worksheet_exists


def iter_generated_files(json_dir: Path, lang_filter: str | None = None) -> list[Path]:
    """Return generated JSON files from json_dir, optionally filtered by language."""
    if not json_dir.exists():
        return []

    files = []
    for path in sorted(json_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        try:
            details = parse_generated_filename(path.name)
        except ValueError:
            continue
        if lang_filter and details.get("language") != lang_filter:
            continue
        files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-insert generated worksheet JSON files into the database.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be inserted without modifying the DB.")
    parser.add_argument("--force", action="store_true", help="Attempt insert even when a worksheet_id already exists.")
    parser.add_argument("--lang", metavar="LANG", help="Only process files for this language code (en/mr).")
    parser.add_argument("--json-dir", default=None, help="Directory to scan for generated worksheet JSON files (default: <storage_root>/json).")
    args = parser.parse_args()

    json_dir = Path(args.json_dir) if args.json_dir else Path(settings.storage_root) / "json"
    files = iter_generated_files(json_dir, args.lang)
    if not files:
        print(f"No generated worksheet JSON files found in {json_dir}.")
        sys.exit(0)

    inserted = skipped = errors = 0
    with Session(engine) as session:
        for json_path in files:
            filename = json_path.name
            try:
                details = parse_generated_filename(filename)
                worksheet_id = details.get("worksheet_id")
                if worksheet_id is None:
                    with open(json_path, "r", encoding="utf-8") as f:
                        worksheet_json = json.load(f)
                    worksheet_id = worksheet_json.get("worksheet_id")

                if worksheet_id is not None and not args.force and not args.dry_run and worksheet_exists(session, int(worksheet_id)):
                    print(f"[SKIP]     {filename}  ->  worksheet_id={worksheet_id}  (already in DB)")
                    skipped += 1
                    continue

                with open(json_path, "r", encoding="utf-8") as f:
                    worksheet_json = json.load(f)

                category = infer_worksheet_category(worksheet_json, filename)
                if args.dry_run:
                    target_id = worksheet_id if worksheet_id is not None else "auto"
                    print(f"[DRY-RUN]  {filename}  ->  worksheet_id={target_id}  category={category}")
                    inserted += 1
                    continue

                result = insert_worksheet(
                    session,
                    worksheet_json,
                    worksheet_id=int(worksheet_id) if worksheet_id is not None else None,
                    worksheet_category=category,
                )
                print(
                    f"[OK]       {filename}  ->  worksheet_id={result['worksheet_id']}  "
                    f"({len(result['question_ids'])} questions, category={category})"
                )
                inserted += 1
            except Exception as exc:
                session.rollback()
                print(f"[ERROR]    {filename}  ->  {exc}")
                errors += 1

    label = "would insert" if args.dry_run else "inserted"
    print(f"\nDone: {label}={inserted}, skipped={skipped}, errors={errors}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
