"""Bulk-insert generated worksheet JSON files into the database.

This script reads generated worksheet JSON files from files/json/ and inserts each
one by its worksheet_id, without requiring a separate CSV mapping file.

Typical usage:
  python3 bulk_insert_worksheets.py --dry-run
  python3 bulk_insert_worksheets.py
  python3 bulk_insert_worksheets.py --lang mr
  python3 bulk_insert_worksheets.py --force
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from db.flows import add_worksheet_to_db
from db.worksheets import get_worksheet

JSON_DIR = REPO_ROOT / "files" / "json"
FILENAME_RE = re.compile(
    r"^(?:"
    r"(?P<worksheet_id>\d+)_(?P<language>[a-z]{2})|"
    r"(?P<language2>[a-z]{2})_(?P<category>practice|homework)_(?P<level>.+?)(?:_(?P<index>\d+))?|"
    r"(?P<category2>homework|practice)_(?P<level2>.+)|"
    r"(?P<worksheet_id2>\d+)_(?P<language3>[a-z]{2})_(?P<category3>practice|homework)"
    r")\.json$"
)


def parse_generated_filename(filename: str) -> dict:
    """Infer worksheet_id, language, and category from a generated JSON filename."""
    name = Path(filename).name
    match = FILENAME_RE.match(name)
    if not match:
        raise ValueError(f"Unsupported generated worksheet filename: {filename!r}")

    data = {
        "worksheet_id": None,
        "language": None,
        "worksheet_category": None,
    }
    if match.group("worksheet_id"):
        data["worksheet_id"] = int(match.group("worksheet_id"))
        data["language"] = match.group("language")
    elif match.group("language2"):
        data["language"] = match.group("language2")
        data["worksheet_category"] = match.group("category")
    elif match.group("category2"):
        data["worksheet_category"] = match.group("category2")
        data["language"] = None
    elif match.group("worksheet_id2"):
        data["worksheet_id"] = int(match.group("worksheet_id2"))
        data["language"] = match.group("language3")
        data["worksheet_category"] = match.group("category3")

    return data


def worksheet_exists(worksheet_id: int) -> bool:
    """Return True if the worksheet is already in the database."""
    try:
        return get_worksheet(worksheet_id) is not None
    except Exception:
        return False


def infer_worksheet_category(worksheet_json: dict, filename: str | None = None) -> str:
    """Determine the worksheet category from the JSON payload or filename."""
    if isinstance(worksheet_json, dict):
        category = worksheet_json.get("worksheet_category")
        if category in {"practice", "homework"}:
            return category

    if filename:
        name = Path(filename).name.lower()
        if "practice" in name:
            return "practice"
        if "homework" in name:
            return "homework"

    return "practice"


def iter_generated_files(lang_filter: str | None = None, subfolder: str | None = None) -> list[Path]:
    """Return generated JSON files from a directory, optionally filtered by language."""
    files = []
    base_dir = JSON_DIR / subfolder if subfolder else JSON_DIR
    if not base_dir.exists():
        return files

    iterator = base_dir.rglob("*.json") if subfolder else sorted(base_dir.iterdir())
    if subfolder:
        for path in sorted(iterator):
            if not path.is_file():
                continue
            try:
                details = parse_generated_filename(path.name)
            except ValueError:
                continue
            if lang_filter and details.get("language") != lang_filter:
                continue
            files.append(path)
        return files

    for path in iterator:
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
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be inserted without modifying the DB.")
    parser.add_argument("--force", action="store_true",
                        help="Attempt insert even when a worksheet_id already exists.")
    parser.add_argument("--lang", metavar="LANG",
                        help="Only process files for this language code (en/mr).")
    parser.add_argument("--subfolder", default=None,
                        help="Optional subfolder under files/json to scan for generated worksheet JSON files.")
    args = parser.parse_args()

    files = iter_generated_files(args.lang, args.subfolder)
    if not files:
        target_dir = JSON_DIR / args.subfolder if args.subfolder else JSON_DIR
        print(f"No generated worksheet JSON files found in {target_dir}.")
        sys.exit(0)

    inserted = skipped = errors = 0
    for json_path in files:
        filename = json_path.name
        try:
            details = parse_generated_filename(filename)
            worksheet_id = details.get("worksheet_id")
            if worksheet_id is None:
                with open(json_path, "r", encoding="utf-8") as f:
                    worksheet_json = json.load(f)
                worksheet_id = worksheet_json.get("worksheet_id")

            if worksheet_id is not None and not args.force and not args.dry_run and worksheet_exists(int(worksheet_id)):
                print(f"[SKIP]     {filename}  →  worksheet_id={worksheet_id}  (already in DB)")
                skipped += 1
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                worksheet_json = json.load(f)

            category = infer_worksheet_category(worksheet_json, filename)
            if args.dry_run:
                target_id = worksheet_id if worksheet_id is not None else "auto"
                print(f"[DRY-RUN]  {filename}  →  worksheet_id={target_id}  category={category}")
                inserted += 1
                continue

            result = add_worksheet_to_db(
                worksheet_json,
                worksheet_id=int(worksheet_id) if worksheet_id is not None else None,
                worksheet_category=category,
            )
            print(f"[OK]       {filename}  →  worksheet_id={result['worksheet_id']}  "
                  f"({len(result['question_ids'])} questions, category={category})")
            inserted += 1
        except Exception as exc:
            print(f"[ERROR]    {filename}  →  {exc}")
            errors += 1

    label = "would insert" if args.dry_run else "inserted"
    print(f"\nDone: {label}={inserted}, skipped={skipped}, errors={errors}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
