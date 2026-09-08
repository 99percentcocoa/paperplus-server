#!/usr/bin/env python3
"""Import a student CSV file into the database.

Each imported student is assigned a four-digit numeric ID starting at 0002 (0001 is
reserved for a test student), unless --start-id is given.

Example:
    python3 scripts/import_students_from_csv.py students.csv --school-code PSV
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import Session

from app.db.session import engine
from app.domain.worksheet_import import next_student_id, upsert_school, upsert_student


def extract_student_name(row: dict) -> str | None:
    for key, value in row.items():
        if value is None:
            continue
        normalized_key = str(key).strip().lower()
        if "name" not in normalized_key and "नाव" not in normalized_key:
            continue
        candidate = str(value).strip()
        if candidate:
            return candidate
    return None


def student_names_from_csv(csv_path: Path) -> list[str]:
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is missing a header row: {csv_path}")

        names = []
        for row in reader:
            name = extract_student_name(row)
            if name:
                names.append(name)
    return names


def import_students_from_csv(session: Session, csv_path: Path, school_code: str, start_id: int | None = None) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not school_code or not school_code.strip():
        raise ValueError("school_code is required")

    school_code = school_code.strip()
    upsert_school(session, school_code, school_code)

    next_id = str(start_id).zfill(4) if start_id is not None else next_student_id(session)
    imported = []
    for name in student_names_from_csv(csv_path):
        upsert_student(session, next_id, name, school_code)
        imported.append(next_id)
        next_id = str(int(next_id) + 1).zfill(4)

    session.commit()
    return imported


def main() -> None:
    parser = argparse.ArgumentParser(description="Import students from a CSV file into the database.")
    parser.add_argument("csv_path", help="Path to the student CSV file.")
    parser.add_argument("--school-code", help="School code to assign to all imported students. Prompted if omitted.")
    parser.add_argument("--start-id", type=int, default=None, help="Optional numeric starting ID. Defaults to the next free 4-digit id.")
    args = parser.parse_args()

    school_code = args.school_code or input("Enter school code: ").strip()

    try:
        with Session(engine) as session:
            imported = import_students_from_csv(session, Path(args.csv_path), school_code, args.start_id)
        print(f"Imported {len(imported)} students for school {school_code}.")
        if imported:
            print(f"First ID: {imported[0]} | Last ID: {imported[-1]}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
