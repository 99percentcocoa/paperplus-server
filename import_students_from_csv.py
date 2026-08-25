#!/usr/bin/env python3
"""Import a student CSV file into the database.

The file can contain student rows with names and class/grade info. Each imported
student is assigned a four-digit numeric ID starting at 0002, while 0001 is
reserved for a test student.
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from db import upsert_school, upsert_student, next_student_id


def normalize_student_id(value):
    if value is None:
        raise ValueError("student_id is required")
    text = str(value).strip()
    if not text:
        raise ValueError("student_id cannot be blank")
    if not text.isdigit():
        raise ValueError(f"student_id must be numeric, got {value!r}")
    return text.zfill(4)


def next_student_id_from_rows(existing_ids):
    return next_student_id(existing_ids)


def extract_student_name(row):
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


def student_rows_from_csv(csv_path, school_code):
    """Yield normalized student rows from a CSV file."""
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is missing a header row: {csv_path}")

        rows = []
        for row in reader:
            name = extract_student_name(row)
            if not name:
                continue
            rows.append({
                "student_name": name,
                "school_code": school_code,
            })

    return rows


def import_students_from_csv(csv_path, school_code, start_id=None):
    """Insert students from a CSV file for the provided school code."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if school_code is None or not str(school_code).strip():
        raise ValueError("school_code is required")

    school_code = str(school_code).strip()
    upsert_school(school_code, school_code)

    existing_ids = []
    try:
        from db.connection import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT student_id FROM students WHERE student_id ~ '^[0-9]{4}$' ORDER BY student_id"
                )
                existing_ids = [row["student_id"] for row in cur.fetchall()]
    except Exception:
        existing_ids = []

    next_id = normalize_student_id(start_id or next_student_id(existing_ids))
    imported = []

    for row in student_rows_from_csv(csv_path, school_code):
        student_name = row["student_name"]
        upsert_student(next_id, student_name, school_code)
        imported.append(next_id)
        next_id = normalize_student_id(int(next_id) + 1)

    return imported


def main():
    parser = argparse.ArgumentParser(description="Import students from a CSV file into the database.")
    parser.add_argument("csv_path", help="Path to the student CSV file.")
    parser.add_argument("--school-code", help="School code to assign to all imported students. If omitted, you will be prompted.")
    parser.add_argument("--start-id", type=int, default=None, help="Optional numeric starting ID. Defaults to 0002 and increments from there.")
    args = parser.parse_args()

    school_code = args.school_code
    if not school_code:
        school_code = input("Enter school code: ").strip()

    try:
        imported = import_students_from_csv(args.csv_path, school_code, args.start_id)
        print(f"Imported {len(imported)} students for school {args.school_code}.")
        if imported:
            print(f"First ID: {imported[0]} | Last ID: {imported[-1]}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
