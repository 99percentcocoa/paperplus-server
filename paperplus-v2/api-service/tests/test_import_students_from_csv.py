"""Unit tests for scripts/import_students_from_csv.py's CSV parsing -- the DB-writing half
(upsert_student/upsert_school) is already covered by tests/test_worksheet_import.py."""

from pathlib import Path

from scripts.import_students_from_csv import extract_student_id, extract_student_name, student_rows_from_csv


def _write_csv(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "students.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_extract_student_id_matches_id_and_roll_columns_but_not_school_id():
    assert extract_student_id({"Student ID": "0007", "Name": "Asha"}) == "0007"
    assert extract_student_id({"Roll Number": "42", "Name": "Asha"}) == "42"
    assert extract_student_id({"school_id": "PSV", "Name": "Asha"}) is None
    assert extract_student_id({"Name": "Asha"}) is None


def test_student_rows_from_csv_with_explicit_ids(tmp_path):
    csv_path = _write_csv(tmp_path, "student_id,student_name\n0007,Asha\n0008,Rohan\n")
    assert student_rows_from_csv(csv_path) == [("0007", "Asha"), ("0008", "Rohan")]


def test_student_rows_from_csv_without_id_column_leaves_id_none(tmp_path):
    csv_path = _write_csv(tmp_path, "student_name\nAsha\nRohan\n")
    assert student_rows_from_csv(csv_path) == [(None, "Asha"), (None, "Rohan")]


def test_student_rows_from_csv_skips_blank_names_and_blank_ids(tmp_path):
    csv_path = _write_csv(tmp_path, "student_id,student_name\n,Asha\n0009,\n0010,Rohan\n")
    assert student_rows_from_csv(csv_path) == [(None, "Asha"), ("0010", "Rohan")]


def test_extract_student_name_supports_marathi_header(tmp_path):
    assert extract_student_name({"विद्यार्थ्याचे नाव": "Asha"}) == "Asha"
