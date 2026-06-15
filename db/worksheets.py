"""Worksheet CRUD operations."""

import json

from .connection import get_connection


def create_worksheet(worksheet_json: dict, is_test: bool = False, worksheet_id: int = None) -> int:
    """Insert a new worksheet row and return the worksheet_id.

    If *worksheet_id* is supplied the row is inserted with that exact ID and
    the serial sequence is advanced to MAX(worksheet_id) so subsequent
    auto-generated IDs never collide with it.
    """
    lang = worksheet_json.get("language")
    level = worksheet_json.get("level")
    title = worksheet_json.get("title")
    max_score = len(worksheet_json.get("questions", []))
    with get_connection() as conn:
        with conn.cursor() as cur:
            if worksheet_id is not None:
                cur.execute(
                    """INSERT INTO worksheets (worksheet_id, worksheet_level, is_test, max_score, lang, title, worksheet_json)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       RETURNING worksheet_id""",
                    (worksheet_id, level, is_test, max_score, lang, title, json.dumps(worksheet_json)),
                )
                inserted_id = cur.fetchone()["worksheet_id"]
                # Keep the sequence in sync so future auto-IDs don't collide.
                cur.execute(
                    """SELECT setval(
                           pg_get_serial_sequence('worksheets', 'worksheet_id'),
                           (SELECT MAX(worksheet_id) FROM worksheets)
                       )"""
                )
                return inserted_id
            else:
                cur.execute(
                    """INSERT INTO worksheets (worksheet_level, is_test, max_score, lang, title, worksheet_json)
                       VALUES (%s, %s, %s, %s, %s, %s)
                       RETURNING worksheet_id""",
                    (level, is_test, max_score, lang, title, json.dumps(worksheet_json)),
                )
                return cur.fetchone()["worksheet_id"]


def check_test(worksheet_id: int) -> bool:
    """Check if a worksheet is a test."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT is_test FROM worksheets WHERE worksheet_id = %s", (worksheet_id,))
            row = cur.fetchone()
            return row["is_test"] if row else False


def get_worksheet(worksheet_id: int) -> dict | None:
    """Return a worksheet row (including its JSON) by ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM worksheets WHERE worksheet_id = %s", (worksheet_id,))
            return cur.fetchone()


def get_worksheet_json(worksheet_id: int) -> dict | list | None:
    """Return only the worksheet_json for PDF generation."""
    row = get_worksheet(worksheet_id)
    return row["worksheet_json"] if row else None


def get_answer_key(worksheet_id: int) -> list | None:
    """Extract the answer key from a worksheet's JSON."""
    ws_json = get_worksheet_json(worksheet_id)
    if ws_json is None:
        return None
    questions = ws_json if isinstance(ws_json, list) else ws_json.get("questions", [])
    return [q["correct_option"] for q in questions]


def list_worksheets(level: str = None, lang: str = None, is_test: bool = None) -> list[dict]:
    """List worksheets with optional filters."""
    clauses, params = [], []
    if level:
        clauses.append("worksheet_level = %s")
        params.append(level)
    if lang:
        clauses.append("lang = %s")
        params.append(lang)
    if is_test is not None:
        clauses.append("is_test = %s")
        params.append(is_test)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM worksheets {where} ORDER BY worksheet_id", params)
            return cur.fetchall()
