"""Question CRUD operations."""

import json

from .connection import get_connection
from .skills import upsert_skill


def insert_questions_for_worksheet(worksheet_id: int, questions: list[dict]) -> list[int]:
    """Bulk-insert questions for a worksheet.

    Legacy worksheets include a per-question `skill_code`. New OMR sheets do not,
    so they are stored under a synthetic `omr` skill that keeps the database schema
    valid without breaking older worksheet imports.
    """
    upsert_skill("omr", "OMR placeholder skill", "omr", 1.0)
    question_ids = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            for position, q in enumerate(questions, start=1):
                skill_code = q.get("skill_code") or q.get("skill") or "omr"
                index = q.get("index", position)
                cur.execute(
                    """INSERT INTO questions (worksheet_id, skill_code, index, question_json)
                       VALUES (%s, %s, %s, %s)
                       RETURNING question_id""",
                    (worksheet_id, skill_code, index, json.dumps(q)),
                )
                question_ids.append(cur.fetchone()["question_id"])
    return question_ids


def get_questions_for_worksheet(worksheet_id: int) -> list[dict]:
    """Return all question rows for a worksheet, ordered by index."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM questions WHERE worksheet_id = %s ORDER BY index",
                (worksheet_id,),
            )
            return cur.fetchall()


def get_question(question_id: int) -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM questions WHERE question_id = %s", (question_id,))
            return cur.fetchone()
