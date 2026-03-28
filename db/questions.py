"""Question CRUD operations."""

import json

from .connection import get_connection


def insert_questions_for_worksheet(worksheet_id: int, questions: list[dict]) -> list[int]:
    """Bulk-insert questions for a worksheet. Returns generated question_ids."""
    question_ids = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            for position, q in enumerate(questions, start=1):
                skill_code = q["skill_code"]
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
