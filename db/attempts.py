"""Attempt CRUD operations."""

from .connection import get_connection


def insert_attempts(student_id: str, submission_id: int, worksheet_id: int,
                    attempts: list[dict]):
    """Bulk-insert attempt rows from processed answers."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            for a in attempts:
                cur.execute(
                    """INSERT INTO attempts
                           (student_id, submission_id, question_id, worksheet_id, is_correct, skill_code)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (student_id, submission_id, a["question_id"],
                     worksheet_id, a["is_correct"], a["skill_code"]),
                )


def delete_attempts_for_submission(submission_id: int):
    """Delete all attempts tied to a submission (used before overwrite)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM attempts WHERE submission_id = %s", (submission_id,))


def get_attempts_for_submission(submission_id: int) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM attempts WHERE submission_id = %s ORDER BY question_id",
                (submission_id,),
            )
            return cur.fetchall()
