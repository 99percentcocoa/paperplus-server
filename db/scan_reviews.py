"""Scan review queue CRUD helpers."""

import json

from .connection import get_connection


def create_scan_review(
    *,
    submission_id: int = None,
    student_id: str = None,
    worksheet_id: int = None,
    detected_roll_number: str = None,
    status: str = "failed",
    error_reason: str = None,
    original_answers: list | dict = None,
    original_score: int = 0,
) -> int:
    """Create a failed or review-needed scan record."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO scan_reviews (
                    submission_id,
                    student_id,
                    worksheet_id,
                    detected_roll_number,
                    status,
                    error_reason,
                    original_answers,
                    original_score
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING review_id
                """,
                (
                    submission_id,
                    student_id,
                    worksheet_id,
                    detected_roll_number,
                    status,
                    error_reason,
                    json.dumps(original_answers) if original_answers is not None else None,
                    original_score,
                ),
            )
            return cur.fetchone()["review_id"]


def get_scan_review(review_id: int) -> dict | None:
    """Fetch one review record by id."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM scan_reviews WHERE review_id = %s", (review_id,))
            return cur.fetchone()


def list_scan_reviews(status: str = None, limit: int = 100) -> list[dict]:
    """List review records, optionally filtered by status."""
    query = "SELECT * FROM scan_reviews"
    params = []
    if status:
        query += " WHERE status = %s"
        params.append(status)
    query += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            return cur.fetchall()


def correct_scan_review(
    review_id: int,
    *,
    corrected_student_id: str,
    corrected_answers: list | dict,
    corrected_score: int,
    corrected_by: str = "admin",
) -> dict:
    """Write a corrected value set back to the review row."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE scan_reviews
                SET student_id = %s,
                    detected_roll_number = %s,
                    corrected_answers = %s,
                    corrected_score = %s,
                    status = 'corrected',
                    corrected_by = %s,
                    corrected_at = NOW(),
                    updated_at = NOW()
                WHERE review_id = %s
                RETURNING *
                """,
                (
                    corrected_student_id,
                    corrected_student_id,
                    json.dumps(corrected_answers),
                    corrected_score,
                    corrected_by,
                    review_id,
                ),
            )
            return cur.fetchone()
