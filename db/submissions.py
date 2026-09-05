"""Submission CRUD operations."""

import json

from .connection import get_connection


def create_submission(student_id: str, worksheet_id: int,
                      score: int, from_number: str, answers_json: dict | list,
                      worksheet_category: str = None) -> int:
    """Record a graded submission. Returns submission_id."""
    if worksheet_category is None:
        worksheet_category = "practice"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO submissions
                       (student_id, worksheet_id, score, from_number, answers_json, worksheet_category)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   RETURNING submission_id""",
                (student_id, worksheet_id, score, from_number,
                 json.dumps(answers_json), worksheet_category),
            )
            return cur.fetchone()["submission_id"]


def get_submission(submission_id: int) -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM submissions WHERE submission_id = %s", (submission_id,))
            return cur.fetchone()


def get_submissions_for_worksheet(worksheet_id: int, student_id: str = None,
                                 worksheet_category: str = None) -> list[dict]:
    """Find submissions for a worksheet, optionally filtered by student or category."""
    clauses = ["worksheet_id = %s"]
    params = [worksheet_id]
    if student_id:
        clauses.append("student_id = %s")
        params.append(student_id)
    if worksheet_category:
        clauses.append("worksheet_category = %s")
        params.append(worksheet_category)
    where = "WHERE " + " AND ".join(clauses)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM submissions {where} ORDER BY submitted_at DESC", params
            )
            return cur.fetchall()


def get_latest_submission(student_id: str, worksheet_id: int) -> dict | None:
    """Return the most recent submission for a student + worksheet."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM submissions
                   WHERE student_id = %s AND worksheet_id = %s
                   ORDER BY submitted_at DESC LIMIT 1""",
                (student_id, worksheet_id),
            )
            return cur.fetchone()


def get_latest_submission_by_worksheet(worksheet_id: int) -> dict | None:
    """Return the most recent submission for a worksheet (any student)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM submissions
                   WHERE worksheet_id = %s
                   ORDER BY submitted_at DESC LIMIT 1""",
                (worksheet_id,),
            )
            return cur.fetchone()


def get_combined_worksheet_answers(worksheet_id: int, student_id: str | None = None) -> list[dict]:
    """Return every answer for a worksheet ordered by question number.

    This combines multiple submitted scans for the same worksheet, which is needed
    for multi-page OMR sheets where page 1 and page 2 are scanned separately.
    If a question appears in more than one submission, the newest submission wins.
    """
    clauses = ["worksheet_id = %s"]
    params = [worksheet_id]
    if student_id:
        clauses.append("student_id = %s")
        params.append(student_id)

    where = "WHERE " + " AND ".join(clauses)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM submissions {where} ORDER BY submitted_at DESC, submission_id DESC",
                params,
            )
            rows = cur.fetchall()

    latest_by_question: dict[int, dict] = {}
    for row in rows:
        payload = row.get("answers_json")
        if payload is None:
            continue
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (TypeError, ValueError):
                continue
        if not isinstance(payload, list):
            continue

        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                q_index = int(item.get("question_index", 0))
            except (TypeError, ValueError):
                continue
            if q_index <= 0:
                continue
            latest_by_question[q_index] = {
                "question_index": q_index,
                "selected_option": item.get("selected_option", ""),
                "is_correct": bool(item.get("is_correct", False)),
                "student_id": row.get("student_id"),
                "submission_id": row.get("submission_id"),
            }

    return [latest_by_question[q_index] for q_index in sorted(latest_by_question)]


def get_worksheet_answers_and_score(worksheet_id: int, student_id: str | None = None) -> dict:
    """Return the merged full answer list and combined total score for a worksheet."""
    answers = get_combined_worksheet_answers(worksheet_id, student_id=student_id)
    total_score = sum(1 for item in answers if item.get("is_correct"))
    return {
        "worksheet_id": worksheet_id,
        "student_id": student_id,
        "answers": answers,
        "total_score": total_score,
        "total_questions": len(answers),
    }


def overwrite_submission(submission_id: int, score: int, answers_json: dict | list):
    """Overwrite an existing submission's score and answers."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE submissions
                   SET score = %s, answers_json = %s, submitted_at = now()
                   WHERE submission_id = %s""",
                (score, json.dumps(answers_json), submission_id),
            )
