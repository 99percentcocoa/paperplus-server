"""Assignment CRUD operations."""

from .connection import get_connection


def create_assignment(student_id: str, worksheet_id: int) -> int:
    """Assign a worksheet to a student. Returns the new assignment_id."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO assignments (student_id, worksheet_id)
                   VALUES (%s, %s)
                   RETURNING assignment_id""",
                (student_id, worksheet_id),
            )
            return cur.fetchone()["assignment_id"]


def get_assignment(assignment_id: int) -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM assignments WHERE assignment_id = %s", (assignment_id,))
            return cur.fetchone()


def get_assignment_by_student_worksheet(student_id: str, worksheet_id: int) -> dict | None:
    """Find the most recent unsubmitted assignment for a student + worksheet pair."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM assignments
                   WHERE student_id = %s AND worksheet_id = %s AND submitted_at IS NULL
                   ORDER BY assigned_at DESC LIMIT 1""",
                (student_id, worksheet_id),
            )
            return cur.fetchone()


def get_assignment_by_worksheet(worksheet_id: int) -> dict | None:
    """Find the most recent unsubmitted assignment for a worksheet."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM assignments
                   WHERE worksheet_id = %s AND submitted_at IS NULL
                   ORDER BY assigned_at DESC LIMIT 1""",
                (worksheet_id,),
            )
            return cur.fetchone()


def mark_assignment_submitted(assignment_id: int):
    """Set the submitted_at timestamp on an assignment."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE assignments SET submitted_at = now() WHERE assignment_id = %s",
                (assignment_id,),
            )


def list_assignments(student_id: str = None, pending_only: bool = False) -> list[dict]:
    """List assignments, optionally filtered."""
    clauses, params = [], []
    if student_id:
        clauses.append("student_id = %s")
        params.append(student_id)
    if pending_only:
        clauses.append("submitted_at IS NULL")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM assignments {where} ORDER BY assigned_at DESC", params
            )
            return cur.fetchall()
