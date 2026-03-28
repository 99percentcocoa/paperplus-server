"""Student CRUD operations."""

from .connection import get_connection


def upsert_student(student_id: str, student_name: str, school_code: str, current_level: str = "A"):
    """Insert a student or do nothing on conflict."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO students (student_id, student_name, student_school_code, current_level)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (student_id) DO NOTHING""",
                (student_id, student_name, school_code, current_level),
            )


def get_student(student_id: str) -> dict | None:
    """Return a student row or None."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM students WHERE student_id = %s", (student_id,))
            return cur.fetchone()


def update_student_level(student_id: str, new_level: str):
    """Change a student's current worksheet level."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE students SET current_level = %s WHERE student_id = %s",
                (new_level, student_id),
            )


def deactivate_student(student_id: str):
    """Mark a student as inactive."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE students SET is_active = false WHERE student_id = %s",
                (student_id,),
            )


def list_students(school_code: str = None, active_only: bool = True) -> list[dict]:
    """Return students, optionally filtered by school and active status."""
    clauses, params = [], []
    if school_code:
        clauses.append("student_school_code = %s")
        params.append(school_code)
    if active_only:
        clauses.append("is_active = true")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM students {where} ORDER BY student_name", params)
            return cur.fetchall()
