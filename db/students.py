"""Student CRUD operations."""

from .connection import get_connection


def normalize_student_id(student_id: str | int) -> str:
    """Return a 4-digit student ID string like 0002."""
    if student_id is None:
        raise ValueError("student_id is required")

    value = str(student_id).strip()
    if not value:
        raise ValueError("student_id cannot be blank")

    if not value.isdigit():
        raise ValueError(f"student_id must be numeric, got {student_id!r}")

    return value.zfill(4)


def next_student_id(existing_ids: list[str] | None = None) -> str:
    """Return the next default student ID, starting at 0002."""
    if existing_ids is None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT student_id FROM students WHERE student_id ~ '^[0-9]{4}$'")
                existing_ids = [row["student_id"] for row in cur.fetchall()]

    numeric_ids = []
    for student_id in existing_ids or []:
        if student_id is None:
            continue
        normalized = str(student_id).strip()
        if normalized.isdigit():
            numeric_ids.append(int(normalized))

    if not numeric_ids:
        return "0002"

    return str(max(numeric_ids) + 1).zfill(4)


def upsert_student(student_id: str, student_name: str, school_code: str, current_level: str = "A"):
    """Insert a student or do nothing on conflict."""
    student_id = normalize_student_id(student_id)
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
