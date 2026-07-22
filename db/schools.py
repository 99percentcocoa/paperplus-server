"""School CRUD operations."""

from .connection import get_connection


def upsert_school(school_code: str, school_name: str):
    """Insert a school or update on conflict."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO schools (school_code, school_name)
                   VALUES (%s, %s)
                   ON CONFLICT (school_code) DO UPDATE SET school_name = EXCLUDED.school_name""",
                (school_code, school_name),
            )


def get_school(school_code: str) -> dict | None:
    """Return a school row or None."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM schools WHERE school_code = %s", (school_code,))
            return cur.fetchone()


def list_schools() -> list[dict]:
    """Return all schools."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM schools ORDER BY school_name")
            return cur.fetchall()
