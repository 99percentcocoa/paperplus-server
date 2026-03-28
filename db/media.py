"""Media reference operations."""

from .connection import get_connection


def create_media(owner_type: str, owner_id: int, storage_path: str) -> int:
    """Insert a media reference and return its media_id."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO media (owner_type, owner_id, storage_path)
                   VALUES (%s, %s, %s)
                   RETURNING media_id""",
                (owner_type, owner_id, storage_path),
            )
            return cur.fetchone()["media_id"]


def get_media(owner_type: str, owner_id: int) -> list[dict]:
    """Return all media rows for a given owner."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM media WHERE owner_type = %s AND owner_id = %s ORDER BY created_at",
                (owner_type, owner_id),
            )
            return cur.fetchall()
