"""Schema migrations."""

import logging
from pathlib import Path

from .connection import get_connection

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def run_migrations():
    """Apply all SQL migration files that have not yet been applied."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version text PRIMARY KEY,
                    applied_at timestamp with time zone DEFAULT now()
                )
            """)
            cur.execute("SELECT version FROM schema_migrations")
            applied = {row["version"] for row in cur.fetchall()}

        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        for mf in migration_files:
            version = mf.stem
            if version in applied:
                continue
            logger.info("Applying migration %s", version)
            sql = mf.read_text()
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)",
                    (version,),
                )
        conn.commit()
