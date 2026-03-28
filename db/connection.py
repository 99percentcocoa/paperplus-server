"""Shared connection helpers for the Postgres backend."""

import sys
import logging
from pathlib import Path
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SETTINGS

logger = logging.getLogger(__name__)

DATABASE_URL = SETTINGS.DATABASE_URL


@contextmanager
def get_connection():
    """Yield a psycopg connection that auto-commits on success and rolls back on error."""
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_cursor(conn=None):
    """Yield a dict-row cursor. If *conn* is None a new connection is created."""
    if conn is not None:
        with conn.cursor(row_factory=dict_row) as cur:
            yield cur
    else:
        with get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                yield cur
