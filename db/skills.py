"""Skill CRUD operations."""

import json
from pathlib import Path

from .connection import get_connection


def upsert_skill(skill_code: str, skill_name: str, skill_level: str, skill_weight: float = 1.0):
    """Insert or update a skill."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO skills (skill_code, skill_name, skill_level, skill_weight)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (skill_code) DO UPDATE
                       SET skill_name = EXCLUDED.skill_name,
                           skill_level = EXCLUDED.skill_level,
                           skill_weight = EXCLUDED.skill_weight""",
                (skill_code, skill_name, skill_level, skill_weight),
            )


def get_skill(skill_code: str) -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM skills WHERE skill_code = %s", (skill_code,))
            return cur.fetchone()


def list_skills(level: str = None) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            if level:
                cur.execute("SELECT * FROM skills WHERE skill_level = %s ORDER BY skill_code", (level,))
            else:
                cur.execute("SELECT * FROM skills ORDER BY skill_code")
            return cur.fetchall()


def import_skills_from_json(json_path: str | Path):
    """Bulk-import skills from the services/skills.json file."""
    data = json.loads(Path(json_path).read_text())
    skills = data if isinstance(data, list) else list(data.values())
    with get_connection() as conn:
        with conn.cursor() as cur:
            for s in skills:
                cur.execute(
                    """INSERT INTO skills (skill_code, skill_name, skill_level)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (skill_code) DO UPDATE
                           SET skill_name = EXCLUDED.skill_name,
                               skill_level = EXCLUDED.skill_level""",
                    (s["code"], s["skill"], s["difficulty_level"]),
                )
