"""Student skill mastery tracking."""

from .connection import get_connection


def update_skill_mastery(student_id: str, skill_code: str, mastery_score: float):
    """Upsert a mastery score for a student x skill pair."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO student_skill_mastery (student_id, skill_code, mastery_score)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (student_id, skill_code) DO UPDATE
                       SET mastery_score = EXCLUDED.mastery_score,
                           last_updated = now()""",
                (student_id, skill_code, mastery_score),
            )


def get_skill_mastery(student_id: str) -> list[dict]:
    """Return all mastery rows for a student."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT m.*, s.skill_name, s.skill_level
                   FROM student_skill_mastery m
                   JOIN skills s USING (skill_code)
                   WHERE m.student_id = %s
                   ORDER BY s.skill_level, s.skill_code""",
                (student_id,),
            )
            return cur.fetchall()


def recalculate_skill_mastery(student_id: str, skill_code: str):
    """Recalculate mastery for one student x skill from attempt history."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT COUNT(*) AS total,
                          SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) AS correct
                   FROM attempts
                   WHERE student_id = %s AND skill_code = %s""",
                (student_id, skill_code),
            )
            row = cur.fetchone()
            total = row["total"]
            if total == 0:
                return
            mastery = row["correct"] / total
    update_skill_mastery(student_id, skill_code, mastery)
