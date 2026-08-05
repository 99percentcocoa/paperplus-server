"""Student skill mastery tracking."""

import logging

from .connection import get_connection
from .students import get_student, update_student_level

logger = logging.getLogger(__name__)

# Worksheet levels A-G map onto skill difficulty levels 1-7 respectively;
# each worksheet level's dominant (50%-weighted) skill tier drives traversal.
LEVEL_ORDER = "ABCDEFG"
LEVEL_TO_DIFFICULTY = {letter: str(i + 1) for i, letter in enumerate(LEVEL_ORDER)}

ADVANCE_THRESHOLD = 0.75
REGRESS_THRESHOLD = 0.35


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


def get_level_mastery_average(student_id: str, level_letter: str) -> float | None:
    """Average mastery across all skills whose difficulty tier matches *level_letter*."""
    difficulty = LEVEL_TO_DIFFICULTY.get(level_letter)
    if difficulty is None:
        return None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT AVG(m.mastery_score) AS avg_mastery
                   FROM student_skill_mastery m
                   JOIN skills s USING (skill_code)
                   WHERE m.student_id = %s AND s.skill_level = %s""",
                (student_id, difficulty),
            )
            row = cur.fetchone()
            return float(row["avg_mastery"]) if row and row["avg_mastery"] is not None else None


def get_level_skill_coverage(student_id: str, level_letter: str) -> tuple[int, int]:
    """Return (skills_attempted, total_skills) for the difficulty tier of *level_letter*."""
    difficulty = LEVEL_TO_DIFFICULTY.get(level_letter)
    if difficulty is None:
        return (0, 0)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT
                       (SELECT COUNT(*) FROM skills WHERE skill_level = %s) AS total,
                       (SELECT COUNT(*) FROM student_skill_mastery m
                            JOIN skills s USING (skill_code)
                            WHERE m.student_id = %s AND s.skill_level = %s) AS attempted""",
                (difficulty, student_id, difficulty),
            )
            row = cur.fetchone()
            return (row["attempted"], row["total"])


def determine_level_change(current_level: str, avg_mastery: float | None, full_coverage: bool = True) -> str:
    """Pure decision function: >75% advances (only with full skill coverage), <35% regresses, otherwise stays."""
    if avg_mastery is None or current_level not in LEVEL_ORDER:
        return current_level

    index = LEVEL_ORDER.index(current_level)
    if avg_mastery > ADVANCE_THRESHOLD and full_coverage and index < len(LEVEL_ORDER) - 1:
        return LEVEL_ORDER[index + 1]
    if avg_mastery < REGRESS_THRESHOLD and index > 0:
        return LEVEL_ORDER[index - 1]
    return current_level


def evaluate_and_update_level(student_id: str) -> dict | None:
    """Recompute a student's level based on mastery of their current level's skills.

    Advancing requires the student to have attempted every skill in the current
    tier; regressing does not (it acts as a safety net regardless of coverage).
    Updates `students.current_level` in place if a threshold is crossed.
    Returns a summary dict, or None if the student doesn't exist.
    """
    student = get_student(student_id)
    if student is None:
        return None

    current_level = student["current_level"]
    if current_level is None or current_level not in LEVEL_ORDER:
        # Student was never initialized with a level; default to the base level.
        logger.info(
            "Student %s has no level assigned (was %r); defaulting to level 'A'.",
            student_id, current_level,
        )
        current_level = "A"
        update_student_level(student_id, current_level)

    avg_mastery = get_level_mastery_average(student_id, current_level)
    attempted, total = get_level_skill_coverage(student_id, current_level)
    full_coverage = total > 0 and attempted >= total
    new_level = determine_level_change(current_level, avg_mastery, full_coverage)

    if new_level != current_level:
        update_student_level(student_id, new_level)

    logger.info(
        "Level recalculation for student %s: %s -> %s (avg_mastery=%s, skills_attempted=%s/%s, changed=%s)",
        student_id, current_level, new_level, avg_mastery, attempted, total, new_level != current_level,
    )

    return {
        "student_id": student_id,
        "old_level": current_level,
        "new_level": new_level,
        "mastery_average": avg_mastery,
        "skills_attempted": attempted,
        "skills_total": total,
        "changed": new_level != current_level,
    }
