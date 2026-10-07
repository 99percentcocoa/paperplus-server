"""Mastery recalculation and level advancement — ported from the old db/mastery.py formulas."""

from datetime import datetime, timezone

from sqlmodel import Session, select

from app.models import Attempt, Skill, Student, StudentSkillMastery
from app.models.mastery import MasteryHistory

LEVEL_ORDER = "ABCDEFG"
LEVEL_TO_DIFFICULTY = {level: str(i + 1) for i, level in enumerate(LEVEL_ORDER)}
ADVANCE_THRESHOLD = 0.75
REGRESS_THRESHOLD = 0.35


def recalculate_skill_mastery(session: Session, student_id: str, skill_code: str) -> float | None:
    """mastery_score = correct_attempts / total_attempts for this student+skill."""
    attempts = session.exec(
        select(Attempt).where(Attempt.student_id == student_id, Attempt.skill_code == skill_code)
    ).all()
    total = len(attempts)
    if total == 0:
        return None

    correct = sum(1 for a in attempts if a.is_correct)
    mastery_score = correct / total
    now = datetime.now(timezone.utc)

    existing = session.get(StudentSkillMastery, (student_id, skill_code))
    if existing is not None:
        existing.mastery_score = mastery_score
        existing.last_updated = now
        session.add(existing)
    else:
        session.add(StudentSkillMastery(student_id=student_id, skill_code=skill_code, mastery_score=mastery_score, last_updated=now))

    session.add(MasteryHistory(student_id=student_id, skill_code=skill_code, mastery_score=mastery_score, recorded_at=now))
    session.commit()
    return mastery_score


def evaluate_and_update_level(session: Session, student_id: str) -> dict | None:
    """Advance one level if avg_mastery > 0.75 AND every skill in the tier was attempted;
    regress one level if avg_mastery < 0.35 (no coverage requirement); otherwise stay put.
    """
    student = session.get(Student, student_id)
    if student is None or not student.current_level:
        return None

    current_level = student.current_level
    difficulty = LEVEL_TO_DIFFICULTY.get(current_level)
    if difficulty is None:
        return None

    tier_skills = session.exec(select(Skill).where(Skill.skill_level == difficulty)).all()
    skills_total = len(tier_skills)
    if skills_total == 0:
        return None

    masteries = []
    for skill in tier_skills:
        mastery = session.get(StudentSkillMastery, (student_id, skill.skill_code))
        if mastery is not None and mastery.mastery_score is not None:
            masteries.append(mastery.mastery_score)

    skills_attempted = len(masteries)
    mastery_average = sum(masteries) / skills_attempted if masteries else 0.0
    full_coverage = skills_attempted == skills_total

    new_level = current_level
    current_index = LEVEL_ORDER.index(current_level)
    if mastery_average > ADVANCE_THRESHOLD and full_coverage and current_index < len(LEVEL_ORDER) - 1:
        new_level = LEVEL_ORDER[current_index + 1]
    elif mastery_average < REGRESS_THRESHOLD and current_index > 0:
        new_level = LEVEL_ORDER[current_index - 1]

    changed = new_level != current_level
    if changed:
        student.current_level = new_level
        session.add(student)
        session.commit()

    return {
        "student_id": student_id,
        "old_level": current_level,
        "new_level": new_level,
        "mastery_average": mastery_average,
        "skills_attempted": skills_attempted,
        "skills_total": skills_total,
        "changed": changed,
    }
