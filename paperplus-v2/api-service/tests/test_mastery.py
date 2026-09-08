import pytest
from sqlmodel import Session, delete, select

from app.db.session import engine
from app.domain.mastery import evaluate_and_update_level, recalculate_skill_mastery
from app.models import Attempt, Question, Skill, Student, StudentSkillMastery, Submission, Worksheet
from app.models.mastery import MasteryHistory


@pytest.fixture
def session():
    with Session(engine) as s:
        yield s


@pytest.fixture
def student_with_skills(session: Session):
    student = Student(student_id="8888", student_name="Mastery Test", current_level="A")
    skill = Skill(skill_code="MT1", skill_name="Mastery Test Skill", skill_level="1")
    worksheet = Worksheet(worksheet_level="A", worksheet_category="practice")
    session.add_all([student, skill, worksheet])
    session.commit()
    session.refresh(worksheet)

    question = Question(worksheet_id=worksheet.worksheet_id, skill_code="MT1", index=1)
    session.add(question)
    session.commit()
    session.refresh(question)

    submission = Submission(student_id="8888", worksheet_id=worksheet.worksheet_id, score=0, answers_json=[])
    session.add(submission)
    session.commit()
    session.refresh(submission)

    yield student, skill, worksheet, question, submission

    session.exec(delete(MasteryHistory).where(MasteryHistory.student_id == "8888"))
    session.exec(delete(StudentSkillMastery).where(StudentSkillMastery.student_id == "8888"))
    session.exec(delete(Attempt).where(Attempt.student_id == "8888"))
    session.exec(delete(Submission).where(Submission.submission_id == submission.submission_id))
    session.exec(delete(Question).where(Question.question_id == question.question_id))
    session.exec(delete(Worksheet).where(Worksheet.worksheet_id == worksheet.worksheet_id))
    session.exec(delete(Student).where(Student.student_id == "8888"))
    session.exec(delete(Skill).where(Skill.skill_code == "MT1"))
    session.commit()


def _add_attempt(session: Session, student_id: str, skill_code: str, is_correct: bool, question: Question, submission: Submission):
    session.add(
        Attempt(
            student_id=student_id,
            submission_id=submission.submission_id,
            question_id=question.question_id,
            worksheet_id=question.worksheet_id,
            is_correct=is_correct,
            skill_code=skill_code,
        )
    )
    session.commit()


def test_recalculate_skill_mastery_computes_ratio_and_writes_history(session: Session, student_with_skills):
    student, skill, _worksheet, question, submission = student_with_skills
    _add_attempt(session, student.student_id, skill.skill_code, True, question, submission)
    _add_attempt(session, student.student_id, skill.skill_code, False, question, submission)
    _add_attempt(session, student.student_id, skill.skill_code, True, question, submission)

    mastery_score = recalculate_skill_mastery(session, student.student_id, skill.skill_code)

    assert mastery_score == pytest.approx(2 / 3)
    row = session.get(StudentSkillMastery, (student.student_id, skill.skill_code))
    assert row.mastery_score == pytest.approx(2 / 3)
    history = session.exec(select(MasteryHistory).where(MasteryHistory.student_id == student.student_id)).all()
    assert len(history) == 1


def test_recalculate_skill_mastery_no_attempts_returns_none(session: Session, student_with_skills):
    student, skill, *_ = student_with_skills
    assert recalculate_skill_mastery(session, student.student_id, skill.skill_code) is None


def test_evaluate_and_update_level_advances_on_high_mastery(session: Session, student_with_skills):
    student, skill, _worksheet, question, submission = student_with_skills
    for _ in range(4):
        _add_attempt(session, student.student_id, skill.skill_code, True, question, submission)
    recalculate_skill_mastery(session, student.student_id, skill.skill_code)

    # Advancing requires full coverage of every tier-"1" skill, not just the fixture's own
    # "MT1" -- the real seeded catalog (scripts/seed_skills.py) also has tier-"1" skills, so
    # give those high mastery too. These extra rows belong to this test only and are cleaned
    # up below; the underlying Skill catalog rows are left untouched.
    other_tier1_codes = session.exec(
        select(Skill.skill_code).where(Skill.skill_level == "1", Skill.skill_code != skill.skill_code)
    ).all()
    for code in other_tier1_codes:
        session.add(StudentSkillMastery(student_id=student.student_id, skill_code=code, mastery_score=1.0))
    session.commit()

    result = evaluate_and_update_level(session, student.student_id)

    assert result["changed"] is True
    assert result["new_level"] == "B"
    session.refresh(student)
    assert student.current_level == "B"

    for code in other_tier1_codes:
        session.exec(
            delete(StudentSkillMastery).where(StudentSkillMastery.student_id == student.student_id, StudentSkillMastery.skill_code == code)
        )
    session.commit()


def test_evaluate_and_update_level_regresses_on_low_mastery(session: Session, student_with_skills):
    # Regressing FROM level "B" requires a skill in tier "2" (LEVEL_TO_DIFFICULTY["B"]),
    # unlike the shared fixture's tier-"1" skill used for the "A" advancement test above.
    student, _skill, worksheet, _question, submission = student_with_skills
    student.current_level = "B"
    session.add(student)
    session.commit()

    tier2_skill = Skill(skill_code="MT2", skill_name="Tier 2 skill", skill_level="2")
    session.add(tier2_skill)
    session.commit()
    question = Question(worksheet_id=worksheet.worksheet_id, skill_code="MT2", index=2)
    session.add(question)
    session.commit()
    session.refresh(question)

    for _ in range(4):
        _add_attempt(session, student.student_id, "MT2", False, question, submission)
    recalculate_skill_mastery(session, student.student_id, "MT2")

    result = evaluate_and_update_level(session, student.student_id)

    assert result["changed"] is True
    assert result["new_level"] == "A"

    session.exec(delete(MasteryHistory).where(MasteryHistory.student_id == student.student_id, MasteryHistory.skill_code == "MT2"))
    session.exec(delete(StudentSkillMastery).where(StudentSkillMastery.student_id == student.student_id, StudentSkillMastery.skill_code == "MT2"))
    session.exec(delete(Attempt).where(Attempt.student_id == student.student_id, Attempt.skill_code == "MT2"))
    session.exec(delete(Question).where(Question.question_id == question.question_id))
    session.exec(delete(Skill).where(Skill.skill_code == "MT2"))
    session.commit()

