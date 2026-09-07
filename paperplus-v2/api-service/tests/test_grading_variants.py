import pytest
from sqlmodel import Session, delete

from app.db.session import engine
from app.domain.grading import resolve_answer_key
from app.models import Question, QuestionOption, Skill, Worksheet
from app.models.worksheet import QuestionPaperVariant


@pytest.fixture
def session():
    with Session(engine) as s:
        yield s


@pytest.fixture
def worksheet_with_variants(session: Session):
    skill = Skill(skill_code="GV1", skill_name="Grading Variant Test Skill", skill_level="1")
    worksheet = Worksheet(worksheet_level="A", worksheet_category="omr")
    session.add_all([skill, worksheet])
    session.commit()
    session.refresh(worksheet)

    question = Question(worksheet_id=worksheet.worksheet_id, skill_code="GV1", index=1)
    session.add(question)
    session.commit()
    session.refresh(question)

    # Canonical answer: "A" is correct.
    session.add(QuestionOption(question_id=question.question_id, option_label="A", option_value="2", is_correct=True))
    session.add(QuestionOption(question_id=question.question_id, option_label="B", option_value="3", is_correct=False))
    # Variant "B" of this worksheet has a different fixed correct answer: "C".
    session.add(
        QuestionPaperVariant(
            worksheet_id=worksheet.worksheet_id, question_paper_code="B", question_id=question.question_id, correct_option_label="C"
        )
    )
    session.commit()

    yield worksheet, question

    session.exec(delete(QuestionPaperVariant).where(QuestionPaperVariant.worksheet_id == worksheet.worksheet_id))
    session.exec(delete(QuestionOption).where(QuestionOption.question_id == question.question_id))
    session.exec(delete(Question).where(Question.question_id == question.question_id))
    session.exec(delete(Worksheet).where(Worksheet.worksheet_id == worksheet.worksheet_id))
    session.exec(delete(Skill).where(Skill.skill_code == "GV1"))
    session.commit()


def test_resolve_answer_key_uses_canonical_when_no_variant_code_given(session: Session, worksheet_with_variants):
    worksheet, question = worksheet_with_variants
    answer_key = resolve_answer_key(session, worksheet.worksheet_id)
    assert answer_key[question.index] == "A"


def test_resolve_answer_key_uses_canonical_when_variant_not_seeded(session: Session, worksheet_with_variants):
    worksheet, question = worksheet_with_variants
    # "D" has no QuestionPaperVariant row seeded -> falls back to canonical.
    answer_key = resolve_answer_key(session, worksheet.worksheet_id, question_paper_code="D")
    assert answer_key[question.index] == "A"


def test_resolve_answer_key_prefers_seeded_variant_answer(session: Session, worksheet_with_variants):
    worksheet, question = worksheet_with_variants
    answer_key = resolve_answer_key(session, worksheet.worksheet_id, question_paper_code="B")
    assert answer_key[question.index] == "C"
