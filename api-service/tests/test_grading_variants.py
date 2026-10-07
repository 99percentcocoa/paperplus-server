import pytest
from sqlmodel import Session, delete

from app.db.session import engine
from app.domain.grading import resolve_answer_key
from app.models import Question, QuestionOption, Skill, Worksheet
from app.models.worksheet import OMRAnswerSet, QuestionPaperVariant


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


@pytest.fixture
def worksheet_with_code_level_key(session: Session):
    """A second, independent worksheet sharing the SAME code-level answer set as any other
    worksheet using code 'E' -- proves the key is not tied to a specific worksheet_id.
    """
    skill = Skill(skill_code="GV2", skill_name="Grading Code-Level Test Skill", skill_level="1")
    worksheet_1 = Worksheet(worksheet_level="A", worksheet_category="omr")
    worksheet_2 = Worksheet(worksheet_level="A", worksheet_category="omr")
    session.add_all([skill, worksheet_1, worksheet_2])
    session.commit()
    session.refresh(worksheet_1)
    session.refresh(worksheet_2)

    question_1 = Question(worksheet_id=worksheet_1.worksheet_id, skill_code="GV2", index=1)
    question_2 = Question(worksheet_id=worksheet_2.worksheet_id, skill_code="GV2", index=1)
    session.add_all([question_1, question_2])
    session.commit()
    session.refresh(question_1)
    session.refresh(question_2)

    # Neither worksheet has a canonical is_correct option or a per-worksheet variant -- the
    # only source of truth is the code-level OMRAnswerSet row (worksheet_id=NULL).
    omr_set = OMRAnswerSet(
        template_name="basic_omr", question_paper_code="E", worksheet_id=None, answer_key_json={"1": "C"}
    )
    session.add(omr_set)
    session.commit()

    yield worksheet_1, worksheet_2, question_1, question_2

    session.exec(delete(OMRAnswerSet).where(OMRAnswerSet.question_paper_code == "E", OMRAnswerSet.worksheet_id.is_(None)))
    session.exec(delete(Question).where(Question.question_id.in_([question_1.question_id, question_2.question_id])))
    session.exec(delete(Worksheet).where(Worksheet.worksheet_id.in_([worksheet_1.worksheet_id, worksheet_2.worksheet_id])))
    session.exec(delete(Skill).where(Skill.skill_code == "GV2"))
    session.commit()


def test_resolve_answer_key_uses_code_level_key_across_different_worksheets(session: Session, worksheet_with_code_level_key):
    """The same question_paper_code answer key must resolve identically for two unrelated
    worksheet_ids, since basic_omr codes are worksheet-independent by design.
    """
    worksheet_1, worksheet_2, question_1, question_2 = worksheet_with_code_level_key

    answer_key_1 = resolve_answer_key(session, worksheet_1.worksheet_id, question_paper_code="E")
    answer_key_2 = resolve_answer_key(session, worksheet_2.worksheet_id, question_paper_code="E")

    assert answer_key_1[question_1.index] == "C"
    assert answer_key_2[question_2.index] == "C"


def test_resolve_answer_key_prefers_worksheet_variant_over_code_level_key(session: Session, worksheet_with_code_level_key):
    """A per-worksheet question_paper_variant override still wins over the shared code-level key."""
    worksheet_1, _worksheet_2, question_1, _question_2 = worksheet_with_code_level_key

    session.add(
        QuestionPaperVariant(
            worksheet_id=worksheet_1.worksheet_id, question_paper_code="E", question_id=question_1.question_id, correct_option_label="Z"
        )
    )
    session.commit()

    answer_key = resolve_answer_key(session, worksheet_1.worksheet_id, question_paper_code="E")
    assert answer_key[question_1.index] == "Z"

    session.exec(delete(QuestionPaperVariant).where(QuestionPaperVariant.worksheet_id == worksheet_1.worksheet_id))
    session.commit()
