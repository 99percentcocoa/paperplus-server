import pytest
from sqlmodel import Session, delete, select

from app.db.session import engine
from app.domain.errors import InvalidSubmissionDataError
from app.domain.worksheet_import import (
    infer_worksheet_category,
    insert_question_paper_variant,
    insert_worksheet,
    next_student_id,
    parse_generated_filename,
    seed_skills,
    upsert_school,
    upsert_student,
    worksheet_exists,
)
from app.models import Question, QuestionOption, School, Skill, Student, Worksheet
from app.models.worksheet import QuestionPaperVariant


@pytest.fixture
def session():
    with Session(engine) as s:
        yield s


def _cleanup_worksheet(session: Session, worksheet_id: int) -> None:
    session.exec(delete(QuestionPaperVariant).where(QuestionPaperVariant.worksheet_id == worksheet_id))
    for question in session.exec(select(Question).where(Question.worksheet_id == worksheet_id)).all():
        session.exec(delete(QuestionOption).where(QuestionOption.question_id == question.question_id))
    session.exec(delete(Question).where(Question.worksheet_id == worksheet_id))
    session.exec(delete(Worksheet).where(Worksheet.worksheet_id == worksheet_id))
    session.commit()


WORKSHEET_JSON = {
    "title": "Test Worksheet",
    "level": "A",
    "language": "en",
    "worksheet_category": "homework",
    "questions": [
        {
            "index": 1,
            "question_text": "2 + 2",
            "skill_code": "WI1",
            "options": ["3", "4", "5", "6"],
            "correct_option": "B",
        },
        {
            "index": 2,
            "question_text": "5 - 2",
            "skill_code": "WI1",
            "options": ["2", "3", "4", "5"],
            "correct_option": "B",
        },
    ],
}


def test_parse_generated_filename_id_and_language():
    details = parse_generated_filename("1000_mr.json")
    assert details["worksheet_id"] == 1000
    assert details["language"] == "mr"


def test_parse_generated_filename_rejects_unsupported_name():
    with pytest.raises(ValueError):
        parse_generated_filename("not-a-worksheet.txt")


def test_infer_worksheet_category_prefers_payload_value():
    assert infer_worksheet_category({"worksheet_category": "homework"}) == "homework"


def test_infer_worksheet_category_falls_back_to_filename():
    assert infer_worksheet_category({}, filename="8001_en_practice.json") == "practice"


def test_infer_worksheet_category_defaults_to_practice():
    assert infer_worksheet_category({}) == "practice"


def test_insert_worksheet_requires_questions(session: Session):
    with pytest.raises(InvalidSubmissionDataError):
        insert_worksheet(session, {"questions": []})


def test_insert_worksheet_auto_id_creates_questions_and_options(session: Session):
    result = insert_worksheet(session, WORKSHEET_JSON, worksheet_category="homework")
    worksheet_id = result["worksheet_id"]
    try:
        assert worksheet_id is not None
        assert len(result["question_ids"]) == 2
        assert worksheet_exists(session, worksheet_id)

        worksheet = session.get(Worksheet, worksheet_id)
        assert worksheet.worksheet_category == "homework"
        assert worksheet.max_score == 2
        assert worksheet.lang == "en"

        first_question = session.get(Question, result["question_ids"][0])
        assert first_question.index == 1
        assert first_question.skill_code == "WI1"

        options = session.exec(select(QuestionOption).where(QuestionOption.question_id == first_question.question_id)).all()
        assert {o.option_label for o in options} == {"A", "B", "C", "D"}
        correct = [o for o in options if o.is_correct]
        assert len(correct) == 1
        assert correct[0].option_label == "B"
    finally:
        _cleanup_worksheet(session, worksheet_id)
        session.exec(delete(Skill).where(Skill.skill_code == "WI1"))
        session.commit()


def test_insert_worksheet_explicit_id_is_respected(session: Session):
    explicit_id = 900123
    result = insert_worksheet(session, WORKSHEET_JSON, worksheet_id=explicit_id, worksheet_category="homework")
    try:
        assert result["worksheet_id"] == explicit_id
    finally:
        _cleanup_worksheet(session, explicit_id)
        session.exec(delete(Skill).where(Skill.skill_code == "WI1"))
        session.commit()


def test_insert_worksheet_infers_category_when_omitted(session: Session):
    payload = dict(WORKSHEET_JSON)
    payload["worksheet_category"] = "practice"
    result = insert_worksheet(session, payload)
    worksheet_id = result["worksheet_id"]
    try:
        worksheet = session.get(Worksheet, worksheet_id)
        assert worksheet.worksheet_category == "practice"
    finally:
        _cleanup_worksheet(session, worksheet_id)
        session.exec(delete(Skill).where(Skill.skill_code == "WI1"))
        session.commit()


def test_insert_question_paper_variant_overrides_canonical_answer(session: Session):
    result = insert_worksheet(session, WORKSHEET_JSON, worksheet_category="omr")
    worksheet_id = result["worksheet_id"]
    try:
        variant_ids = insert_question_paper_variant(session, worksheet_id, "b", ["C", "D"])
        assert len(variant_ids) == 2

        variants = session.exec(
            select(QuestionPaperVariant).where(
                QuestionPaperVariant.worksheet_id == worksheet_id, QuestionPaperVariant.question_paper_code == "B"
            )
        ).all()
        by_label = sorted(v.correct_option_label for v in variants)
        assert by_label == ["C", "D"]
    finally:
        session.exec(delete(QuestionPaperVariant).where(QuestionPaperVariant.worksheet_id == worksheet_id))
        _cleanup_worksheet(session, worksheet_id)
        session.exec(delete(Skill).where(Skill.skill_code == "WI1"))
        session.commit()


def test_insert_question_paper_variant_rejects_invalid_code(session: Session):
    result = insert_worksheet(session, WORKSHEET_JSON, worksheet_category="omr")
    worksheet_id = result["worksheet_id"]
    try:
        with pytest.raises(ValueError):
            insert_question_paper_variant(session, worksheet_id, "Z", ["A", "B"])
    finally:
        _cleanup_worksheet(session, worksheet_id)
        session.exec(delete(Skill).where(Skill.skill_code == "WI1"))
        session.commit()


def test_insert_question_paper_variant_rejects_length_mismatch(session: Session):
    result = insert_worksheet(session, WORKSHEET_JSON, worksheet_category="omr")
    worksheet_id = result["worksheet_id"]
    try:
        with pytest.raises(InvalidSubmissionDataError):
            insert_question_paper_variant(session, worksheet_id, "A", ["A"])
    finally:
        _cleanup_worksheet(session, worksheet_id)
        session.exec(delete(Skill).where(Skill.skill_code == "WI1"))
        session.commit()


def test_seed_skills_is_idempotent_and_covers_known_codes(session: Session, tmp_path):
    skills_file = tmp_path / "skills.json"
    skills_file.write_text(
        '[{"code": "SEEDTEST", "skill": "Seed Test Skill", "difficulty_level": "1"}]',
        encoding="utf-8",
    )
    try:
        count = seed_skills(session, skills_path=skills_file)
        assert count == 1
        skill = session.get(Skill, "SEEDTEST")
        assert skill.skill_name == "Seed Test Skill"

        # Re-seeding with an updated name updates the existing row rather than duplicating it.
        skills_file.write_text(
            '[{"code": "SEEDTEST", "skill": "Updated Name", "difficulty_level": "2"}]',
            encoding="utf-8",
        )
        seed_skills(session, skills_path=skills_file)
        session.refresh(skill)
        assert skill.skill_name == "Updated Name"
        assert skill.skill_level == "2"
    finally:
        session.exec(delete(Skill).where(Skill.skill_code == "SEEDTEST"))
        session.commit()


def test_upsert_school_and_student_and_next_id(session: Session):
    school = upsert_school(session, "WITEST", "Worksheet Import Test School")
    session.commit()
    try:
        assert school.school_code == "WITEST"

        first_id = next_student_id(session)
        student = upsert_student(session, first_id, "Test Student One", "WITEST")
        session.commit()

        second_id = next_student_id(session)
        assert int(second_id) == int(first_id) + 1

        # upsert is idempotent: re-inserting the same id returns the existing row, not a duplicate.
        same_student = upsert_student(session, first_id, "Ignored Name", "WITEST")
        session.commit()
        assert same_student.student_id == student.student_id
        assert same_student.student_name == "Test Student One"
    finally:
        session.exec(delete(Student).where(Student.student_school_code == "WITEST"))
        session.exec(delete(School).where(School.school_code == "WITEST"))
        session.commit()
