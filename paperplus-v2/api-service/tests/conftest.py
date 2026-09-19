"""Shared test hygiene and fixtures. Tests here run against the real dev DB, and
handle_incoming_image now records a Scan row for every image it receives -- so any test that
drives it would otherwise leak scans (scans have no FK to worksheets/students, so fixture
teardowns never remove them).
"""

import itertools
from pathlib import Path

import pytest
from PIL import Image
from sqlmodel import Session, delete, select

from app.core.config import settings
from app.db.session import engine
from app.models import (
    Attempt,
    ProcessingEvent,
    Question,
    QuestionOption,
    School,
    Scan,
    ScanReview,
    Skill,
    Student,
    StudentSkillMastery,
    Submission,
    Worksheet,
)
from app.models.mastery import MasteryHistory
from app.services.submission_service import handle_incoming_image
from shared.contracts import ProcessingResult, QuestionMark
from tests.fakes import FakeCommunicationClient, FakeVisionClient

# WhatsApp numbers and correlation-id prefixes used only by tests.
TEST_FROM_NUMBERS = ("+911234567890", "+911111111111", "+912222222222", "+910000000099")

ENV_STUDENT_ID = "9995"
ENV_SCHOOL_CODE = "CRT"
ENV_FROM_NUMBER = "+910000000099"


@pytest.fixture(autouse=True)
def _cleanup_test_scans():
    yield
    with Session(engine) as session:
        # Reviews first, matched by test correlation-id prefixes: a vision-failure review has no
        # student/worksheet to match on, and once its scan is deleted it can't be found via scan_id.
        session.exec(delete(ScanReview).where(
            ScanReview.correlation_id.like("corr-%") | ScanReview.correlation_id.like("crt-%")
        ))
        session.exec(delete(Scan).where(Scan.from_number.in_(TEST_FROM_NUMBERS)))
        session.exec(delete(Scan).where(Scan.correlation_id.like("corr-%")))
        session.commit()


class ScanEnv:
    """A 4-question worksheet (correct answer A everywhere), a registered student, and a helper
    that pushes a fake vision result through the real handle_incoming_image."""

    def __init__(self, session: Session, worksheet: Worksheet, student: Student, storage: Path):
        self.session = session
        self.worksheet = worksheet
        self.student = student
        self.storage = storage
        self.comm = FakeCommunicationClient()
        self._counter = itertools.count(1)
        self.extra_student_ids: list[str] = []

    def add_student(self, student_id: str) -> Student:
        """Register another student mid-test (torn down with the fixture)."""
        student = Student(student_id=student_id, student_name=f"Extra {student_id}", student_school_code=ENV_SCHOOL_CODE)
        self.session.add(student)
        self.session.commit()
        self.extra_student_ids.append(student_id)
        return student

    def run_scan(self, marks: dict[int, str | None], roll_number: str | None = None, page_no: int = 1) -> str:
        """marks: {question_index: marked option or None}. Returns the correlation_id used."""
        n = next(self._counter)
        correlation_id = f"crt-{self.worksheet.worksheet_id}-{n}"
        dewarped = self.storage / "dewarped" / f"{correlation_id}_dewarped.jpg"
        upload = self.storage / "uploads" / f"{correlation_id}.jpg"
        for path in (dewarped, upload):
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (700, 500), color="white").save(path)
        result = ProcessingResult(
            worksheet_id=self.worksheet.worksheet_id,
            page_no=page_no,
            first_question_index=1,
            template_name="regular",
            roll_number=roll_number or self.student.student_id,
            roll_number_confidence=None,
            question_paper_code="",
            question_marks=[
                QuestionMark(
                    question_index=idx, marked_option=option, confidence=0.9,
                    roi_x1=10, roi_y1=10 + 60 * (idx - 1), roi_x2=300, roi_y2=60 + 60 * (idx - 1),
                )
                for idx, option in sorted(marks.items())
            ],
            dewarped_image_path=str(dewarped),
        )
        handle_incoming_image(
            self.session, FakeVisionClient(result), self.comm, ENV_FROM_NUMBER, str(upload), correlation_id
        )
        return correlation_id


@pytest.fixture
def session():
    with Session(engine) as s:
        yield s


@pytest.fixture
def scan_env(session: Session, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_root", str(tmp_path))

    skill_preexisted = session.get(Skill, "1A") is not None
    skill = session.get(Skill, "1A") or Skill(skill_code="1A", skill_name="1-digit addition", skill_level="1")
    school = School(school_code=ENV_SCHOOL_CODE, school_name="Corrections Test School")
    student = Student(student_id=ENV_STUDENT_ID, student_name="Corrections Test", student_school_code=ENV_SCHOOL_CODE)
    worksheet = Worksheet(worksheet_level="A", worksheet_category="practice", lang="en", title="Corrections worksheet")
    session.add_all([skill, school, student, worksheet])
    session.commit()
    session.refresh(worksheet)

    for index in (1, 2, 3, 4):
        question = Question(
            worksheet_id=worksheet.worksheet_id, skill_code="1A", index=index,
            question_json={"question_text": f"Q{index}", "options": ["a", "b", "c", "d"]},
        )
        session.add(question)
        session.commit()
        session.refresh(question)
        for label in "ABCD":
            session.add(QuestionOption(
                question_id=question.question_id, option_label=label, option_value=label.lower(), is_correct=(label == "A"),
            ))
    session.commit()

    env = ScanEnv(session, worksheet, student, tmp_path)
    yield env

    session.rollback()
    worksheet_id = worksheet.worksheet_id
    submission_ids = [s.submission_id for s in session.exec(select(Submission).where(Submission.worksheet_id == worksheet_id)).all()]
    session.exec(delete(ScanReview).where(
        (ScanReview.worksheet_id == worksheet_id)
        | (ScanReview.student_id.in_([ENV_STUDENT_ID, *env.extra_student_ids]))
        | (ScanReview.detected_roll_number.in_([ENV_STUDENT_ID, "0000", *env.extra_student_ids]))
    ))
    scan_ids = [x.id for x in session.exec(select(Scan).where(Scan.from_number == ENV_FROM_NUMBER)).all()]
    if scan_ids:  # e.g. vision-failure reviews, which have no student/worksheet to match on
        session.exec(delete(ScanReview).where(ScanReview.scan_id.in_(scan_ids)))
    session.exec(delete(Scan).where(Scan.from_number == ENV_FROM_NUMBER))
    if submission_ids:
        session.exec(delete(ProcessingEvent).where(ProcessingEvent.submission_id.in_(submission_ids)))
    session.exec(delete(Attempt).where(Attempt.worksheet_id == worksheet_id))
    session.exec(delete(Submission).where(Submission.worksheet_id == worksheet_id))
    all_student_ids = [ENV_STUDENT_ID, *env.extra_student_ids]
    session.exec(delete(MasteryHistory).where(MasteryHistory.student_id.in_(all_student_ids)))
    session.exec(delete(StudentSkillMastery).where(StudentSkillMastery.student_id.in_(all_student_ids)))
    for question in session.exec(select(Question).where(Question.worksheet_id == worksheet_id)).all():
        session.exec(delete(QuestionOption).where(QuestionOption.question_id == question.question_id))
    session.exec(delete(Question).where(Question.worksheet_id == worksheet_id))
    session.exec(delete(Worksheet).where(Worksheet.worksheet_id == worksheet_id))
    session.exec(delete(Student).where(Student.student_id.in_(all_student_ids)))
    session.exec(delete(School).where(School.school_code == ENV_SCHOOL_CODE))
    if not skill_preexisted:
        session.exec(delete(Skill).where(Skill.skill_code == "1A"))
    session.commit()
