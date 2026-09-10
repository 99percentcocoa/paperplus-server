"""Full integration test: webhook -> grading -> state machine -> mastery, against the real
paperplus_v2 Postgres DB (fixtures create+clean up their own rows). VisionClient and
CommunicationClient are faked so no real HTTP/Exotel calls happen.
"""

import pytest
from sqlmodel import Session, delete, select

from app.db.session import engine
from app.models import (
    Attempt,
    ProcessingEvent,
    Question,
    QuestionOption,
    School,
    ScanReview,
    Skill,
    Student,
    StudentSkillMastery,
    Submission,
    Worksheet,
)
from app.models.mastery import MasteryHistory
from app.models.submission import ProcessingState
from app.models.worksheet import WorksheetPage
from app.services.submission_service import handle_incoming_image
from shared.contracts import ProcessingResult, QuestionMark
from tests.fakes import FailingFakeVisionClient, FakeCommunicationClient, FakeVisionClient


@pytest.fixture
def session():
    with Session(engine) as s:
        yield s


@pytest.fixture
def worksheet_with_questions(session: Session):
    school = School(school_code="TST", school_name="Test School")
    student = Student(student_id="9999", student_name="Test Student", student_school_code="TST", current_level="A")
    # "1A" is also a real seeded skill (scripts/seed_skills.py) -- reuse it if present rather
    # than inserting a duplicate PK, and don't delete it on teardown if we didn't create it.
    skill_preexisted = session.get(Skill, "1A") is not None
    skill = session.get(Skill, "1A") or Skill(skill_code="1A", skill_name="1-digit addition", skill_level="1")
    worksheet = Worksheet(worksheet_level="A", worksheet_category="practice", lang="en")
    session.add_all([school, student, skill, worksheet])
    session.commit()
    session.refresh(worksheet)

    questions = []
    for index in (1, 2):
        question = Question(worksheet_id=worksheet.worksheet_id, skill_code="1A", index=index)
        session.add(question)
        session.commit()
        session.refresh(question)
        session.add(QuestionOption(question_id=question.question_id, option_label="A", option_value="2", is_correct=True))
        session.add(QuestionOption(question_id=question.question_id, option_label="B", option_value="3", is_correct=False))
        questions.append(question)
    session.commit()

    yield worksheet, student

    # Cleanup (reverse FK order)
    session.exec(delete(ProcessingEvent).where(ProcessingEvent.submission_id.in_(
        select_submission_ids(session, worksheet.worksheet_id)
    )))
    session.exec(delete(ScanReview).where(ScanReview.worksheet_id == worksheet.worksheet_id))
    session.exec(delete(Attempt).where(Attempt.worksheet_id == worksheet.worksheet_id))
    session.exec(delete(Submission).where(Submission.worksheet_id == worksheet.worksheet_id))
    session.exec(delete(MasteryHistory).where(MasteryHistory.student_id == "9999"))
    session.exec(delete(StudentSkillMastery).where(StudentSkillMastery.student_id == "9999"))
    for q in questions:
        session.exec(delete(QuestionOption).where(QuestionOption.question_id == q.question_id))
    session.exec(delete(Question).where(Question.worksheet_id == worksheet.worksheet_id))
    session.exec(delete(Worksheet).where(Worksheet.worksheet_id == worksheet.worksheet_id))
    session.exec(delete(Student).where(Student.student_id == "9999"))
    session.exec(delete(School).where(School.school_code == "TST"))
    if not skill_preexisted:
        session.exec(delete(Skill).where(Skill.skill_code == "1A"))
    session.commit()


def select_submission_ids(session: Session, worksheet_id: int) -> list[int]:
    from sqlmodel import select

    return [s.submission_id for s in session.exec(select(Submission).where(Submission.worksheet_id == worksheet_id)).all()]


def test_handle_incoming_image_grades_and_transitions_to_graded(session: Session, worksheet_with_questions):
    worksheet, student = worksheet_with_questions

    result = ProcessingResult(
        worksheet_id=worksheet.worksheet_id,
        page_no=1,
        first_question_index=1,
        template_name="regular",
        roll_number=student.student_id,
        roll_number_confidence=None,
        question_paper_code="",
        question_marks=[
            QuestionMark(question_index=1, marked_option="A", confidence=0.9),  # correct
            QuestionMark(question_index=2, marked_option="B", confidence=0.9),  # incorrect
        ],
    )

    vision_client = FakeVisionClient(result)
    comm_client = FakeCommunicationClient()

    handle_incoming_image(session, vision_client, comm_client, "+911234567890", "/fake/path.jpg", "corr-1")

    submission = select_submission_ids(session, worksheet.worksheet_id)
    assert len(submission) == 1

    from sqlmodel import select

    saved = session.exec(select(Submission).where(Submission.submission_id == submission[0])).first()
    assert saved.score == 1
    assert saved.state == ProcessingState.GRADED.value

    attempts = session.exec(select(Attempt).where(Attempt.submission_id == submission[0])).all()
    assert len(attempts) == 2
    assert sum(1 for a in attempts if a.is_correct) == 1

    mastery = session.get(StudentSkillMastery, (student.student_id, "1A"))
    assert mastery is not None
    assert mastery.mastery_score == 0.5

    assert len(comm_client.sent_messages) == 1
    assert "Your marks: 1/2" in comm_client.sent_messages[0][1]


def test_handle_incoming_image_draws_and_sends_checked_image(session: Session, worksheet_with_questions, tmp_path, monkeypatch):
    """When vision-service returns a dewarped_image_path, grading must annotate it (✔/✘ per
    question ROI box) and send it back over WhatsApp via send_image, storing the resulting
    path/url on the Submission row.
    """
    from PIL import Image

    from app.core import config as config_module

    monkeypatch.setattr(config_module.settings, "storage_root", str(tmp_path))

    worksheet, student = worksheet_with_questions

    dewarped_path = tmp_path / "dewarped.jpg"
    Image.new("RGB", (400, 300), color="white").save(dewarped_path)

    result = ProcessingResult(
        worksheet_id=worksheet.worksheet_id,
        page_no=1,
        first_question_index=1,
        template_name="regular",
        roll_number=student.student_id,
        roll_number_confidence=None,
        question_paper_code="",
        question_marks=[
            QuestionMark(question_index=1, marked_option="A", confidence=0.9, roi_x1=10, roi_y1=10, roi_x2=100, roi_y2=60),
            QuestionMark(question_index=2, marked_option="B", confidence=0.9, roi_x1=10, roi_y1=70, roi_x2=100, roi_y2=120),
        ],
        dewarped_image_path=str(dewarped_path),
    )

    comm_client = FakeCommunicationClient()
    handle_incoming_image(session, FakeVisionClient(result), comm_client, "+911234567890", "/fake/path.jpg", "corr-checked-1")

    assert len(comm_client.sent_images) == 1
    to_number, image_url, _caption = comm_client.sent_images[0]
    assert to_number == "+911234567890"
    assert image_url.endswith("/files/checked/corr-checked-1_checked.jpg")

    submission_id = select_submission_ids(session, worksheet.worksheet_id)[0]
    saved = session.exec(select(Submission).where(Submission.submission_id == submission_id)).first()
    assert saved.checked_image_url == image_url
    assert saved.checked_image_path is not None
    from pathlib import Path
    assert Path(saved.checked_image_path).is_file()


def test_handle_incoming_image_reports_unrecognized_student(session: Session, worksheet_with_questions):
    worksheet, _student = worksheet_with_questions

    result = ProcessingResult(
        worksheet_id=worksheet.worksheet_id,
        page_no=1,
        first_question_index=1,
        template_name="regular",
        roll_number="0000",  # not a registered student
        roll_number_confidence=None,
        question_paper_code="",
        question_marks=[],
    )

    comm_client = FakeCommunicationClient()
    handle_incoming_image(session, FakeVisionClient(result), comm_client, "+911234567890", "/fake/path.jpg", "corr-2")

    assert len(comm_client.sent_messages) == 1
    assert "Roll number not recognized" in comm_client.sent_messages[0][1]
    assert select_submission_ids(session, worksheet.worksheet_id) == []

    # InvalidStudentError also writes a ScanReview (student_id/worksheet_id both None here,
    # since the roll number never resolved to a real student) -- clean it up rather than
    # leaking an orphan row into the shared dev DB on every test run.
    session.exec(delete(ScanReview).where(ScanReview.detected_roll_number == "0000"))
    session.commit()


def test_handle_incoming_image_refuses_to_grade_with_no_answer_key(session: Session, worksheet_with_questions):
    """A worksheet whose questions have no canonical is_correct option and no seeded
    question_paper_variant for the scanned code must not silently grade everything wrong.
    """
    worksheet, student = worksheet_with_questions

    # Strip the fixture's canonical answer key so resolve_answer_key finds nothing, simulating
    # an OMR worksheet whose question-paper-code variant was never seeded.
    for option in session.exec(select(QuestionOption).where(QuestionOption.question_id.in_(
        select(Question.question_id).where(Question.worksheet_id == worksheet.worksheet_id)
    ))).all():
        option.is_correct = False
        session.add(option)
    session.commit()

    result = ProcessingResult(
        worksheet_id=worksheet.worksheet_id,
        page_no=1,
        first_question_index=1,
        template_name="basic_omr",
        roll_number=student.student_id,
        roll_number_confidence=None,
        question_paper_code="A",  # no question_paper_variant seeded for "A"
        question_marks=[
            QuestionMark(question_index=1, marked_option="A", confidence=0.9),
            QuestionMark(question_index=2, marked_option="B", confidence=0.9),
        ],
    )

    comm_client = FakeCommunicationClient()
    handle_incoming_image(session, FakeVisionClient(result), comm_client, "+911234567890", "/fake/path.jpg", "corr-3")

    assert len(comm_client.sent_messages) == 1
    assert "not ready to be graded" in comm_client.sent_messages[0][1]
    assert select_submission_ids(session, worksheet.worksheet_id) == []

    review = session.exec(
        select(ScanReview).where(ScanReview.worksheet_id == worksheet.worksheet_id)
    ).first()
    assert review is not None
    assert review.status == "needs_review"  # content-seeding gap, not a bad scan
    assert review.student_id == student.student_id
    session.exec(delete(ScanReview).where(ScanReview.review_id == review.review_id))
    session.commit()


def test_handle_incoming_image_records_failed_scan_review_on_vision_error(session: Session):
    comm_client = FakeCommunicationClient()
    handle_incoming_image(session, FailingFakeVisionClient(), comm_client, "+911234567890", "/fake/path.jpg", "corr-vision-fail")

    assert len(comm_client.sent_messages) == 1
    review = session.exec(
        select(ScanReview).where(ScanReview.error_reason.contains("connection refused"))
    ).first()
    assert review is not None
    assert review.status == "failed"
    assert review.student_id is None
    assert review.worksheet_id is None
    session.exec(delete(ScanReview).where(ScanReview.review_id == review.review_id))
    session.commit()


def test_handle_incoming_image_records_failed_scan_review_on_invalid_worksheet(session: Session, worksheet_with_questions):
    worksheet, student = worksheet_with_questions

    result = ProcessingResult(
        worksheet_id=999999999,  # does not exist
        page_no=1,
        first_question_index=1,
        template_name="regular",
        roll_number=student.student_id,
        roll_number_confidence=None,
        question_paper_code="",
        question_marks=[],
    )

    comm_client = FakeCommunicationClient()
    handle_incoming_image(session, FakeVisionClient(result), comm_client, "+911234567890", "/fake/path.jpg", "corr-bad-ws")

    review = session.exec(
        select(ScanReview).where(ScanReview.detected_roll_number == student.student_id)
    ).first()
    assert review is not None
    assert review.status == "failed"
    assert review.student_id == student.student_id  # roll number was valid, resolved before the worksheet lookup failed
    assert review.worksheet_id is None  # the scanned id doesn't reference a real worksheet, so it can't be set as the FK
    assert "999999999" in review.error_reason
    session.exec(delete(ScanReview).where(ScanReview.review_id == review.review_id))
    session.commit()


@pytest.fixture
def two_page_worksheet(session: Session):
    """A 2-page worksheet: page 1 = questions 1-2, page 2 = questions 3-4."""
    student = Student(student_id="9998", student_name="Two Page Test", current_level="A")
    skill_preexisted = session.get(Skill, "1A") is not None
    skill = session.get(Skill, "1A") or Skill(skill_code="1A", skill_name="1-digit addition", skill_level="1")
    worksheet = Worksheet(worksheet_level="A", worksheet_category="omr", lang="en", page_count=2)
    session.add_all([student, skill, worksheet])
    session.commit()
    session.refresh(worksheet)

    questions = []
    for index in (1, 2, 3, 4):
        question = Question(worksheet_id=worksheet.worksheet_id, skill_code="1A", index=index)
        session.add(question)
        session.commit()
        session.refresh(question)
        session.add(QuestionOption(question_id=question.question_id, option_label="A", option_value="2", is_correct=True))
        session.add(QuestionOption(question_id=question.question_id, option_label="B", option_value="3", is_correct=False))
        questions.append(question)
    session.add(WorksheetPage(worksheet_id=worksheet.worksheet_id, page_no=1, first_question_index=1, last_question_index=2))
    session.add(WorksheetPage(worksheet_id=worksheet.worksheet_id, page_no=2, first_question_index=3, last_question_index=4))
    session.commit()

    yield worksheet, student

    session.exec(delete(ProcessingEvent).where(ProcessingEvent.submission_id.in_(
        select_submission_ids(session, worksheet.worksheet_id)
    )))
    session.exec(delete(Attempt).where(Attempt.worksheet_id == worksheet.worksheet_id))
    session.exec(delete(Submission).where(Submission.worksheet_id == worksheet.worksheet_id))
    session.exec(delete(MasteryHistory).where(MasteryHistory.student_id == "9998"))
    session.exec(delete(StudentSkillMastery).where(StudentSkillMastery.student_id == "9998"))
    session.exec(delete(WorksheetPage).where(WorksheetPage.worksheet_id == worksheet.worksheet_id))
    for q in questions:
        session.exec(delete(QuestionOption).where(QuestionOption.question_id == q.question_id))
    session.exec(delete(Question).where(Question.worksheet_id == worksheet.worksheet_id))
    session.exec(delete(Worksheet).where(Worksheet.worksheet_id == worksheet.worksheet_id))
    session.exec(delete(Student).where(Student.student_id == "9998"))
    if not skill_preexisted:
        session.exec(delete(Skill).where(Skill.skill_code == "1A"))
    session.commit()


def test_multi_page_submission_merges_instead_of_overwriting(session: Session, two_page_worksheet):
    worksheet, student = two_page_worksheet
    comm_client = FakeCommunicationClient()

    page1 = ProcessingResult(
        worksheet_id=worksheet.worksheet_id, page_no=1, first_question_index=1, template_name="basic_omr",
        roll_number=student.student_id, roll_number_confidence=None, question_paper_code="",
        question_marks=[
            QuestionMark(question_index=1, marked_option="A", confidence=0.9),
            QuestionMark(question_index=2, marked_option="A", confidence=0.9),
        ],
    )
    handle_incoming_image(session, FakeVisionClient(page1), comm_client, "+911234567890", "/fake/p1.jpg", "corr-p1")

    submission_ids = select_submission_ids(session, worksheet.worksheet_id)
    assert len(submission_ids) == 1
    from sqlmodel import select
    after_page1 = session.exec(select(Submission).where(Submission.submission_id == submission_ids[0])).first()
    assert after_page1.score == 2  # both q1 and q2 marked "A", which is the correct option for both
    assert len(after_page1.answers_json) == 2

    page2 = ProcessingResult(
        worksheet_id=worksheet.worksheet_id, page_no=2, first_question_index=3, template_name="basic_omr",
        roll_number=student.student_id, roll_number_confidence=None, question_paper_code="",
        question_marks=[
            QuestionMark(question_index=3, marked_option="B", confidence=0.9),  # incorrect
            QuestionMark(question_index=4, marked_option="A", confidence=0.9),  # correct
        ],
    )
    handle_incoming_image(session, FakeVisionClient(page2), comm_client, "+911234567890", "/fake/p2.jpg", "corr-p2")

    # Still exactly one Submission row for this student+worksheet (merged, not a second row).
    submission_ids = select_submission_ids(session, worksheet.worksheet_id)
    assert len(submission_ids) == 1
    final = session.exec(select(Submission).where(Submission.submission_id == submission_ids[0])).first()

    assert [a["question_index"] for a in final.answers_json] == [1, 2, 3, 4]
    # page 1's answers must be untouched by the page-2 submission.
    assert final.answers_json[0] == after_page1.answers_json[0]
    assert final.answers_json[1] == after_page1.answers_json[1]
    assert final.score == 3  # q1 correct, q2 correct, q3 incorrect, q4 correct

    attempts = session.exec(select(Attempt).where(Attempt.submission_id == final.submission_id)).all()
    assert len(attempts) == 4
