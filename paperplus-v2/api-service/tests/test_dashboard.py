"""Tests for app/routes/dashboard.py's DB-derived /metrics endpoint (Phase 5). Runs against the
real dev DB like other tests here, so assertions use before/after deltas rather than absolute
counts -- other data may already exist in the shared DB.
"""

import pytest
from sqlmodel import Session, delete, select

from app.db.session import engine
from app.models import School, ScanReview, Skill, Student, Submission, Worksheet
from app.models.submission import ProcessingState, ScanReviewStatus
from app.routes.dashboard import metrics


@pytest.fixture
def session():
    with Session(engine) as s:
        yield s


def test_metrics_counts_graded_submissions_and_scan_reviews(session: Session):
    skill_preexisted = session.get(Skill, "1A") is not None
    skill = session.get(Skill, "1A") or Skill(skill_code="1A", skill_name="1-digit addition", skill_level="1")
    school = School(school_code="MTEST", school_name="Metrics Test School")
    student = Student(student_id="9997", student_name="Metrics Test", student_school_code="MTEST", current_level="A")
    worksheet = Worksheet(worksheet_level="A", worksheet_category="practice", lang="en")
    session.add_all([skill, school, student, worksheet])
    session.commit()
    session.refresh(worksheet)

    before = metrics(session)

    graded_submission = Submission(
        student_id=student.student_id,
        worksheet_id=worksheet.worksheet_id,
        worksheet_category="practice",
        score=1,
        state=ProcessingState.GRADED.value,
    )
    review = ScanReview(
        student_id=student.student_id,
        worksheet_id=worksheet.worksheet_id,
        status=ScanReviewStatus.NEEDS_REVIEW.value,
        error_reason="test review",
        correlation_id="corr-metrics-test",
    )
    session.add_all([graded_submission, review])
    session.commit()

    after = metrics(session)

    assert after["submissions"]["total"] == before["submissions"]["total"] + 1
    assert after["submissions"]["graded"] == before["submissions"]["graded"] + 1
    assert after["scan_reviews"]["total"] == before["scan_reviews"]["total"] + 1
    assert after["scan_reviews"]["by_status"]["needs_review"] == before["scan_reviews"]["by_status"].get("needs_review", 0) + 1
    assert "generated_at" in after

    session.exec(delete(ScanReview).where(ScanReview.correlation_id == "corr-metrics-test"))
    session.exec(delete(Submission).where(Submission.submission_id == graded_submission.submission_id))
    session.exec(delete(Student).where(Student.student_id == "9997"))
    session.exec(delete(Worksheet).where(Worksheet.worksheet_id == worksheet.worksheet_id))
    session.exec(delete(School).where(School.school_code == "MTEST"))
    if not skill_preexisted:
        session.exec(delete(Skill).where(Skill.skill_code == "1A"))
    session.commit()
