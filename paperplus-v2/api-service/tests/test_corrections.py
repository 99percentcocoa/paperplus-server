"""Corrections service: fixing answers on a graded submission, and turning a failed scan into a
submission once an admin picks the student. Runs real scans through handle_incoming_image (fake
vision/comm clients) against the real dev DB, so what's corrected is exactly what production creates.
"""

from pathlib import Path

import pytest
from sqlmodel import select

from app.models import Attempt, Scan, ScanReview, StudentSkillMastery, Submission
from app.services.corrections import (
    CorrectionError,
    NotFoundError,
    correct_submission,
    resolve_review,
    set_review_status,
)
from tests.conftest import ENV_STUDENT_ID
from tests.fakes import FailingFakeVisionClient

# Correct answer is "A" for every question. This scan gets Q1 and Q3 right, Q2 wrong, Q4 blank.
INITIAL_MARKS = {1: "A", 2: "B", 3: "A", 4: None}


def _submission(env):
    return env.session.exec(select(Submission).where(Submission.worksheet_id == env.worksheet.worksheet_id)).one()


def test_correct_submission_recomputes_score_answers_attempts_and_audits(scan_env):
    env = scan_env
    env.run_scan(INITIAL_MARKS)
    submission = _submission(env)
    assert submission.score == 2

    corrected = correct_submission(env.session, submission.submission_id, {2: "A", 4: "A"}, "Asha")

    assert corrected.score == 4
    by_index = {a["question_index"]: a for a in corrected.answers_json}
    assert by_index[2] == {"question_index": 2, "selected_option": "A", "is_correct": True}
    assert by_index[4]["selected_option"] == "A" and by_index[4]["is_correct"] is True
    assert by_index[1]["is_correct"] is True  # untouched answers stay as they were

    attempts = env.session.exec(select(Attempt).where(Attempt.submission_id == submission.submission_id)).all()
    assert len(attempts) == 4 and all(a.is_correct for a in attempts)

    mastery = env.session.get(StudentSkillMastery, (ENV_STUDENT_ID, "1A"))
    assert mastery.mastery_score == 1.0

    audit = env.session.exec(
        select(ScanReview).where(ScanReview.submission_id == submission.submission_id)
    ).one()
    assert (audit.status, audit.corrected_by, audit.original_score, audit.corrected_score) == ("corrected", "Asha", 2, 4)


def test_rescanning_the_same_worksheet_bumps_submitted_at_so_it_shows_as_recent(scan_env):
    env = scan_env
    env.run_scan(INITIAL_MARKS)
    first = _submission(env).submitted_at

    env.run_scan(INITIAL_MARKS)  # student resends the same worksheet

    env.session.expire_all()
    assert _submission(env).submitted_at > first


def test_correction_can_blank_an_answer(scan_env):
    env = scan_env
    env.run_scan(INITIAL_MARKS)
    submission = _submission(env)

    corrected = correct_submission(env.session, submission.submission_id, {1: ""}, "Asha")

    assert corrected.score == 1
    q1 = next(a for a in corrected.answers_json if a["question_index"] == 1)
    assert q1["selected_option"] == "" and q1["is_correct"] is False


def test_correction_regenerates_the_checked_image(scan_env):
    env = scan_env
    env.run_scan(INITIAL_MARKS)
    submission = _submission(env)
    scan = env.session.exec(select(Scan).where(Scan.submission_id == submission.submission_id)).one()
    before = Path(scan.checked_image_path).read_bytes()

    correct_submission(env.session, submission.submission_id, {2: "A"}, "Asha")

    env.session.refresh(scan)
    assert Path(scan.checked_image_path).read_bytes() != before  # score badge + Q2 mark changed
    assert env.session.get(Submission, submission.submission_id).checked_image_url.endswith("_checked.jpg")


@pytest.mark.parametrize("bad", [{2: "Z"}, {99: "A"}])
def test_correction_rejects_invalid_option_or_question_without_changing_anything(scan_env, bad):
    env = scan_env
    env.run_scan(INITIAL_MARKS)
    submission = _submission(env)

    with pytest.raises(CorrectionError):
        correct_submission(env.session, submission.submission_id, bad, "Asha")

    env.session.refresh(submission)
    assert submission.score == 2


def test_correct_submission_unknown_id_and_empty_corrections(scan_env):
    env = scan_env
    with pytest.raises(NotFoundError):
        correct_submission(env.session, 999999999, {1: "A"}, "Asha")
    env.run_scan(INITIAL_MARKS)
    with pytest.raises(CorrectionError):
        correct_submission(env.session, _submission(env).submission_id, {}, "Asha")


def test_resolve_review_creates_submission_for_unrecognized_student(scan_env):
    env = scan_env
    env.run_scan(INITIAL_MARKS, roll_number="0000")  # roll number nobody has -> failed scan
    assert env.session.exec(select(Submission).where(Submission.worksheet_id == env.worksheet.worksheet_id)).first() is None
    review = env.session.exec(select(ScanReview).where(ScanReview.detected_roll_number == "0000")).one()
    assert review.status == "failed" and review.scan_id is not None

    submission = resolve_review(env.session, review.review_id, ENV_STUDENT_ID, {3: "B"}, "Asha")

    assert submission.student_id == ENV_STUDENT_ID
    assert submission.score == 1  # Q1 right; Q3 changed to B (wrong); Q2 wrong; Q4 blank
    q3 = next(a for a in submission.answers_json if a["question_index"] == 3)
    assert q3["selected_option"] == "B" and q3["is_correct"] is False
    assert len(env.session.exec(select(Attempt).where(Attempt.submission_id == submission.submission_id)).all()) == 4

    env.session.refresh(review)
    assert review.status == "corrected" and review.submission_id == submission.submission_id
    assert (review.original_score, review.corrected_score) == (2, 1)
    scan = env.session.get(Scan, review.scan_id)
    assert scan.outcome == "graded" and scan.submission_id == submission.submission_id
    assert scan.checked_image_path and Path(scan.checked_image_path).is_file()


def test_resolve_review_rejections(scan_env):
    env = scan_env
    env.run_scan(INITIAL_MARKS, roll_number="0000")
    review = env.session.exec(select(ScanReview).where(ScanReview.detected_roll_number == "0000")).one()

    with pytest.raises(CorrectionError, match="does not exist"):
        resolve_review(env.session, review.review_id, "8888", {}, "Asha")  # unknown student
    with pytest.raises(CorrectionError, match="does not exist on this worksheet"):
        resolve_review(env.session, review.review_id, ENV_STUDENT_ID, {77: "A"}, "Asha")
    with pytest.raises(NotFoundError):
        resolve_review(env.session, 999999999, ENV_STUDENT_ID, {}, "Asha")

    resolve_review(env.session, review.review_id, ENV_STUDENT_ID, {}, "Asha")
    with pytest.raises(CorrectionError, match="already"):
        resolve_review(env.session, review.review_id, ENV_STUDENT_ID, {}, "Asha")


def test_resolve_review_rejects_correcting_questions_not_on_the_scanned_page(scan_env):
    env = scan_env
    env.run_scan({1: "A", 2: "B"}, roll_number="0000")  # this page only detected Q1-Q2
    review = env.session.exec(select(ScanReview).where(ScanReview.detected_roll_number == "0000")).one()

    with pytest.raises(CorrectionError, match="not on the scanned page"):
        resolve_review(env.session, review.review_id, ENV_STUDENT_ID, {4: "A"}, "Asha")


def test_resolve_review_needs_stored_vision_result(scan_env):
    """A scan where vision-service itself failed has nothing to grade from -- it can only be dismissed."""
    from app.services.submission_service import handle_incoming_image

    env = scan_env
    handle_incoming_image(
        env.session, FailingFakeVisionClient(), env.comm, "+910000000099", str(env.storage / "x.jpg"), "crt-vision-fail"
    )
    review = env.session.exec(select(ScanReview).where(ScanReview.correlation_id == "crt-vision-fail")).one()
    assert review.scan_id is not None
    assert env.session.get(Scan, review.scan_id).vision_result is None

    with pytest.raises(CorrectionError, match="no stored vision result"):
        resolve_review(env.session, review.review_id, ENV_STUDENT_ID, {}, "Asha")

    dismissed = set_review_status(env.session, review.review_id, "approved", "Asha")
    assert dismissed.status == "approved" and dismissed.corrected_by == "Asha"
    with pytest.raises(CorrectionError, match="Unknown status"):
        set_review_status(env.session, review.review_id, "bogus")


def _review(env, roll_number):
    return env.session.exec(select(ScanReview).where(ScanReview.detected_roll_number == roll_number)).one()


def test_failed_review_is_closed_when_the_same_scan_later_grades(scan_env):
    """The reported bug: 'student not registered' failures stayed open in the dashboard forever
    after the student was added and the worksheet was successfully rescanned."""
    env = scan_env
    env.run_scan(INITIAL_MARKS, roll_number="9994")  # student 9994 doesn't exist yet
    assert _review(env, "9994").status == "failed"

    env.add_student("9994")
    env.run_scan(INITIAL_MARKS, roll_number="9994")  # resent after registering

    env.session.expire_all()
    review = _review(env, "9994")
    submission = env.session.exec(select(Submission).where(Submission.student_id == "9994")).one()
    assert (review.status, review.corrected_by, review.submission_id) == (
        "corrected", "system (later scan graded)", submission.submission_id,
    )
    assert review.student_id == "9994" and review.worksheet_id == env.worksheet.worksheet_id


def test_failed_review_stays_open_for_a_different_page(scan_env):
    env = scan_env
    env.run_scan(INITIAL_MARKS, roll_number="9994", page_no=1)
    env.add_student("9994")
    env.run_scan(INITIAL_MARKS, roll_number="9994", page_no=2)  # page 2 graded; page 1 still unresolved

    env.session.expire_all()
    assert _review(env, "9994").status == "failed"


def test_other_students_reviews_and_scanless_legacy_reviews(scan_env):
    from app.models import ScanReview as Review

    env = scan_env
    env.run_scan(INITIAL_MARKS, roll_number="0000")  # someone else's failure must survive
    legacy = Review(detected_roll_number=ENV_STUDENT_ID, status="failed", error_reason="legacy, no stored scan")
    env.session.add(legacy)
    env.session.commit()

    env.run_scan(INITIAL_MARKS)  # env student grades fine

    env.session.expire_all()
    assert _review(env, "0000").status == "failed"
    assert env.session.get(Review, legacy.review_id).status == "corrected"  # legacy: matched on roll number


def test_resolving_a_review_also_closes_duplicate_open_reviews(scan_env):
    env = scan_env
    env.run_scan(INITIAL_MARKS, roll_number="0000")
    env.run_scan(INITIAL_MARKS, roll_number="0000")  # same failure, resent
    first, second = env.session.exec(
        select(ScanReview).where(ScanReview.detected_roll_number == "0000").order_by(ScanReview.review_id)
    ).all()

    resolve_review(env.session, first.review_id, ENV_STUDENT_ID, {}, "Asha")

    env.session.expire_all()
    assert env.session.get(ScanReview, second.review_id).status == "corrected"
