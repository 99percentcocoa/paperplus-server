"""scripts/reprocess_failed_scans.py's logic: re-run scans that failed only because vision-service
was unreachable, never overwrite an existing submission, never touch genuine bad-photo failures."""

from sqlmodel import select

from app.models import Scan, ScanReview, Submission
from app.models.submission import ScanReviewStatus
from app.services.reprocess import find_infra_failed_reviews, reprocess_review
from shared.contracts import ProcessingResult, QuestionMark
from tests.fakes import FakeVisionClient, FailingFakeVisionClient


def _failed(env, reason):
    """A vision-failure scan/review as handle_incoming_image records it, with a chosen reason."""
    corr = env.run_failed_scan()
    scan = env.session.exec(select(Scan).where(Scan.correlation_id == corr)).one()
    review = env.session.exec(select(ScanReview).where(ScanReview.scan_id == scan.id)).one()
    review.error_reason = reason
    env.session.add(review)
    env.session.commit()
    return review, scan


def _result(env, roll_number=None):
    return ProcessingResult(
        worksheet_id=env.worksheet.worksheet_id, page_no=1, first_question_index=1, template_name="regular",
        roll_number=roll_number or env.student.student_id, roll_number_confidence=None, question_paper_code="",
        question_marks=[
            QuestionMark(question_index=i, marked_option="A", confidence=0.9,
                         roi_x1=10, roi_y1=10, roi_x2=300, roi_y2=60) for i in (1, 2, 3, 4)
        ],
        dewarped_image_path=None,
    )


def test_selects_only_infrastructure_failures(scan_env):
    timeout, _ = _failed(scan_env, "vision-service call failed: timed out")
    crash, _ = _failed(scan_env, "vision-service call failed (500): boom")
    bad_photo, _ = _failed(scan_env, "vision-service call failed (422): Tag family '36h11' requires at least 4 detections, but got 1")
    ids = {r.review_id for r, _ in find_infra_failed_reviews(scan_env.session)}
    assert {timeout.review_id, crash.review_id} <= ids
    assert bad_photo.review_id not in ids


def test_reprocess_grades_and_closes_old_review(scan_env):
    review, scan = _failed(scan_env, "vision-service call failed: timed out")
    outcome = reprocess_review(scan_env.session, FakeVisionClient(_result(scan_env)), review, scan)

    assert outcome.status == "graded"
    scan_env.session.refresh(review)
    assert review.status == ScanReviewStatus.CORRECTED.value
    submission = scan_env.session.get(Submission, review.submission_id)
    assert submission.student_id == scan_env.student.student_id and submission.score == 4
    # Idempotent: no longer selected on a second run.
    assert review.review_id not in {r.review_id for r, _ in find_infra_failed_reviews(scan_env.session)}


def test_reprocess_sends_nothing_over_whatsapp(scan_env):
    review, scan = _failed(scan_env, "vision-service call failed: timed out")
    before = len(scan_env.comm.sent_messages)  # the original failure reply
    reprocess_review(scan_env.session, FakeVisionClient(_result(scan_env)), review, scan)
    assert len(scan_env.comm.sent_messages) == before


def test_vision_still_down_leaves_review_untouched(scan_env):
    review, scan = _failed(scan_env, "vision-service call failed: timed out")
    outcome = reprocess_review(scan_env.session, FailingFakeVisionClient(), review, scan)
    assert outcome.status == "still_failing"
    scan_env.session.refresh(review)
    assert review.status == ScanReviewStatus.FAILED.value and review.submission_id is None
    assert len(scan_env.session.exec(select(ScanReview).where(ScanReview.scan_id == scan.id)).all()) == 1


def test_existing_submission_is_never_overwritten(scan_env):
    scan_env.run_scan({1: "A", 2: "A", 3: "A", 4: "A"})
    original = scan_env.session.exec(
        select(Submission).where(Submission.worksheet_id == scan_env.worksheet.worksheet_id)
    ).one()
    review, scan = _failed(scan_env, "vision-service call failed: timed out")

    outcome = reprocess_review(scan_env.session, FakeVisionClient(_result(scan_env)), review, scan)

    assert outcome.status == "skipped"
    scan_env.session.refresh(review)
    assert review.status == ScanReviewStatus.FAILED.value
    scan_env.session.refresh(original)
    assert original.score == 4


def test_unknown_student_supersedes_old_review(scan_env):
    review, scan = _failed(scan_env, "vision-service call failed: timed out")
    outcome = reprocess_review(scan_env.session, FakeVisionClient(_result(scan_env, roll_number="0000")), review, scan)

    assert outcome.status == "superseded"
    scan_env.session.refresh(review)
    assert review.status == ScanReviewStatus.APPROVED.value
    new_reviews = scan_env.session.exec(select(ScanReview).where(ScanReview.detected_roll_number == "0000")).all()
    assert len(new_reviews) == 1 and new_reviews[0].status == ScanReviewStatus.FAILED.value
