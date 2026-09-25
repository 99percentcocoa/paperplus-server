"""Orchestrates: vision-service call -> student/worksheet validation -> grading ->
persistence (via the Phase 2 state machine) -> mastery/level update -> WhatsApp reply.

Adaptation note: because vision-service is a single synchronous call (no queue, per the
redevelopment decision), the state machine transitions all happen back-to-back right after
the call returns, rather than being updated incrementally as each processing stage completes
in a background worker. The Submission row can only be created once we know student_id +
worksheet_id (both are NOT NULL), which vision-service is what supplies — so UPLOADED is
the first state recorded, immediately followed by the rest.
"""

import logging
from pathlib import Path

from sqlmodel import Session, select

from app.core.config import settings
from app.domain.annotation import draw_checked_image
from app.domain.errors import InvalidAnswerKeyError, InvalidStudentError, InvalidWorksheetError, VisionClientError
from app.domain.grading import grade_marks, resolve_answer_key
from app.domain.mastery import evaluate_and_update_level, recalculate_skill_mastery
from app.domain.submission_merge import merge_page_answers, resolve_page_range
from app.domain.submission_state import (
    transition_to_dewarped,
    transition_to_graded,
    transition_to_preprocessing,
    transition_to_registering,
    transition_to_scoring,
)
from app.models import Attempt, Question, Scan, ScanReview, Student, Submission, Worksheet
from app.models.core import utcnow
from app.models.submission import ProcessingState, ScanOutcome, ScanReviewStatus
from app.routes.files import checked_image_url
from app.services.communication import CommunicationClient
from app.services.sheets_logging import log_to_sheet_async
from app.services.vision_client import VisionClient
from shared.logging_config import correlation_id_var

logger = logging.getLogger(__name__)

MESSAGES = {
    "vision_failed": (
        "The worksheet could not be read properly. Please try again. ⟳ \n"
        "आपल्या फोटोमध्ये कार्यपत्रिका नीट दिसली नाही. कृपया पुन्हा प्रयत्न करा किंवा कार्यपत्रिका तुमच्या शिक्षकांना द्या. ⟳"
    ),
    "invalid_student": (
        "Roll number not recognized. Please check and try again. ⟳ \n"
        "रोल नंबर ओळखता आला नाही. कृपया तपासून परत पाठवा किंवा कार्यपत्रिका तुमच्या शिक्षकांना द्या. ⟳"
    ),
    "invalid_worksheet": (
        "This worksheet could not be processed. Please try again. ⟳ \n"
        "ही कार्यपत्रिका तपासता आली नाही. कृपया परत प्रयत्न करा किंवा कार्यपत्रिका तुमच्या शिक्षकांना द्या. ⟳"
    ),
    "invalid_answer_key": (
        "This worksheet is not ready to be graded yet. Please contact your facilitator. ⟳ \n"
        "ही कार्यपत्रिका तपासण्यासाठी तयार नाही. कृपया आपल्या शिक्षकांशी संपर्क साधा. ⟳"
    ),
}


def handle_incoming_image(
    session: Session,
    vision_client: VisionClient,
    comm_client: CommunicationClient,
    from_number: str,
    image_path: str,
    correlation_id: str,
) -> None:
    # Set (not just read) here, not only in webhook.py, so this contextvar is always correct
    # regardless of caller (scripts/test_local_image.py and tests call this directly, bypassing
    # webhook.py's own correlation_id_var.set()) -- every log call below this point, and the
    # ScanReview rows written on failure, then automatically carry the right correlation_id.
    correlation_id_var.set(correlation_id)

    try:
        result = vision_client.process(image_path, correlation_id)
    except VisionClientError as exc:
        logger.exception("vision-service call failed")
        scan = _record_scan(session, correlation_id, from_number, image_path)
        _record_scan_review(
            session, status=ScanReviewStatus.FAILED, error_reason=str(exc), scan=scan,
        )
        comm_client.send_message(from_number, MESSAGES["vision_failed"])
        return

    logger.info(
        "vision-service result: worksheet_id=%s page_no=%s template_name=%s roll_number=%s "
        "roll_number_confidence=%s question_paper_code=%s question_marks_count=%s",
        result.worksheet_id, result.page_no, result.template_name,
        result.roll_number, result.roll_number_confidence, result.question_paper_code,
        len(result.question_marks),
    )
    scan = _record_scan(session, correlation_id, from_number, image_path, result)

    try:
        student = _validate_student(session, result.roll_number)
        worksheet = _validate_worksheet(session, result.worksheet_id)
        answer_key = _validate_answer_key(session, worksheet.worksheet_id, result.question_paper_code)
    except InvalidStudentError as exc:
        _record_scan_review(
            session, status=ScanReviewStatus.FAILED, error_reason=str(exc), detected_roll_number=result.roll_number,
            scan=scan,
        )
        comm_client.send_message(from_number, MESSAGES["invalid_student"])
        return
    except InvalidWorksheetError as exc:
        # worksheet_id itself doesn't reference a real row (that's the failure), so it can't be
        # set as the FK -- captured in error_reason as text instead. `student` is bound here since
        # _validate_student already succeeded on the line above.
        _record_scan_review(
            session, status=ScanReviewStatus.FAILED, error_reason=str(exc),
            student_id=student.student_id, detected_roll_number=result.roll_number, scan=scan,
        )
        comm_client.send_message(from_number, MESSAGES["invalid_worksheet"])
        return
    except InvalidAnswerKeyError as exc:
        logger.error(
            "No resolvable answer key: worksheet_id=%s question_paper_code=%r (no question_paper_variant "
            "seeded for this code, and no canonical question_options fallback either) -- refusing to "
            "grade rather than silently score 0.",
            worksheet.worksheet_id, result.question_paper_code,
        )
        # NEEDS_REVIEW rather than FAILED: this is a content-seeding gap (missing answer key),
        # not a bad/unreadable scan -- surfaces separately on a dashboard "failed scans" view.
        _record_scan_review(
            session, status=ScanReviewStatus.NEEDS_REVIEW, error_reason=str(exc),
            student_id=student.student_id, worksheet_id=worksheet.worksheet_id, detected_roll_number=result.roll_number,
            scan=scan,
        )
        comm_client.send_message(from_number, MESSAGES["invalid_answer_key"])
        return

    scanned_answers, page_score = grade_marks(result.question_marks, answer_key)
    page_range = resolve_page_range(session, worksheet.worksheet_id, result.page_no)

    submission, answers_payload, score = persist_graded_scan(
        session,
        student=student,
        worksheet=worksheet,
        from_number=from_number,
        scanned_answers=scanned_answers,
        page_range=page_range,
        template_name=result.template_name,
        roll_number=result.roll_number,
        correlation_id=correlation_id,
    )
    scan.outcome = ScanOutcome.GRADED.value
    scan.submission_id = submission.submission_id
    session.add(scan)
    session.commit()
    close_superseded_reviews(
        session, roll_number=result.roll_number, student_id=student.student_id,
        worksheet_id=worksheet.worksheet_id, submission_id=submission.submission_id, page_no=result.page_no,
    )

    comm_client.send_message(from_number, f"Your marks: {score}/{len(answers_payload)}")

    _annotate_and_send_checked_image(
        session, comm_client, submission, result, scanned_answers, page_score, from_number, correlation_id, scan,
    )


def persist_graded_scan(
    session: Session,
    *,
    student: Student,
    worksheet: Worksheet,
    from_number: str | None,
    scanned_answers: list[dict],
    page_range: tuple[int, int] | None,
    template_name: str | None,
    roll_number: str | None,
    correlation_id: str,
) -> tuple[Submission, list[dict], int]:
    """Everything after grading: merge this page into the student's submission, walk the state
    machine, rewrite attempts, refresh mastery/level, mark graded. Shared by the WhatsApp webhook
    flow and the admin dashboard's failed-scan resolution so both persist identically.
    Returns (submission, merged answers_payload, merged score).
    """
    submission, answers_payload, score = _create_or_overwrite_submission(
        session, student, worksheet, from_number, scanned_answers, page_range
    )

    transition_to_preprocessing(session, submission, service_name="api-service", correlation_id=correlation_id)
    transition_to_dewarped(session, submission, service_name="api-service", correlation_id=correlation_id,
                            detail={"template_name": template_name})
    transition_to_registering(session, submission, service_name="api-service", correlation_id=correlation_id,
                               detail={"roll_number": roll_number})
    transition_to_scoring(session, submission, service_name="api-service", correlation_id=correlation_id)

    attempts_count = insert_attempts(session, student.student_id, submission, worksheet.worksheet_id, answers_payload)
    skill_codes = affected_skills(session, worksheet.worksheet_id, answers_payload)
    for skill_code in skill_codes:
        recalculate_skill_mastery(session, student.student_id, skill_code)
    level_update = evaluate_and_update_level(session, student.student_id)

    transition_to_graded(
        session, submission, service_name="api-service", correlation_id=correlation_id,
        detail={"score": score, "attempts_count": attempts_count, "level_update": level_update},
    )
    return submission, answers_payload, score


SYSTEM_RESOLVER = "system (later scan graded)"
OPEN_REVIEW_STATUSES = (ScanReviewStatus.FAILED.value, ScanReviewStatus.NEEDS_REVIEW.value)


def close_superseded_reviews(
    session: Session,
    *,
    roll_number: str,
    student_id: str,
    worksheet_id: int,
    submission_id: int,
    match_worksheet_id: int | None = None,
    page_no: int | None = None,
    exclude_review_id: int | None = None,
) -> int:
    """Once a scan for (student, worksheet, page) grades successfully, earlier open failures for
    that same situation ("student not registered yet", "worksheet not inserted yet", "no answer
    key yet") are obsolete -- close them and link the submission instead of leaving them in the
    dashboard's failed-scans list forever. Conservative: a review only matches on the same roll
    number AND the same worksheet AND (if both known) the same page. A review with a real scan
    whose worksheet couldn't be decoded is left open (it isn't demonstrably the same scan);
    legacy reviews with no stored scan and no worksheet match on roll number alone.

    roll_number / match_worksheet_id are what the scan *said* (an admin may have assigned it to a
    different student or worksheet); student_id / worksheet_id are what it was graded as.
    """
    match_worksheet_id = match_worksheet_id if match_worksheet_id is not None else worksheet_id
    reviews = session.exec(
        select(ScanReview).where(
            ScanReview.status.in_(OPEN_REVIEW_STATUSES), ScanReview.detected_roll_number == roll_number
        )
    ).all()
    closed = 0
    for review in reviews:
        if review.review_id == exclude_review_id:
            continue
        scan = session.get(Scan, review.scan_id) if review.scan_id else None
        review_worksheet = review.worksheet_id or (scan.worksheet_id if scan else None)
        if review_worksheet is None:
            if scan is not None:
                continue
        elif review_worksheet != match_worksheet_id:
            continue
        if scan is not None and scan.page_no is not None and page_no is not None and scan.page_no != page_no:
            continue
        review.status = ScanReviewStatus.CORRECTED.value
        review.student_id = student_id
        review.worksheet_id = worksheet_id
        review.submission_id = submission_id
        review.corrected_by = SYSTEM_RESOLVER
        review.corrected_at = review.updated_at = utcnow()
        session.add(review)
        closed += 1
    if closed:
        session.commit()
        logger.info("Closed %s earlier failed-scan review(s) superseded by a graded scan", closed)
    return closed


def _record_scan(session: Session, correlation_id: str, from_number: str | None, image_path: str, result=None) -> Scan:
    """One Scan row per received image (success or failure) -- keeps the file locations and the
    full vision result so the dashboard can show and re-grade it later."""
    scan = Scan(correlation_id=correlation_id, from_number=from_number, upload_path=image_path)
    if result is not None:
        scan.dewarped_path = result.dewarped_image_path
        scan.worksheet_id = result.worksheet_id
        scan.page_no = result.page_no
        scan.template_name = result.template_name
        scan.roll_number = result.roll_number
        scan.question_paper_code = result.question_paper_code
        scan.vision_result = result.model_dump()
    session.add(scan)
    session.commit()
    session.refresh(scan)
    return scan


def _record_scan_review(
    session: Session,
    *,
    status: ScanReviewStatus,
    error_reason: str,
    student_id: str | None = None,
    worksheet_id: int | None = None,
    detected_roll_number: str | None = None,
    scan: Scan | None = None,
) -> None:
    """Persists a failed/needs-review scan that never reached a Submission row (schema requires
    submission_id on ProcessingEvent, but ScanReview.submission_id is nullable for exactly this
    case) -- so failed scans show up on a "failed scans" dashboard view instead of only in logs.
    correlation_id is read from the contextvar (set once per request in webhook.py) rather than
    threaded through as a parameter, so this row can be cross-referenced with the structured
    log lines that explain what happened in more detail than error_reason alone captures.
    """
    session.add(
        ScanReview(
            student_id=student_id,
            worksheet_id=worksheet_id,
            detected_roll_number=detected_roll_number,
            correlation_id=correlation_id_var.get(),
            scan_id=scan.id if scan is not None else None,
            status=status.value,
            error_reason=error_reason,
        )
    )
    session.commit()


def _annotate_and_send_checked_image(
    session: Session,
    comm_client: CommunicationClient,
    submission: Submission,
    result,
    scanned_answers: list[dict],
    page_score: int,
    from_number: str,
    correlation_id: str,
    scan: Scan | None = None,
) -> None:
    """Draws the correct/incorrect annotation for THIS scanned page (not the merged multi-page
    total -- the image only has this page's pixels) and sends it back over WhatsApp, mirroring
    the old system's per-scan checked-image send. Best-effort: any failure here is logged but
    must not fail the grading that already succeeded and was already messaged to the student.
    """
    if not result.dewarped_image_path:
        logger.warning(
            "No dewarped_image_path returned by vision-service for correlation_id=%s -- skipping "
            "checked-image annotation.", correlation_id,
        )
        return

    output_filename = f"{correlation_id}_checked.jpg"
    output_path = Path(settings.storage_root) / "checked" / output_filename

    try:
        draw_checked_image(
            result.dewarped_image_path, result.question_marks, scanned_answers,
            page_score, len(scanned_answers), str(output_path),
        )
    except Exception:
        logger.exception("Failed to draw checked image for correlation_id=%s", correlation_id)
        return

    url = checked_image_url(output_filename)
    submission.checked_image_path = str(output_path)
    submission.checked_image_url = url
    session.add(submission)
    if scan is not None:
        scan.checked_image_path = str(output_path)
        session.add(scan)
    session.commit()

    comm_client.send_image(from_number, url, "")
    log_to_sheet_async(
        from_number, url, scanned_answers, page_score, result.roll_number, result.worksheet_id,
    )


def _validate_student(session: Session, roll_number: str | None) -> Student:
    if not roll_number:
        raise InvalidStudentError("No roll number detected.")
    student = session.get(Student, roll_number)
    if student is None:
        raise InvalidStudentError(f"No registered student found for student_id '{roll_number}'.")
    return student


def _validate_worksheet(session: Session, worksheet_id: int | None) -> Worksheet:
    if worksheet_id is None:
        raise InvalidWorksheetError("No worksheet_id decoded from the scan.")
    worksheet = session.get(Worksheet, worksheet_id)
    if worksheet is None:
        raise InvalidWorksheetError(f"No worksheet found for worksheet_id '{worksheet_id}'.")
    return worksheet


def _validate_answer_key(session: Session, worksheet_id: int, question_paper_code: str | None) -> dict[int, str]:
    answer_key = resolve_answer_key(session, worksheet_id, question_paper_code)
    if not answer_key:
        raise InvalidAnswerKeyError(
            f"No resolvable answer key for worksheet_id={worksheet_id} question_paper_code={question_paper_code!r}."
        )
    return answer_key


def _create_or_overwrite_submission(
    session: Session,
    student: Student,
    worksheet: Worksheet,
    from_number: str,
    scanned_answers: list[dict],
    page_range: tuple[int, int] | None,
) -> tuple[Submission, list[dict], int]:
    """Merges this scan's page into any prior submission (see app.domain.submission_merge),
    returning the full merged answers_payload and its recomputed total score.
    """
    existing = session.exec(
        select(Submission)
        .where(Submission.student_id == student.student_id, Submission.worksheet_id == worksheet.worksheet_id)
        .order_by(Submission.submitted_at.desc())
    ).first()

    existing_answers = existing.answers_json if existing is not None else None
    merged_answers = merge_page_answers(existing_answers, scanned_answers, page_range)
    score = sum(1 for a in merged_answers if a.get("is_correct"))

    if existing is not None:
        for attempt in session.exec(select(Attempt).where(Attempt.submission_id == existing.submission_id)).all():
            session.delete(attempt)
        existing.score = score
        existing.from_number = from_number
        existing.answers_json = merged_answers
        # A rescan is a new submission event: without this the dashboard's recent-submissions
        # list (ordered by submitted_at) shows nothing new after a resend of the same worksheet.
        existing.submitted_at = utcnow()
        # Reset to UPLOADED so the caller's transition_to_* calls are valid again (a prior
        # submission is already GRADED/FAILED, which has no further allowed transitions).
        existing.state = ProcessingState.UPLOADED.value
        existing.processing_started_at = None
        existing.processing_completed_at = None
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return existing, merged_answers, score

    submission = Submission(
        student_id=student.student_id,
        worksheet_id=worksheet.worksheet_id,
        worksheet_category=worksheet.worksheet_category,
        score=score,
        from_number=from_number,
        answers_json=merged_answers,
    )
    session.add(submission)
    session.commit()
    session.refresh(submission)
    return submission, merged_answers, score


def insert_attempts(session: Session, student_id: str, submission: Submission, worksheet_id: int, answers_payload: list[dict]) -> int:
    questions_by_index = {
        q.index: q for q in session.exec(select(Question).where(Question.worksheet_id == worksheet_id)).all()
    }
    count = 0
    for ans in answers_payload:
        question = questions_by_index.get(ans["question_index"])
        if question is None:
            continue
        session.add(
            Attempt(
                student_id=student_id,
                submission_id=submission.submission_id,
                question_id=question.question_id,
                worksheet_id=worksheet_id,
                is_correct=ans["is_correct"],
                skill_code=question.skill_code,
            )
        )
        count += 1
    session.commit()
    return count


def affected_skills(session: Session, worksheet_id: int, answers_payload: list[dict]) -> set[str]:
    questions_by_index = {
        q.index: q for q in session.exec(select(Question).where(Question.worksheet_id == worksheet_id)).all()
    }
    return {
        questions_by_index[ans["question_index"]].skill_code
        for ans in answers_payload
        if ans["question_index"] in questions_by_index
    }
