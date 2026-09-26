"""Admin/facilitator corrections: fix individual question answers on a graded submission, and
turn a failed scan (unrecognized roll number, etc.) into a real submission once an admin picks
the right student.

Unlike the old system's correct endpoint (which trusted a client-supplied score and never
touched attempts/mastery), every correction here recomputes is_correct and the score on the
server from the answer key, rewrites attempts, and refreshes mastery/level for the affected
skills. Each correction also leaves a ScanReview audit row holding the original vs corrected
answers (columns that existed but were never written before).
"""

import copy
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import Session, select

from app.core.config import settings
from app.domain.annotation import draw_checked_image
from app.domain.errors import VisionClientError
from app.domain.grading import grade_marks, resolve_answer_key
from app.domain.mastery import evaluate_and_update_level, recalculate_skill_mastery
from app.domain.submission_merge import resolve_page_range
from app.models import (
    Attempt,
    Question,
    QuestionOption,
    Scan,
    ScanReview,
    Student,
    Submission,
    Worksheet,
    WorksheetTemplate,
)
from app.models.submission import ScanOutcome, ScanReviewStatus
from app.routes.files import checked_image_url
from app.services.submission_service import (
    affected_skills,
    close_superseded_reviews,
    insert_attempts,
    persist_graded_scan,
)
from app.services.vision_client import VisionClient
from shared.contracts import QuestionMark

logger = logging.getLogger(__name__)

DEFAULT_OPTION_LABELS = frozenset("ABCD")


class NotFoundError(LookupError):
    """The submission/review being corrected doesn't exist."""


class CorrectionError(ValueError):
    """The requested correction is invalid (bad option, unknown student, nothing to grade from...)."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _valid_labels_by_index(session: Session, worksheet_id: int) -> dict[int, frozenset[str]]:
    questions = session.exec(select(Question).where(Question.worksheet_id == worksheet_id)).all()
    labels: dict[int, set[str]] = {q.question_id: set() for q in questions}
    if labels:
        for option in session.exec(select(QuestionOption).where(QuestionOption.question_id.in_(list(labels)))).all():
            labels[option.question_id].add(option.option_label)
    return {
        q.index: frozenset(labels[q.question_id]) or DEFAULT_OPTION_LABELS
        for q in questions
        if q.index is not None
    }


def _normalize_corrections(
    corrections: dict[int, str | None], valid_labels: dict[int, frozenset[str]]
) -> dict[int, str]:
    """Returns {question_index: option or ""} (empty string = unanswered), rejecting anything that
    isn't a real question of the worksheet or a real option label for it."""
    normalized: dict[int, str] = {}
    for index, option in corrections.items():
        if index not in valid_labels:
            raise CorrectionError(f"Question {index} does not exist on this worksheet.")
        selected = (option or "").strip().upper()
        if selected and selected not in valid_labels[index]:
            raise CorrectionError(
                f"'{selected}' is not a valid option for question {index} "
                f"(expected one of {sorted(valid_labels[index])} or blank)."
            )
        normalized[index] = selected
    return normalized


def _apply_to_answer(answer_key: dict[int, str], index: int, selected: str) -> dict:
    return {
        "question_index": index,
        "selected_option": selected,
        "is_correct": bool(selected) and answer_key.get(index) == selected,
    }


def _regenerate_checked_image(session: Session, scan: Scan, page_answers: list[dict]) -> str | None:
    """Redraws the annotated image for one scanned page from its stored dewarped image + ROI
    boxes. Best-effort: a missing image file just skips it (older/foreign scans have none)."""
    if not scan.dewarped_path or not scan.vision_result or not Path(scan.dewarped_path).is_file():
        return None
    try:
        marks = [QuestionMark(**m) for m in scan.vision_result.get("question_marks", [])]
        page_score = sum(1 for a in page_answers if a["is_correct"])
        filename = f"{scan.correlation_id}_checked.jpg"
        output_path = Path(settings.storage_root) / "checked" / filename
        draw_checked_image(scan.dewarped_path, marks, page_answers, page_score, len(page_answers), str(output_path))
    except Exception:
        logger.exception("Failed to regenerate checked image for scan id=%s", scan.id)
        return None
    scan.checked_image_path = str(output_path)
    session.add(scan)
    return filename


def correct_submission(
    session: Session,
    submission_id: int,
    corrections: dict[int, str | None],
    corrected_by: str,
    question_paper_code: str | None = None,
) -> Submission:
    submission = session.get(Submission, submission_id)
    if submission is None:
        raise NotFoundError(f"Submission {submission_id} not found.")
    if not corrections:
        raise CorrectionError("No corrections supplied.")

    scans = session.exec(
        select(Scan).where(Scan.submission_id == submission_id).order_by(Scan.created_at.desc())
    ).all()
    code = question_paper_code or next((s.question_paper_code for s in scans if s.question_paper_code), None)

    valid_labels = _valid_labels_by_index(session, submission.worksheet_id)
    normalized = _normalize_corrections(corrections, valid_labels)

    answer_key = resolve_answer_key(session, submission.worksheet_id, code)
    if not answer_key:
        raise CorrectionError(
            f"No answer key resolvable for worksheet {submission.worksheet_id} "
            f"(question_paper_code={code!r}); supply one to re-grade."
        )

    original_answers = copy.deepcopy(submission.answers_json or [])
    original_score = submission.score
    by_index = {a["question_index"]: dict(a) for a in original_answers}
    for index, selected in normalized.items():
        by_index[index] = _apply_to_answer(answer_key, index, selected)
    merged = [by_index[i] for i in sorted(by_index)]
    score = sum(1 for a in merged if a.get("is_correct"))

    submission.answers_json = merged
    submission.score = score
    session.add(submission)

    for attempt in session.exec(select(Attempt).where(Attempt.submission_id == submission_id)).all():
        session.delete(attempt)
    session.commit()
    insert_attempts(session, submission.student_id, submission, submission.worksheet_id, merged)

    changed_answers = [by_index[i] for i in normalized]
    for skill_code in affected_skills(session, submission.worksheet_id, changed_answers):
        recalculate_skill_mastery(session, submission.student_id, skill_code)
    evaluate_and_update_level(session, submission.student_id)

    for scan in scans:
        indices = {m["question_index"] for m in (scan.vision_result or {}).get("question_marks", [])}
        page_answers = [a for a in merged if a["question_index"] in indices]
        filename = _regenerate_checked_image(session, scan, page_answers)
        if filename and scan is scans[0]:
            submission.checked_image_path = scan.checked_image_path
            submission.checked_image_url = checked_image_url(filename)

    latest = scans[0] if scans else None
    now = _utcnow()
    session.add(
        ScanReview(
            submission_id=submission_id,
            student_id=submission.student_id,
            worksheet_id=submission.worksheet_id,
            detected_roll_number=latest.roll_number if latest else None,
            correlation_id=latest.correlation_id if latest else None,
            scan_id=latest.id if latest else None,
            status=ScanReviewStatus.CORRECTED.value,
            error_reason="Manual correction from dashboard",
            original_answers=original_answers,
            corrected_answers=merged,
            original_score=original_score,
            corrected_score=score,
            corrected_by=corrected_by,
            corrected_at=now,
        )
    )
    session.add(submission)
    session.commit()
    session.refresh(submission)
    return submission


def resolve_review(
    session: Session,
    review_id: int,
    student_id: str,
    corrections: dict[int, str | None],
    corrected_by: str,
    worksheet_id: int | None = None,
    question_paper_code: str | None = None,
) -> Submission:
    """Turn a failed/needs-review scan into a graded submission for the student the admin picked."""
    review = session.get(ScanReview, review_id)
    if review is None:
        raise NotFoundError(f"Review {review_id} not found.")
    if review.status in (ScanReviewStatus.CORRECTED.value, ScanReviewStatus.APPROVED.value):
        raise CorrectionError(f"Review {review_id} is already {review.status}.")

    scan = session.get(Scan, review.scan_id) if review.scan_id else None
    if scan is None or not scan.vision_result or not scan.vision_result.get("question_marks"):
        raise CorrectionError(
            "This scan has no stored vision result (vision-service failed), so there is nothing to "
            "grade. Ask the student to resend it, and dismiss this entry."
        )

    student = session.get(Student, student_id)
    if student is None:
        raise CorrectionError(f"Student '{student_id}' does not exist.")
    target_worksheet_id = worksheet_id or scan.worksheet_id
    worksheet = session.get(Worksheet, target_worksheet_id) if target_worksheet_id is not None else None
    if worksheet is None:
        raise CorrectionError(f"Worksheet '{target_worksheet_id}' does not exist; supply the right worksheet_id.")

    code = question_paper_code or scan.question_paper_code
    answer_key = resolve_answer_key(session, worksheet.worksheet_id, code)
    if not answer_key:
        raise CorrectionError(
            f"No answer key resolvable for worksheet {worksheet.worksheet_id} (question_paper_code={code!r})."
        )

    marks = [QuestionMark(**m) for m in scan.vision_result["question_marks"]]
    graded, _ = grade_marks(marks, answer_key)
    original_page = copy.deepcopy(graded)
    original_page_score = sum(1 for a in original_page if a["is_correct"])

    page_indices = {a["question_index"] for a in graded}
    normalized = _normalize_corrections(corrections, _valid_labels_by_index(session, worksheet.worksheet_id))
    outside = sorted(set(normalized) - page_indices)
    if outside:
        raise CorrectionError(f"Questions {outside} are not on the scanned page (page {scan.page_no}).")
    scanned_answers = [
        _apply_to_answer(answer_key, a["question_index"], normalized[a["question_index"]])
        if a["question_index"] in normalized else a
        for a in graded
    ]
    corrected_page_score = sum(1 for a in scanned_answers if a["is_correct"])

    submission, _, _ = persist_graded_scan(
        session,
        student=student,
        worksheet=worksheet,
        from_number=scan.from_number,
        scanned_answers=scanned_answers,
        page_range=resolve_page_range(session, worksheet.worksheet_id, scan.page_no),
        template_name=scan.template_name,
        roll_number=student.student_id,
        correlation_id=scan.correlation_id,
    )

    decoded_worksheet_id = scan.worksheet_id  # what the scan said, before any admin override
    scan.outcome = ScanOutcome.GRADED.value
    scan.submission_id = submission.submission_id
    scan.worksheet_id = worksheet.worksheet_id
    filename = _regenerate_checked_image(session, scan, scanned_answers)
    if filename:
        submission.checked_image_path = scan.checked_image_path
        submission.checked_image_url = checked_image_url(filename)
        session.add(submission)
    session.add(scan)

    review.student_id = student.student_id
    review.worksheet_id = worksheet.worksheet_id
    review.submission_id = submission.submission_id
    review.status = ScanReviewStatus.CORRECTED.value
    review.original_answers = original_page
    review.original_score = original_page_score
    review.corrected_answers = scanned_answers
    review.corrected_score = corrected_page_score
    review.corrected_by = corrected_by
    review.corrected_at = _utcnow()
    review.updated_at = _utcnow()
    session.add(review)
    session.commit()
    # Any other open duplicates of this same scan situation (e.g. the student resent it) are done too.
    close_superseded_reviews(
        session, roll_number=review.detected_roll_number or student.student_id, student_id=student.student_id,
        worksheet_id=worksheet.worksheet_id, submission_id=submission.submission_id,
        match_worksheet_id=decoded_worksheet_id, page_no=scan.page_no, exclude_review_id=review_id,
    )
    session.refresh(submission)
    return submission


def retry_scan(session: Session, vision_client: VisionClient, review_id: int, worksheet_id: int) -> Scan:
    """Re-sends a failed scan's original photo to vision-service, telling it which worksheet this
    is so it can be graded even though the scan itself couldn't establish that on its own -- e.g.
    the row tags that carry worksheet_id were unreadable (torn/smudged/glare) even though the rest
    of the page's tags were fine. Used for both "tags not detected" and "roll number invalid"
    failures: neither has any question_marks stored yet (vision-service raised before computing
    them), so there's nothing for the dashboard's correction UI to show until this succeeds.

    template_hint is derived from the admin-supplied worksheet (not guessed from the scan, which
    is exactly what's unreliable here) so vision-service can still infer the right ROI layout even
    if the row tags remain undecodable this time round.

    Does not touch review.status or grade anything -- a successful retry only recovers
    question_marks onto the scan; resolve_review still needs to run afterwards to pick a student
    and turn it into a Submission, same as any other open review.
    """
    review = session.get(ScanReview, review_id)
    if review is None:
        raise NotFoundError(f"Review {review_id} not found.")
    if review.status not in (ScanReviewStatus.FAILED.value, ScanReviewStatus.NEEDS_REVIEW.value):
        raise CorrectionError(f"Review {review_id} is already {review.status}; nothing to retry.")

    scan = session.get(Scan, review.scan_id) if review.scan_id else None
    if scan is None or not scan.upload_path:
        raise CorrectionError("This review has no stored photo to retry.")

    worksheet = session.get(Worksheet, worksheet_id)
    if worksheet is None:
        raise CorrectionError(f"Worksheet '{worksheet_id}' does not exist.")
    template = session.get(WorksheetTemplate, worksheet.template_id) if worksheet.template_id else None

    try:
        result = vision_client.process(scan.upload_path, scan.correlation_id, template_hint=template.name if template else None)
    except VisionClientError as exc:
        raise CorrectionError(
            f"vision-service still could not process this image: {exc}. A clearer photo is likely needed."
        ) from exc

    scan.worksheet_id = worksheet.worksheet_id
    scan.page_no = result.page_no
    scan.template_name = result.template_name
    scan.roll_number = result.roll_number
    scan.question_paper_code = result.question_paper_code
    scan.vision_result = result.model_dump()
    scan.dewarped_path = result.dewarped_image_path
    session.add(scan)

    review.worksheet_id = worksheet.worksheet_id
    review.detected_roll_number = result.roll_number or review.detected_roll_number
    review.updated_at = _utcnow()
    session.add(review)
    session.commit()
    session.refresh(scan)
    return scan


def set_review_status(session: Session, review_id: int, status: str, corrected_by: str | None = None) -> ScanReview:
    """Move a review between needs_review/approved (dismiss)/failed without re-grading anything."""
    if status not in {s.value for s in ScanReviewStatus}:
        raise CorrectionError(f"Unknown status '{status}'.")
    review = session.get(ScanReview, review_id)
    if review is None:
        raise NotFoundError(f"Review {review_id} not found.")
    review.status = status
    review.updated_at = _utcnow()
    if corrected_by:
        review.corrected_by = corrected_by
    session.add(review)
    session.commit()
    session.refresh(review)
    return review
