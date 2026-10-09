"""Re-runs scans that failed only because vision-service was unreachable, overloaded or crashed
(timeouts, connection refused, dropped connections, 5xx) -- as opposed to scans vision-service
looked at and rejected (e.g. "Tag family ... requires at least N detections", HTTP 422), which
would just fail again and need a better photo.

Such a failure leaves a Scan (with the stored photo, no vision result) and a FAILED ScanReview
with no submission. Reprocessing sends that photo through the normal handle_incoming_image path
and, on success, closes the old review against the new submission.
"""

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, or_
from sqlmodel import Session, select

from app.domain.errors import VisionClientError
from app.models import Scan, ScanReview, Student, Submission
from app.models.core import utcnow
from app.models.submission import ScanReviewStatus
from app.services.communication import CapturingCommunicationClient
from app.services.submission_service import handle_incoming_image
from app.services.vision_client import VisionClient
from shared.contracts import ProcessingResult

logger = logging.getLogger(__name__)

RESOLVER = "system (reprocessed after vision-service outage)"

# VisionClientError text for "vision-service didn't answer" ("... failed: timed out") versus
# "vision-service answered and said no" ("... failed (422): ..."). 5xx are included: a crash
# mid-request surfaces as one, and retrying costs nothing.
_INFRA_PREFIX = "vision-service call failed:"
_SERVER_ERROR_PREFIXES = tuple(f"vision-service call failed ({code}" for code in (500, 502, 503, 504))


@dataclass
class ReprocessOutcome:
    review_id: int
    status: str  # graded | still_failing | superseded | skipped
    detail: str = ""


def find_infra_failed_reviews(
    session: Session, *, project_code: str | None = None, limit: int | None = None
) -> list[tuple[ScanReview, Scan]]:
    """Oldest first. Only reviews still open, with no submission, whose stored photo exists."""
    query = (
        select(ScanReview, Scan)
        .join(Scan, Scan.id == ScanReview.scan_id)
        .where(
            ScanReview.status == ScanReviewStatus.FAILED.value,
            ScanReview.submission_id.is_(None),
            Scan.upload_path.is_not(None),
            # An explicit Python None in a JSONB column is stored as JSON null, not SQL NULL.
            or_(Scan.vision_result.is_(None), func.jsonb_typeof(Scan.vision_result) == "null"),
            or_(
                ScanReview.error_reason.startswith(_INFRA_PREFIX),
                *[ScanReview.error_reason.startswith(p) for p in _SERVER_ERROR_PREFIXES],
            ),
        )
        .order_by(ScanReview.created_at, ScanReview.review_id)
    )
    if project_code:
        query = query.where(Scan.project_code == project_code)
    if limit:
        query = query.limit(limit)
    return list(session.exec(query).all())


class _ReplayVisionClient:
    """Hands back a result that was already fetched, so the pre-check below and the real grading
    pass don't each pay for a vision call on the same photo."""

    def __init__(self, result: ProcessingResult):
        self._result = result

    def process(self, image_path, correlation_id, template_hint=None, skip_corner_tags=False,
                roll_number=None, question_paper_code=None):
        return self._result


def reprocess_review(session: Session, vision_client: VisionClient, review: ScanReview, scan: Scan) -> ReprocessOutcome:
    """Never overwrites an existing submission and never messages anyone."""
    if not Path(scan.upload_path).is_file():
        return ReprocessOutcome(review.review_id, "skipped", f"photo not found on disk: {scan.upload_path}")

    correlation_id = str(uuid.uuid4())
    try:
        result = vision_client.process(scan.upload_path, correlation_id)
    except VisionClientError as exc:
        return ReprocessOutcome(review.review_id, "still_failing", str(exc))

    # Grading a page rewrites that student's submission for the worksheet, so refuse if one exists
    # already (a later rescan that worked, or an earlier page of the same sheet): an admin decides.
    if result.roll_number and result.worksheet_id:
        existing = session.exec(
            select(Submission.submission_id).where(
                Submission.student_id == result.roll_number, Submission.worksheet_id == result.worksheet_id,
            )
        ).first()
        if existing is not None and session.get(Student, result.roll_number) is not None:
            return ReprocessOutcome(
                review.review_id, "skipped",
                f"student {result.roll_number} already has submission {existing} for worksheet "
                f"{result.worksheet_id}; resolve from the dashboard instead",
            )

    replies = CapturingCommunicationClient()
    handle_incoming_image(
        session, _ReplayVisionClient(result), replies, scan.from_number, scan.upload_path, correlation_id,
    )

    new_scan = session.exec(select(Scan).where(Scan.correlation_id == correlation_id)).first()
    if new_scan is not None and new_scan.submission_id is not None:
        review.status = ScanReviewStatus.CORRECTED.value
        review.submission_id = new_scan.submission_id
        review.student_id = new_scan.roll_number
        review.worksheet_id = new_scan.worksheet_id
        review.detected_roll_number = new_scan.roll_number
        review.corrected_by = RESOLVER
        review.corrected_at = review.updated_at = utcnow()
        session.add(review)
        session.commit()
        return ReprocessOutcome(review.review_id, "graded", f"submission {new_scan.submission_id}")

    # Vision worked this time but the scan fails on its merits (unknown student, bad worksheet,
    # no answer key). That run recorded its own, more specific review; this outage-era one is
    # now redundant, so dismiss it rather than leave two entries for one photo.
    new_review = session.exec(select(ScanReview).where(ScanReview.correlation_id == correlation_id)).first()
    review.status = ScanReviewStatus.APPROVED.value
    review.corrected_by = RESOLVER
    review.corrected_at = review.updated_at = utcnow()
    session.add(review)
    session.commit()
    reason = new_review.error_reason if new_review else "unknown"
    return ReprocessOutcome(
        review.review_id, "superseded",
        f"now fails on its merits ({reason}); see new review {new_review.review_id if new_review else '?'}",
    )
