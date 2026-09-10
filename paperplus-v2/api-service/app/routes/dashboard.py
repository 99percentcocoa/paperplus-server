"""Dashboard endpoints — health, latency, and (Phase 5) /metrics, replacing the old systemctl
check and in-memory latency list with real DB state.
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlmodel import Session, select

from app.db.session import get_session
from app.models import ScanReview, Submission
from app.models.submission import ProcessingState

router = APIRouter(prefix="/api/dashboard")


@router.get("/health")
def health(session: Session = Depends(get_session)) -> dict:
    try:
        session.exec(select(1))
        db_ok = True
    except Exception:  # noqa: BLE001 - health check should never raise, just report status
        db_ok = False

    return {
        "status": "active" if db_ok else "failed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/latency")
def latency(session: Session = Depends(get_session)) -> dict:
    recent = session.exec(
        select(Submission)
        .where(Submission.processing_completed_at.is_not(None))
        .order_by(Submission.processing_completed_at.desc())
        .limit(25)
    ).all()

    records = []
    durations = []
    for submission in recent:
        if submission.processing_started_at is None:
            continue
        duration_ms = int(
            (submission.processing_completed_at - submission.processing_started_at).total_seconds() * 1000
        )
        durations.append(duration_ms)
        records.append(
            {
                "submission_id": submission.submission_id,
                "student_id": submission.student_id,
                "duration_ms": duration_ms,
                "state": submission.state,
            }
        )

    average_ms = round(sum(durations) / len(durations), 2) if durations else 0

    return {"average_ms": average_ms, "records": records, "count": len(records)}


@router.get("/metrics")
def metrics(session: Session = Depends(get_session)) -> dict:
    """Operational health snapshot: throughput and error/review rates, all-time and last-24h.
    DB-derived rather than in-process counters, so it reflects real state across restarts and
    across however many worker processes are running -- there's no in-memory counter to lose.
    """
    now = datetime.now(timezone.utc)
    since_24h = now - timedelta(hours=24)

    total_submissions = session.exec(select(func.count()).select_from(Submission)).one()
    graded_submissions = session.exec(
        select(func.count()).select_from(Submission).where(Submission.state == ProcessingState.GRADED.value)
    ).one()
    submissions_24h = session.exec(
        select(func.count()).select_from(Submission).where(Submission.submitted_at >= since_24h)
    ).one()

    review_status_counts = dict(
        session.exec(select(ScanReview.status, func.count()).group_by(ScanReview.status)).all()
    )
    review_status_counts_24h = dict(
        session.exec(
            select(ScanReview.status, func.count())
            .where(ScanReview.created_at >= since_24h)
            .group_by(ScanReview.status)
        ).all()
    )
    total_reviews = sum(review_status_counts.values())

    # "Bad outcome" rate = scans that never became a gradeable Submission, as a fraction of
    # everything attempted (graded submissions + those failed/needs-review scans).
    attempted_24h = submissions_24h + sum(review_status_counts_24h.values())
    error_rate_24h = round(sum(review_status_counts_24h.values()) / attempted_24h, 4) if attempted_24h else None

    return {
        "generated_at": now.isoformat(),
        "submissions": {
            "total": total_submissions,
            "graded": graded_submissions,
            "last_24h": submissions_24h,
        },
        "scan_reviews": {
            "total": total_reviews,
            "by_status": review_status_counts,
            "last_24h_by_status": review_status_counts_24h,
        },
        "error_rate_last_24h": error_rate_24h,
    }
