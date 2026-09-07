"""Dashboard endpoints — health + latency, replacing the old systemctl check and in-memory
latency list with real DB state (per Phase 4.4 of the plan).
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from app.db.session import get_session
from app.models import Submission

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
