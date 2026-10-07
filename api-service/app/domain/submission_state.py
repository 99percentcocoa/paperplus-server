"""Explicit state transitions for Submission.state — never assign submission.state directly elsewhere."""

from datetime import datetime, timezone

from sqlmodel import Session

from app.models.submission import ProcessingEvent, ProcessingState, Submission

ALLOWED_TRANSITIONS: dict[ProcessingState, set[ProcessingState]] = {
    ProcessingState.UPLOADED: {ProcessingState.PREPROCESSING, ProcessingState.FAILED},
    ProcessingState.PREPROCESSING: {ProcessingState.DEWARPED, ProcessingState.FAILED},
    ProcessingState.DEWARPED: {ProcessingState.REGISTERING, ProcessingState.FAILED},
    ProcessingState.REGISTERING: {ProcessingState.SCORING, ProcessingState.FAILED},
    ProcessingState.SCORING: {ProcessingState.GRADED, ProcessingState.FAILED},
    ProcessingState.GRADED: set(),
    ProcessingState.FAILED: set(),
}


class InvalidStateTransition(Exception):
    pass


def _transition(
    session: Session,
    submission: Submission,
    new_state: ProcessingState,
    service_name: str,
    detail: dict | None = None,
    correlation_id: str | None = None,
) -> Submission:
    current_state = ProcessingState(submission.state)
    if new_state not in ALLOWED_TRANSITIONS.get(current_state, set()):
        raise InvalidStateTransition(f"cannot transition from {current_state} to {new_state}")

    submission.state = new_state.value
    if new_state == ProcessingState.PREPROCESSING and submission.processing_started_at is None:
        submission.processing_started_at = datetime.now(timezone.utc)
    if new_state in (ProcessingState.GRADED, ProcessingState.FAILED):
        submission.processing_completed_at = datetime.now(timezone.utc)

    session.add(submission)
    session.add(
        ProcessingEvent(
            submission_id=submission.submission_id,
            state=new_state.value,
            service_name=service_name,
            correlation_id=correlation_id,
            detail=detail or {},
        )
    )
    session.commit()
    session.refresh(submission)
    return submission


def transition_to_preprocessing(session: Session, submission: Submission, service_name: str, correlation_id: str | None = None) -> Submission:
    return _transition(session, submission, ProcessingState.PREPROCESSING, service_name, correlation_id=correlation_id)


def transition_to_dewarped(session: Session, submission: Submission, service_name: str, detail: dict | None = None, correlation_id: str | None = None) -> Submission:
    return _transition(session, submission, ProcessingState.DEWARPED, service_name, detail=detail, correlation_id=correlation_id)


def transition_to_registering(session: Session, submission: Submission, service_name: str, detail: dict | None = None, correlation_id: str | None = None) -> Submission:
    return _transition(session, submission, ProcessingState.REGISTERING, service_name, detail=detail, correlation_id=correlation_id)


def transition_to_scoring(session: Session, submission: Submission, service_name: str, correlation_id: str | None = None) -> Submission:
    return _transition(session, submission, ProcessingState.SCORING, service_name, correlation_id=correlation_id)


def transition_to_graded(session: Session, submission: Submission, service_name: str, detail: dict | None = None, correlation_id: str | None = None) -> Submission:
    return _transition(session, submission, ProcessingState.GRADED, service_name, detail=detail, correlation_id=correlation_id)


def transition_to_failed(session: Session, submission: Submission, service_name: str, reason: str, correlation_id: str | None = None) -> Submission:
    return _transition(session, submission, ProcessingState.FAILED, service_name, detail={"reason": reason}, correlation_id=correlation_id)
