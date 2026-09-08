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

from sqlmodel import Session, select

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
from app.models import Attempt, Question, Student, Submission, Worksheet
from app.models.submission import ProcessingState
from app.services.communication import CommunicationClient
from app.services.vision_client import VisionClient

logger = logging.getLogger(__name__)

MESSAGES = {
    "vision_failed": (
        "The worksheet could not be read properly. Please try again. \u27f3 \n"
        "\u0915\u093e\u0930\u094d\u092f\u092a\u0924\u094d\u0930\u093f\u0915\u093e \u0928\u0940\u091f \u0935\u093e\u091a\u0924\u093e \u0906\u0932\u0940 \u0928\u093e\u0939\u0940. \u0915\u0943\u092a\u092f\u093e \u092a\u0941\u0928\u094d\u0939\u093e \u092a\u094d\u0930\u092f\u0924\u094d\u0928 \u0915\u0930\u093e. \u27f3"
    ),
    "invalid_student": (
        "Roll number not recognized. Please check and try again. \u27f3 \n"
        "\u0930\u094b\u0932 \u0928\u0902\u092c\u0930 \u0913\u0933\u0916\u0924\u093e \u0906\u0932\u093e \u0928\u093e\u0939\u0940. \u0915\u0943\u092a\u092f\u093e \u0924\u092a\u093e\u0938\u0942\u0928 \u092a\u0930\u0924 \u092a\u093e\u0920\u0935\u093e. \u27f3"
    ),
    "invalid_worksheet": (
        "This worksheet could not be processed. Please try again. \u27f3 \n"
        "\u0939\u0940 \u0915\u093e\u0930\u094d\u092f\u092a\u0924\u094d\u0930\u093f\u0915\u093e \u0924\u092a\u093e\u0938\u0924\u093e \u0906\u0932\u0940 \u0928\u093e\u0939\u0940. \u0915\u0943\u092a\u092f\u093e \u092a\u0930\u0924 \u092a\u094d\u0930\u092f\u0924\u094d\u0928 \u0915\u0930\u093e. \u27f3"
    ),
    "invalid_answer_key": (
        "This worksheet is not ready to be graded yet. Please contact your facilitator. \u27f3 \n"
        "\u0939\u0940 \u0915\u093e\u0930\u094d\u092f\u092a\u0924\u094d\u0930\u093f\u0915\u093e \u0924\u092a\u093e\u0938\u0923\u094d\u092f\u093e\u0938\u093e\u0920\u0940 \u0924\u092f\u093e\u0930 \u0928\u093e\u0939\u0940. \u0915\u0943\u092a\u092f\u093e \u0906\u092a\u0932\u094d\u092f\u093e \u0938\u0939\u0935\u093e\u092f\u0915\u093e\u0936\u0940 \u0938\u0902\u092a\u0930\u094d\u0915 \u0938\u093e\u0927\u093e. \u27f3"
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
    try:
        result = vision_client.process(image_path, correlation_id)
    except VisionClientError:
        logger.exception("vision-service call failed for correlation_id=%s", correlation_id)
        comm_client.send_message(from_number, MESSAGES["vision_failed"])
        return

    logger.info(
        "vision-service result for correlation_id=%s: worksheet_id=%s page_no=%s "
        "template_name=%s roll_number=%s roll_number_confidence=%s question_paper_code=%s "
        "question_marks_count=%s",
        correlation_id, result.worksheet_id, result.page_no, result.template_name,
        result.roll_number, result.roll_number_confidence, result.question_paper_code,
        len(result.question_marks),
    )

    try:
        student = _validate_student(session, result.roll_number)
        worksheet = _validate_worksheet(session, result.worksheet_id)
        answer_key = _validate_answer_key(session, worksheet.worksheet_id, result.question_paper_code)
    except InvalidStudentError:
        comm_client.send_message(from_number, MESSAGES["invalid_student"])
        return
    except InvalidWorksheetError:
        comm_client.send_message(from_number, MESSAGES["invalid_worksheet"])
        return
    except InvalidAnswerKeyError:
        logger.error(
            "No resolvable answer key for correlation_id=%s: worksheet_id=%s question_paper_code=%r "
            "(no question_paper_variant seeded for this code, and no canonical question_options "
            "fallback either) -- refusing to grade rather than silently score 0.",
            correlation_id, worksheet.worksheet_id, result.question_paper_code,
        )
        comm_client.send_message(from_number, MESSAGES["invalid_answer_key"])
        return

    scanned_answers, _scanned_score = grade_marks(result.question_marks, answer_key)
    page_range = resolve_page_range(session, worksheet.worksheet_id, result.page_no)

    submission, answers_payload, score = _create_or_overwrite_submission(
        session, student, worksheet, from_number, scanned_answers, page_range
    )

    transition_to_preprocessing(session, submission, service_name="api-service", correlation_id=correlation_id)
    transition_to_dewarped(session, submission, service_name="api-service", correlation_id=correlation_id,
                            detail={"template_name": result.template_name})
    transition_to_registering(session, submission, service_name="api-service", correlation_id=correlation_id,
                               detail={"roll_number": result.roll_number})
    transition_to_scoring(session, submission, service_name="api-service", correlation_id=correlation_id)

    attempts_count = _insert_attempts(session, student.student_id, submission, worksheet.worksheet_id, answers_payload)
    affected_skills = _affected_skills(session, worksheet.worksheet_id, answers_payload)
    for skill_code in affected_skills:
        recalculate_skill_mastery(session, student.student_id, skill_code)
    level_update = evaluate_and_update_level(session, student.student_id)

    transition_to_graded(
        session, submission, service_name="api-service", correlation_id=correlation_id,
        detail={"score": score, "attempts_count": attempts_count, "level_update": level_update},
    )

    comm_client.send_message(from_number, f"Your marks: {score}/{len(answers_payload)}")


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


def _insert_attempts(session: Session, student_id: str, submission: Submission, worksheet_id: int, answers_payload: list[dict]) -> int:
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


def _affected_skills(session: Session, worksheet_id: int, answers_payload: list[dict]) -> set[str]:
    questions_by_index = {
        q.index: q for q in session.exec(select(Question).where(Question.worksheet_id == worksheet_id)).all()
    }
    return {
        questions_by_index[ans["question_index"]].skill_code
        for ans in answers_payload
        if ans["question_index"] in questions_by_index
    }
