"""Grade a vision-service scan result against the answer key stored in question_options.

Deliberately does NOT read omr_answer_sets/worksheet_json (the old blob-based answer key
storage) — answer_key resolution uses the normalized question_options table from Phase 1.
question_paper_variants (per-printed-variant option shuffling) is not yet wired in; all
questions are graded against their canonical option order for now.
"""

from sqlmodel import Session, select

from app.models import Question, QuestionOption
from shared.contracts import QuestionMark


def resolve_answer_key(session: Session, worksheet_id: int) -> dict[int, str]:
    """Returns {question_index: correct_option_label} for a worksheet."""
    questions = session.exec(select(Question).where(Question.worksheet_id == worksheet_id)).all()

    answer_key: dict[int, str] = {}
    for question in questions:
        if question.index is None:
            continue
        correct_option = session.exec(
            select(QuestionOption).where(
                QuestionOption.question_id == question.question_id,
                QuestionOption.is_correct == True,  # noqa: E712 - SQLAlchemy needs `==`, not `is`
            )
        ).first()
        if correct_option is not None:
            answer_key[question.index] = correct_option.option_label

    return answer_key


def grade_marks(question_marks: list[QuestionMark], answer_key: dict[int, str]) -> tuple[list[dict], int]:
    """Returns (answers_payload, score) — one dict per question, plus total correct count."""
    answers_payload = []
    score = 0

    for mark in question_marks:
        correct_label = answer_key.get(mark.question_index)
        is_correct = bool(correct_label) and mark.marked_option == correct_label
        if is_correct:
            score += 1
        answers_payload.append(
            {
                "question_index": mark.question_index,
                "selected_option": mark.marked_option or "",
                "is_correct": is_correct,
            }
        )

    return answers_payload, score
