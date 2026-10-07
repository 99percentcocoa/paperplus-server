"""Grade a vision-service scan result against the answer key stored in question_options,
with an optional per-variant override from question_paper_variants.

Deliberately does NOT read omr_answer_sets/worksheet_json (the old blob-based answer key
storage) — answer_key resolution uses the normalized question_options table from Phase 1.
basic_omr worksheets print fixed A-F variants that each share one answer key across every
printed copy (no per-copy bubble shuffling), so question_paper_variants is just a per-variant
correct-label override — no position translation needed.
"""

from sqlmodel import Session, select

from app.models import Question, QuestionOption
from app.models.worksheet import OMRAnswerSet, QuestionPaperVariant
from shared.contracts import QuestionMark


def resolve_answer_key(session: Session, worksheet_id: int, question_paper_code: str | None = None) -> dict[int, str]:
    """Returns {question_index: correct_option_label} for a worksheet.
    Resolution order (first match wins):
    1. worksheet-specific question_paper_variant (for per-worksheet code overrides)
    2. code-level OMRAnswerSet (code-based key shared across all worksheets, worksheet_id=NULL)
    3. canonical question_options.is_correct (fallback for non-code worksheets)
    """
    questions = session.exec(select(Question).where(Question.worksheet_id == worksheet_id)).all()

    variants_by_question_id: dict[int, str] = {}
    if question_paper_code:
        variants = session.exec(
            select(QuestionPaperVariant).where(
                QuestionPaperVariant.worksheet_id == worksheet_id,
                QuestionPaperVariant.question_paper_code == question_paper_code,
            )
        ).all()
        variants_by_question_id = {v.question_id: v.correct_option_label for v in variants}

    # Check for code-level answer set (shared across all worksheets, worksheet_id=NULL)
    code_level_answer_key: dict[int, str] = {}
    if question_paper_code:
        omr_set = session.exec(
            select(OMRAnswerSet).where(
                OMRAnswerSet.question_paper_code == question_paper_code,
                OMRAnswerSet.worksheet_id.is_(None),
            )
        ).first()
        if omr_set and omr_set.answer_key_json:
            code_level_answer_key = {int(k): v for k, v in omr_set.answer_key_json.items()}

    answer_key: dict[int, str] = {}
    for question in questions:
        if question.index is None:
            continue

        if question.question_id in variants_by_question_id:
            answer_key[question.index] = variants_by_question_id[question.question_id]
            continue

        if question.index in code_level_answer_key:
            answer_key[question.index] = code_level_answer_key[question.index]
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
