from app.domain.grading import grade_marks
from shared.contracts import QuestionMark


def test_grade_marks_computes_score_and_payload():
    answer_key = {1: "A", 2: "B", 3: "C"}
    marks = [
        QuestionMark(question_index=1, marked_option="A", confidence=0.9),  # correct
        QuestionMark(question_index=2, marked_option="C", confidence=0.9),  # incorrect
        QuestionMark(question_index=3, marked_option=None, confidence=0.0),  # unanswered
    ]

    payload, score = grade_marks(marks, answer_key)

    assert score == 1
    assert payload == [
        {"question_index": 1, "selected_option": "A", "is_correct": True},
        {"question_index": 2, "selected_option": "C", "is_correct": False},
        {"question_index": 3, "selected_option": "", "is_correct": False},
    ]


def test_grade_marks_missing_answer_key_entry_is_incorrect():
    payload, score = grade_marks([QuestionMark(question_index=5, marked_option="A", confidence=0.9)], {})
    assert score == 0
    assert payload[0]["is_correct"] is False
