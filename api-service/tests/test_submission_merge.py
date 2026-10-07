from app.domain.submission_merge import merge_page_answers


def test_merge_with_no_page_range_replaces_only_new_indices():
    existing = [
        {"question_index": 1, "selected_option": "A", "is_correct": True},
        {"question_index": 2, "selected_option": "B", "is_correct": False},
    ]
    new = [{"question_index": 2, "selected_option": "C", "is_correct": True}]

    merged = merge_page_answers(existing, new, page_range=None)

    assert merged == [
        {"question_index": 1, "selected_option": "A", "is_correct": True},
        {"question_index": 2, "selected_option": "C", "is_correct": True},
    ]


def test_merge_page_range_preserves_other_pages():
    # Page 1 (1-2) already graded; this scan covers page 2 (3-4).
    existing = [
        {"question_index": 1, "selected_option": "A", "is_correct": True},
        {"question_index": 2, "selected_option": "B", "is_correct": False},
    ]
    new = [
        {"question_index": 3, "selected_option": "A", "is_correct": True},
        {"question_index": 4, "selected_option": "B", "is_correct": True},
    ]

    merged = merge_page_answers(existing, new, page_range=(3, 4))

    assert [a["question_index"] for a in merged] == [1, 2, 3, 4]
    assert merged[0] == existing[0]
    assert merged[1] == existing[1]


def test_merge_page_range_fills_gaps_as_unanswered_not_stale_data():
    # Previous scan of THIS SAME page had an answer for question 2; this rescan only
    # detected question 1 (e.g. a row tag failed to decode) -> question 2 must become
    # explicitly unanswered, not silently keep the old "B" answer.
    existing = [
        {"question_index": 1, "selected_option": "A", "is_correct": True},
        {"question_index": 2, "selected_option": "B", "is_correct": False},
    ]
    new = [{"question_index": 1, "selected_option": "A", "is_correct": True}]

    merged = merge_page_answers(existing, new, page_range=(1, 2))

    assert merged == [
        {"question_index": 1, "selected_option": "A", "is_correct": True},
        {"question_index": 2, "selected_option": "", "is_correct": False},
    ]


def test_merge_with_no_existing_submission():
    new = [{"question_index": 1, "selected_option": "A", "is_correct": True}]
    merged = merge_page_answers(None, new, page_range=(1, 2))
    assert merged == [
        {"question_index": 1, "selected_option": "A", "is_correct": True},
        {"question_index": 2, "selected_option": "", "is_correct": False},
    ]
