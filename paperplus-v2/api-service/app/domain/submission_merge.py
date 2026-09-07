"""Page-aware merge of multi-page OMR submissions (Option C).

Unlike the old system's `merge_answers_json` (triggered only when
`worksheet_category == "omr"` and some `question_index > 39` — a hardcoded assumption
about page layout), this uses the real `worksheet_pages.first_question_index` /
`last_question_index` range for the page that was actually scanned. The scanned page's
whole expected range always fully replaces any prior data for those indices — including
filling gaps with an explicit "unanswered" entry — so a partial rescan failure can never
silently resurrect stale answers from a previous scan of the *same* page. Other pages'
answers from a prior submission are left untouched, which is what makes multi-page
worksheets merge correctly across submissions.
"""

from sqlmodel import Session, select

from app.models.worksheet import WorksheetPage


def resolve_page_range(session: Session, worksheet_id: int, page_no: int | None) -> tuple[int, int] | None:
    """Returns (first_question_index, last_question_index) for the given page, if known."""
    if page_no is None:
        return None
    page = session.exec(
        select(WorksheetPage).where(WorksheetPage.worksheet_id == worksheet_id, WorksheetPage.page_no == page_no)
    ).first()
    if page is None:
        return None
    return page.first_question_index, page.last_question_index


def merge_page_answers(
    existing_answers: list[dict] | None,
    new_answers: list[dict],
    page_range: tuple[int, int] | None,
) -> list[dict]:
    """Merge a freshly-graded page's answers into a prior submission's full answer list."""
    existing_by_index = {
        a["question_index"]: a for a in (existing_answers or []) if isinstance(a, dict) and "question_index" in a
    }
    new_by_index = {a["question_index"]: a for a in new_answers}

    if page_range is None:
        # No worksheet_pages metadata (e.g. single-page worksheet): the new scan's own
        # detected indices replace any existing entries at those indices; everything else
        # from the prior submission (if any) is left as-is.
        merged = {**existing_by_index, **new_by_index}
        return [merged[i] for i in sorted(merged)]

    first, last = page_range
    page_slice = {
        idx: new_by_index.get(idx, {"question_index": idx, "selected_option": "", "is_correct": False})
        for idx in range(first, last + 1)
    }

    merged = dict(existing_by_index)
    merged.update(page_slice)
    return [merged[i] for i in sorted(merged)]
