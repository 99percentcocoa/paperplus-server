#!/usr/bin/env python3
"""Register a basic_omr worksheet in the database, straight from its id -- no JSON file needed,
since an OMR sheet has no question content (just numbered A-D bubbles).

For NEW sheets use scripts/generate_worksheets.py --type omr, which registers and prints them in
one step. This script is for sheets that were printed by the pre-v2 generator but never
registered. Those sheets have 2 pages (Q1-39 and Q40-78), so the default here is 78 questions
(then seed answer keys with insert_code_answer_key.py):

    python3 scripts/insert_omr_worksheet.py --id 5001
    python3 scripts/insert_omr_worksheet.py --start-id 5001 --count 100     # ids 5001-5100

Ids that are already OMR worksheets are skipped, so re-running a range is safe. If any id in the
range is a non-OMR worksheet (homework/practice), nothing is inserted -- OMR sheets printed with
that id would be graded against the wrong questions. The range is inserted in one transaction.

(The pre-v2 blank_omr.json has only 39 questions -- inserting it with insert_single_worksheet.py
would leave page 2's questions ungradeable.)
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import Session, select

from app.db.session import engine
from app.domain.worksheet_import import insert_worksheet
from app.generation.composition import compose_omr_worksheet
from app.models import Worksheet
from shared.worksheet_templates import QUESTIONS_PER_PAGE


def main() -> None:
    per_page = QUESTIONS_PER_PAGE["basic_omr"]
    parser = argparse.ArgumentParser(description="Register blank basic_omr worksheets by id (one, or a range).")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--id", type=int, help="Worksheet id printed on the sheet.")
    which.add_argument("--start-id", type=int, help="First id of a range (use with --count).")
    parser.add_argument("--count", type=int, default=1, help="How many consecutive ids from --start-id (default 1).")
    parser.add_argument("--questions", type=int, default=2 * per_page,
                        help=f"Total questions across all pages (default {2 * per_page} = 2 pages of {per_page}).")
    parser.add_argument("--title", default="Basic OMR Sheet")
    parser.add_argument("--language", choices=["en", "mr"], default="en")
    parser.add_argument("--dry-run", action="store_true", help="Report what would happen without writing anything.")
    args = parser.parse_args()
    if args.questions < 1:
        parser.error("--questions must be at least 1")
    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.id is not None and args.count != 1:
        parser.error("--count goes with --start-id, not --id")
    first = args.id if args.id is not None else args.start_id
    ids = range(first, first + args.count)

    with Session(engine) as session:
        existing = {
            w.worksheet_id: w.worksheet_category
            for w in session.exec(select(Worksheet).where(Worksheet.worksheet_id.in_(list(ids)))).all()
        }
        conflicts = sorted(i for i, category in existing.items() if category != "omr")
        if conflicts:
            print(f"Error: these ids are already non-OMR worksheets, nothing inserted: {conflicts}", file=sys.stderr)
            raise SystemExit(1)
        to_insert = [i for i in ids if i not in existing]
        if existing:
            print(f"Skipping {len(existing)} id(s) that are already OMR worksheets: {_ranges(sorted(existing))}")
        if not to_insert:
            print("Nothing to insert.")
            return
        if args.dry_run:
            print(f"Dry run: would insert {len(to_insert)} OMR worksheet(s): {_ranges(to_insert)}")
            return

        payload = compose_omr_worksheet(args.questions, language=args.language, title=args.title)
        for worksheet_id in to_insert:
            insert_worksheet(
                session, payload, worksheet_id=worksheet_id, worksheet_category="omr",
                template_name="basic_omr", commit=False,
            )
        session.commit()
        pages = session.get(Worksheet, to_insert[0]).page_count
    print(
        f"Inserted {len(to_insert)} OMR worksheet(s) ({_ranges(to_insert)}): "
        f"{args.questions} questions, {pages} page(s) each."
    )


def _ranges(ids: list[int]) -> str:
    """[1, 2, 3, 7] -> "1-3, 7"."""
    parts, start = [], None
    for i, value in enumerate(ids):
        if start is None:
            start = value
        if i + 1 == len(ids) or ids[i + 1] != value + 1:
            parts.append(str(start) if start == value else f"{start}-{value}")
            start = None
    return ", ".join(parts)


if __name__ == "__main__":
    main()
