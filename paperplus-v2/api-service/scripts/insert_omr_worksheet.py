#!/usr/bin/env python3
"""Register a basic_omr worksheet in the database, straight from its id -- no JSON file needed,
since an OMR sheet has no question content (just numbered A-D bubbles).

The printed PDF still comes from the old repo's generator, which prints the worksheet id into the
sheet's tags. It prints 2 pages (Q1-39 and Q40-78), so the default here is 78 questions:

    # old repo root: print the sheet
    python3 batch_generate_worksheets.py --type omr --worksheet-id 5001
    # here: register it (then seed answer keys with insert_code_answer_key.py)
    python3 scripts/insert_omr_worksheet.py --id 5001

(The old blank_omr.json has only 39 questions -- inserting it with insert_single_worksheet.py
would leave page 2's questions ungradeable.)
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import Session

from app.db.session import engine
from app.domain.worksheet_import import insert_worksheet
from app.models import Worksheet
from shared.worksheet_templates import QUESTIONS_PER_PAGE


def blank_omr_json(question_count: int, title: str, language: str) -> dict:
    return {
        "title": title,
        "worksheet_category": "omr",
        "template_name": "basic_omr",
        "language": language,
        "questions": [
            {"index": i, "question_text": "", "options": ["", "", "", ""], "correct_option": ""}
            for i in range(1, question_count + 1)
        ],
    }


def main() -> None:
    per_page = QUESTIONS_PER_PAGE["basic_omr"]
    parser = argparse.ArgumentParser(description="Register a blank basic_omr worksheet by id.")
    parser.add_argument("--id", type=int, required=True, help="Worksheet id printed on the sheet.")
    parser.add_argument("--questions", type=int, default=2 * per_page,
                        help=f"Total questions across all pages (default {2 * per_page} = 2 pages of {per_page}).")
    parser.add_argument("--title", default="Basic OMR Sheet")
    parser.add_argument("--language", choices=["en", "mr"], default="en")
    args = parser.parse_args()
    if args.questions < 1:
        parser.error("--questions must be at least 1")

    with Session(engine) as session:
        if session.get(Worksheet, args.id) is not None:
            print(f"Error: worksheet {args.id} already exists.", file=sys.stderr)
            raise SystemExit(1)
        result = insert_worksheet(
            session, blank_omr_json(args.questions, args.title, args.language),
            worksheet_id=args.id, worksheet_category="omr", template_name="basic_omr",
        )
        pages = session.get(Worksheet, args.id).page_count
    print(f"Inserted OMR worksheet {result['worksheet_id']}: {len(result['question_ids'])} questions, {pages} page(s).")


if __name__ == "__main__":
    main()
