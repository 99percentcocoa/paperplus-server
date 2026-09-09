#!/usr/bin/env python3
"""Insert a code-level answer key into omr_answer_sets (shared across all worksheets using that code).

Code-level keys are independent of worksheet_id and apply globally whenever that question_paper_code
is scanned. This is the recommended way to seed answer keys for basic_omr worksheets where all copies
of code "A" have the same answers, all copies of code "D" have the same answers, etc.

Example:
    python3 scripts/insert_code_answer_key.py --code D --answer-key A,B,C,D,A,B,C,D,... \\
      --template-name basic_omr
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import Session, select

from app.db.session import engine
from app.models.worksheet import OMRAnswerSet


def parse_answer_key(value: str) -> list[str]:
    tokens = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not tokens:
        raise ValueError("--answer-key must not be empty")
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Insert a code-level answer key (shared across all worksheets for that code)."
    )
    parser.add_argument(
        "--code", required=True, help="Question-paper code, a single letter A-F."
    )
    parser.add_argument(
        "--answer-key",
        required=True,
        help="Comma-separated correct option labels, one per question index in order (starting from 1), e.g. A,B,C,D",
    )
    parser.add_argument(
        "--template-name",
        default="basic_omr",
        help="Template name (default: basic_omr). Used for lookup; must match the template_name in your worksheets.",
    )
    args = parser.parse_args()

    answer_key_list = parse_answer_key(args.answer_key)
    # Convert to {question_index: label} dict (1-indexed)
    answer_key_dict = {str(i + 1): label for i, label in enumerate(answer_key_list)}

    with Session(engine) as session:
        existing = session.exec(
            select(OMRAnswerSet).where(
                OMRAnswerSet.question_paper_code == args.code.upper(),
                OMRAnswerSet.worksheet_id.is_(None),
            )
        ).first()

        if existing:
            existing.answer_key_json = answer_key_dict
            session.add(existing)
            print(
                f"Updated existing code-level answer key for code={args.code.upper()} "
                f"template={args.template_name} ({len(answer_key_dict)} questions)"
            )
        else:
            new_row = OMRAnswerSet(
                template_name=args.template_name,
                question_paper_code=args.code.upper(),
                worksheet_id=None,  # NULL = code-level (global) key
                answer_key_json=answer_key_dict,
            )
            session.add(new_row)
            print(
                f"Inserted new code-level answer key for code={args.code.upper()} "
                f"template={args.template_name} ({len(answer_key_dict)} questions)"
            )

        session.commit()


if __name__ == "__main__":
    main()
