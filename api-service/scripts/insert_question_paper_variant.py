#!/usr/bin/env python3
"""Seed a question_paper_variants answer key for one basic_omr question-paper code.

Every printed copy of a given code (A-F) shares one fixed answer key, so this takes one
correct label per question index in order (not a per-copy bubble-position shuffle).

Example:
    python3 scripts/insert_question_paper_variant.py --worksheet-id 9001 --code A --answer-key A,B,C,D,...
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import Session

from app.db.session import engine
from app.domain.worksheet_import import insert_question_paper_variant


def parse_answer_key(value: str) -> list[str]:
    tokens = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not tokens:
        raise ValueError("--answer-key must not be empty")
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Insert a question-paper-code answer-key variant for a worksheet.")
    parser.add_argument("--worksheet-id", type=int, required=True, help="Worksheet id these questions belong to.")
    parser.add_argument("--code", required=True, help="Question-paper code, a single letter A-F.")
    parser.add_argument("--answer-key", required=True, help="Comma-separated correct option labels, one per question index in order, e.g. A,B,C,D")
    args = parser.parse_args()

    answer_key = parse_answer_key(args.answer_key)

    with Session(engine) as session:
        variant_ids = insert_question_paper_variant(session, args.worksheet_id, args.code, answer_key)

    print(f"Inserted/updated {len(variant_ids)} question_paper_variants rows for worksheet_id={args.worksheet_id} code={args.code.upper()}")


if __name__ == "__main__":
    main()
