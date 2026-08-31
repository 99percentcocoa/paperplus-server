#!/usr/bin/env python3
"""Create sample OMR answer-key variants keyed by question-paper code.

Example:
    python3 sample_insert_omr_answer_variants.py --template basic_omr --code A --worksheet-id 9001 --answer-key A,B,C,D
    python3 sample_insert_omr_answer_variants.py --template basic_omr --code B --worksheet-id 9001 --answer-key D,C,B,A
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from db.worksheets import save_omr_answer_key


def parse_answer_key(value: str) -> list[str]:
    tokens = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not tokens:
        raise ValueError("answer-key must not be empty")
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Insert sample answer-key variants for OMR paper codes.")
    parser.add_argument("--template", required=True, help="Template name, e.g. basic_omr")
    parser.add_argument("--code", required=True, help="Question-paper code, e.g. A or B")
    parser.add_argument("--worksheet-id", type=int, default=None, help="Optional worksheet_id to link to this answer variant")
    parser.add_argument("--answer-key", required=True, help="Comma-separated correct options, e.g. A,B,C,D")
    args = parser.parse_args()

    answer_key = parse_answer_key(args.answer_key)
    inserted_id = save_omr_answer_key(
        template_name=args.template,
        question_paper_code=args.code,
        answer_key=answer_key,
        worksheet_id=args.worksheet_id,
    )
    print(f"Inserted OMR answer variant id={inserted_id} for template={args.template} code={args.code}")
    print(f"answer_key={answer_key}")


if __name__ == "__main__":
    main()
