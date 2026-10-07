#!/usr/bin/env python3
"""Generate worksheets: compose the questions, insert them into the database, and write the
printable PDFs -- in one step, so a printed sheet always carries an id the database holds.
Replaces the old repo's batch_generate_worksheets.py + bulk_insert_worksheets.py.

    python3 scripts/generate_worksheets.py --type homework --level A --language mr --count 50
    python3 scripts/generate_worksheets.py --type practice --level D3 --language en --count 5 --seed 42
    python3 scripts/generate_worksheets.py --type omr --start-id 990001 --count 100 --merge
    python3 scripts/generate_worksheets.py --type homework --level C --count 3 --dry-run

Ids: omit --start-id and the database assigns them; with --start-id the whole range must be free
(nothing is written otherwise -- reprint existing sheets with render_worksheet_pdf.py).
PDFs go to --out-dir (default <STORAGE_ROOT>/worksheets/), named like the old generator's
(<id>_<lang>_<type>_<LEVEL>[_page<n>].pdf); --merge also writes one print-ready PDF of the batch.
Every homework/practice worksheet records its seed in worksheet_metadata.generator, so it can be
regenerated identically; --seed fixes the batch's first seed (sheet i uses seed + i).
Run scripts/seed_skills.py once before the first homework/practice batch.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent))

from sqlmodel import Session

from app.core.config import settings
from app.db.session import engine
from app.generation.service import DEFAULT_OMR_QUESTION_COUNT, GenerationError, generate_worksheets


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate, insert and render worksheets.")
    parser.add_argument("--type", dest="worksheet_type", choices=["homework", "practice", "omr"], required=True)
    parser.add_argument("--level", help="Homework: A-G. Practice: theme + level, e.g. A1, S3, M4, D5. Not used for omr.")
    parser.add_argument("--language", choices=["en", "mr"], default="en")
    parser.add_argument("--count", type=int, default=1, help="How many worksheets (default 1).")
    parser.add_argument("--start-id", type=int, help="First worksheet id; the range must be free. Default: database-assigned.")
    parser.add_argument("--seed", type=int, help="Seed of the first worksheet (homework/practice). Default: random.")
    parser.add_argument("--title", help="Title override stored with the worksheet.")
    parser.add_argument("--questions", type=int, default=DEFAULT_OMR_QUESTION_COUNT,
                        help=f"OMR only: total questions across pages (default {DEFAULT_OMR_QUESTION_COUNT} = 2 pages).")
    parser.add_argument("--out-dir", type=Path, default=Path(settings.storage_root) / "worksheets")
    parser.add_argument("--merge", action="store_true", help="Also write one combined PDF of the whole batch for printing.")
    parser.add_argument("--dry-run", action="store_true", help="Generate and render, then roll back; write no files.")
    args = parser.parse_args()

    if args.worksheet_type == "omr" and args.level:
        parser.error("--level is not used for omr worksheets")

    with Session(engine) as session:
        try:
            result = generate_worksheets(
                session,
                worksheet_type=args.worksheet_type,
                level=args.level,
                language=args.language,
                count=args.count,
                start_id=args.start_id,
                seed=args.seed,
                title=args.title,
                omr_question_count=args.questions,
                output_dir=args.out_dir,
                merge=args.merge,
                dry_run=args.dry_run,
            )
        except GenerationError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1)

    ids = [w.worksheet_id for w in result.worksheets]
    pages = sum(len(w.pdfs) for w in result.worksheets)
    seed_note = f", seeds {result.batch_seed}-{result.batch_seed + len(ids) - 1}" if result.batch_seed is not None else ""
    if args.dry_run:
        print(f"Dry run: would generate {len(ids)} {args.worksheet_type} worksheet(s), {pages} page(s){seed_note}. Nothing written.")
        return
    print(f"Generated {len(ids)} {args.worksheet_type} worksheet(s), ids {ids[0]}-{ids[-1]}, {pages} page(s){seed_note}.")
    print(f"PDFs: {args.out_dir}")
    if result.merged_path:
        print(f"Print file: {result.merged_path}")


if __name__ == "__main__":
    main()
