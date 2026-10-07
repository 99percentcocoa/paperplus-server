#!/usr/bin/env python3
"""Reprint worksheets that are already in the database, from their stored questions -- no JSON
files needed. Works for anything inserted with generate_worksheets.py, the insertion scripts, or
the legacy migration.

    python3 scripts/render_worksheet_pdf.py --id 4920
    python3 scripts/render_worksheet_pdf.py --start-id 4920 --count 900 --merge
    python3 scripts/render_worksheet_pdf.py --id 4920 --export-json     # also write <id>.json

--export-json writes the worksheet JSON in the old generator's files/json shape.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT.parent))

from sqlmodel import Session

from app.core.config import settings
from app.db.session import engine
from app.generation.service import GenerationError, render_existing
from app.models import Worksheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Render PDFs for worksheets already in the database.")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--id", type=int, help="One worksheet id.")
    which.add_argument("--start-id", type=int, help="First id of a range (use with --count).")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path(settings.storage_root) / "worksheets")
    parser.add_argument("--merge", action="store_true", help="Also write one combined PDF for printing.")
    parser.add_argument("--export-json", action="store_true", help="Also write each worksheet's JSON next to its PDF.")
    args = parser.parse_args()
    if args.id is not None and args.count != 1:
        parser.error("--count goes with --start-id, not --id")

    first = args.id if args.id is not None else args.start_id
    ids = list(range(first, first + args.count))
    with Session(engine) as session:
        try:
            result = render_existing(session, ids, args.out_dir, merge=args.merge)
        except GenerationError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1)
        if args.export_json:
            for worksheet_id in ids:
                worksheet = session.get(Worksheet, worksheet_id)
                path = args.out_dir / f"{worksheet_id}_{worksheet.lang or 'en'}.json"
                path.write_text(json.dumps(worksheet.worksheet_json, indent=2, ensure_ascii=False), encoding="utf-8")

    pages = sum(len(w.pdfs) for w in result.worksheets)
    print(f"Rendered {len(ids)} worksheet(s), {pages} page(s) -> {args.out_dir}")
    if result.merged_path:
        print(f"Print file: {result.merged_path}")


if __name__ == "__main__":
    main()
