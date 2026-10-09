#!/usr/bin/env python3
"""Re-run scans that failed only because vision-service was down, overloaded or crashed.

Selects open failed reviews whose error is a vision-service timeout / connection error / dropped
connection / 5xx (NOT "tags not detected" 422s, which are bad photos) and sends each stored photo
through the normal grading path. Grades silently: no WhatsApp message is sent to anyone. A student
who already has a submission for the scanned worksheet is skipped, never overwritten.

Dry run by default (database reads only; vision-service is not called):
    python scripts/reprocess_failed_scans.py
    python scripts/reprocess_failed_scans.py --project paperplus --limit 10
Actually reprocess:
    python scripts/reprocess_failed_scans.py --apply --limit 5
    python scripts/reprocess_failed_scans.py --apply
"""

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import Session

from app.db.session import engine
from app.services.reprocess import find_infra_failed_reviews, reprocess_review
from app.services.vision_client import HTTPVisionClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run scans that failed due to vision-service outages.")
    parser.add_argument("--apply", action="store_true", help="Actually reprocess (default: list what would be).")
    parser.add_argument("--project", help="Only scans of this project code (default: all).")
    parser.add_argument("--limit", type=int, help="Process at most this many (oldest first).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    with Session(engine) as session:
        candidates = find_infra_failed_reviews(session, project_code=args.project, limit=args.limit)
        print(f"{len(candidates)} failed scan(s) caused by vision-service errors.")

        if not args.apply:
            for review, scan in candidates:
                print(f"  review {review.review_id}  {review.created_at:%Y-%m-%d %H:%M}  "
                      f"from {scan.from_number or '-'}  {(review.error_reason or '')[:70]}")
            if candidates:
                print("\nDry run only. Re-run with --apply to reprocess these.")
            return

        vision_client = HTTPVisionClient()
        counts: dict[str, int] = {}
        for review, scan in candidates:
            review_id = review.review_id
            try:
                outcome = reprocess_review(session, vision_client, review, scan)
            except Exception as exc:  # one bad scan must not stop the rest
                session.rollback()
                print(f"  review {review_id}: ERROR {type(exc).__name__}: {exc}")
                counts["error"] = counts.get("error", 0) + 1
                continue
            counts[outcome.status] = counts.get(outcome.status, 0) + 1
            print(f"  review {outcome.review_id}: {outcome.status} {outcome.detail}")

        print("\nSummary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
