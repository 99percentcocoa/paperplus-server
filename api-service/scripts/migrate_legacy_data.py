#!/usr/bin/env python3
"""Migrate the legacy PaperPlus database into the v2 database.

SAFETY MODEL
  * The legacy database is opened read-only (default_transaction_read_only=on) -- point it at a
    scratch restore of a pg_dump, never at the live database.
  * Everything happens in ONE transaction on the target. --dry-run runs the full migration and
    verification, prints the report, then rolls everything back (including --wipe-target).
  * The target must be empty unless you pass --wipe-target NAME, which TRUNCATEs every data table
    (except alembic_version) first; NAME must equal the target database's name, as a typo guard.
  * The migration is only committed if verification found no problems.

Typical use (local):
    createdb paperplus_legacy_restore && psql -d paperplus_legacy_restore -f dump.sql
    export LEGACY_DATABASE_URL=postgresql+psycopg://USER:PASS@localhost/paperplus_legacy_restore
    python scripts/migrate_legacy_data.py --dry-run --wipe-target paperplus_v2
    python scripts/migrate_legacy_data.py --wipe-target paperplus_v2

Target defaults to DATABASE_URL from the app settings (api-service/.env).
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlmodel import Session

from app.core.config import settings
from app.domain.legacy_migration import MigrationError, migrate, reset_sequences, target_is_empty, wipe_target


def describe(url_text: str) -> str:
    url = make_url(url_text)
    return f"{url.host or 'local socket'}:{url.port or ''}/{url.database}"


def print_report(report) -> None:
    print("\nRow counts (legacy -> target):")
    for table, (legacy_count, target_count) in report.counts.items():
        flag = "" if legacy_count == target_count else "   <-- MISMATCH"
        print(f"  {table:<24}{legacy_count:>9,} -> {target_count:>9,}{flag}")
    if report.notes:
        print("\nNotes:")
        for note in report.notes:
            print(f"  - {note}")
    if report.problems:
        shown = report.problems[:20]
        print(f"\nPROBLEMS ({len(report.problems)}):")
        for problem in shown:
            print(f"  ! {problem}")
        if len(report.problems) > len(shown):
            print(f"  ... and {len(report.problems) - len(shown)} more")


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate the legacy PaperPlus database into v2.")
    parser.add_argument("--legacy-url", default=os.environ.get("LEGACY_DATABASE_URL"),
                        help="SQLAlchemy URL of the (scratch, restored) legacy DB. Default: $LEGACY_DATABASE_URL")
    parser.add_argument("--target-url", default=settings.database_url, help="v2 database URL. Default: app DATABASE_URL")
    parser.add_argument("--dry-run", action="store_true", help="Run + verify, then roll back everything.")
    parser.add_argument("--wipe-target", metavar="DBNAME",
                        help="TRUNCATE all target data tables first. DBNAME must match the target database name.")
    args = parser.parse_args()

    if not args.legacy_url:
        parser.error("--legacy-url (or $LEGACY_DATABASE_URL) is required")
    legacy_url, target_url = make_url(args.legacy_url), make_url(args.target_url)
    if (legacy_url.host, legacy_url.port, legacy_url.database) == (target_url.host, target_url.port, target_url.database):
        parser.error("legacy and target are the same database")
    if args.wipe_target and args.wipe_target != target_url.database:
        parser.error(f"--wipe-target {args.wipe_target!r} does not match the target database {target_url.database!r}")

    print(f"legacy (read-only): {describe(args.legacy_url)}")
    print(f"target:             {describe(args.target_url)}   {'[DRY RUN]' if args.dry_run else '[WILL COMMIT]'}")

    legacy_engine = create_engine(args.legacy_url, connect_args={"options": "-c default_transaction_read_only=on"})
    target_engine = create_engine(args.target_url)
    started = time.time()

    with legacy_engine.connect() as legacy, Session(target_engine) as session:
        try:
            if args.wipe_target:
                tables = wipe_target(session)
                print(f"wiped {len(tables)} target tables (inside the transaction)")
            occupied = target_is_empty(session)
            if occupied:
                print(f"\nRefusing to migrate: target already has rows in {', '.join(occupied)}. "
                      "Use --wipe-target DBNAME to overwrite.", file=sys.stderr)
                return 2

            report = migrate(legacy, session)
        except MigrationError as exc:
            session.rollback()
            print(f"\nMigration aborted, nothing written: {exc}", file=sys.stderr)
            return 1

        print_report(report)
        print(f"\n(took {time.time() - started:.0f}s)")

        if not report.ok:
            session.rollback()
            print("\nVerification failed -> rolled back, target unchanged.", file=sys.stderr)
            return 1
        if args.dry_run:
            session.rollback()
            print("\nDry run OK -> rolled back, target unchanged.")
            return 0

        session.commit()
        reset_sequences(session)
        print("\nCommitted. Sequences reset.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
