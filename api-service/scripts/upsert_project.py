#!/usr/bin/env python3
"""Create a project (or rename an existing one). Each project gets its own dashboard at
/admin/<code>/; PaperPlus and Navodaya are already created by the migration.

Example:
    python3 scripts/upsert_project.py --code navodaya --name "Navodaya"
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import Session

from app.db.session import engine
from app.domain.worksheet_import import upsert_project


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or rename a project.")
    parser.add_argument("--code", required=True, help="URL-safe code, e.g. 'navodaya' (lowercase letters, digits, '-').")
    parser.add_argument("--name", required=True, help="Display name, e.g. 'Navodaya'.")
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z0-9-]+", args.code):
        parser.error("--code must be lowercase letters, digits or '-' (it becomes part of the dashboard URL)")

    with Session(engine) as session:
        upsert_project(session, args.code, args.name.strip())
        session.commit()
    print(f"Project '{args.code}' ({args.name.strip()}) saved -- dashboard at /admin/{args.code}/")


if __name__ == "__main__":
    main()
