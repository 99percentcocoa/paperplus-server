#!/usr/bin/env python3
"""Seed the canonical skill catalog (app/data/skills.json) into the skills table.

Run this once before inserting worksheets whose questions reference skill codes (e.g. "1A",
"2S2") that must already exist for the questions.skill_code foreign key to succeed.

Example:
    python3 scripts/seed_skills.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import Session

from app.db.session import engine
from app.domain.worksheet_import import seed_skills


def main() -> None:
    with Session(engine) as session:
        count = seed_skills(session)
    print(f"Seeded/updated {count} skills.")


if __name__ == "__main__":
    main()
