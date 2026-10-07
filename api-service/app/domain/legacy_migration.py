"""One-off migration of the legacy (Flask-era) PaperPlus database into the v2 schema.

The legacy database is read through a read-only connection (typically a scratch database restored
from a pg_dump, never the live one); everything is written to the v2 target through one SQLAlchemy
session and NOT committed here -- the caller decides between commit and rollback, which is what
makes a real dry-run possible (even the optional wipe is transactional).

What maps to what (see scripts/migrate_legacy_data.py for the CLI):
  schools, skills, students, student_skill_mastery  -> copied as-is (IDs preserved)
  worksheets + questions                            -> app.domain.worksheet_import.insert_worksheet
      (the same code path used for new worksheets: derives question_options from `correct_option`,
      sets template_id/page_count), worksheet IDs preserved; level/lang/title/max_score taken from
      the legacy columns. Legacy `questions.question_id`s are NOT preserved (they're regenerated) --
      attempts are re-pointed via (worksheet_id, question index).
  submissions                                       -> IDs preserved; answers_json key `answer` is
      renamed to `selected_option`; state='graded'
  attempts                                          -> IDs preserved; question_id re-pointed
Legacy tables with no v2 counterpart to fill (scan_reviews, media, users, schema_migrations) are
not migrated; v2-only tables (mastery_history, scans, ...) start empty -- there's no history to
backfill, only current mastery snapshots.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlmodel import Session, select

from app.domain.worksheet_import import DEFAULT_SKILLS_PATH, insert_worksheet
from app.models import Attempt, School, Skill, Student, StudentSkillMastery, Submission, Worksheet
from app.models.submission import ProcessingState


class MigrationError(Exception):
    pass


@dataclass
class MigrationReport:
    counts: dict[str, tuple[int, int]] = field(default_factory=dict)  # table -> (legacy, target)
    problems: list[str] = field(default_factory=list)  # anything here means: do not commit
    notes: list[str] = field(default_factory=list)  # informational, worth reading

    @property
    def ok(self) -> bool:
        return not self.problems


def convert_answers(answers_json) -> list[dict]:
    """Legacy answers_json ({"answer", "is_correct", "question_index"}) -> v2 shape
    ({"question_index", "selected_option", "is_correct"}, as produced by grade_marks())."""
    if not isinstance(answers_json, list):
        raise MigrationError(f"answers_json is not a list: {type(answers_json).__name__}")
    converted = []
    for entry in answers_json:
        if not isinstance(entry, dict) or "question_index" not in entry:
            raise MigrationError(f"unexpected answers_json entry: {entry!r}")
        selected = entry["selected_option"] if "selected_option" in entry else entry.get("answer")
        converted.append(
            {
                "question_index": int(entry["question_index"]),
                "selected_option": (selected or "").strip().upper(),
                "is_correct": bool(entry.get("is_correct")),
            }
        )
    return converted


# Seeded by Alembic migrations rather than migrated from the legacy DB, so never wiped.
KEPT_TABLES = ("alembic_version", "projects")


def wipe_target(session: Session) -> list[str]:
    """TRUNCATE every data table except KEPT_TABLES. Transactional, so a dry-run undoes it."""
    tables = [
        r[0]
        for r in session.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND NOT (tablename = ANY(:kept)) ORDER BY 1"),
            {"kept": list(KEPT_TABLES)},
        )
    ]
    session.execute(text("TRUNCATE " + ", ".join(f'"{t}"' for t in tables) + " RESTART IDENTITY CASCADE"))
    return tables


def target_is_empty(session: Session) -> list[str]:
    """Names of core tables that already contain rows (empty list = safe to migrate into)."""
    occupied = []
    for table in ("schools", "students", "worksheets", "submissions", "attempts"):
        if session.execute(text(f'SELECT 1 FROM "{table}" LIMIT 1')).first():
            occupied.append(table)
    return occupied


def reset_sequences(session: Session) -> None:
    """Sequences don't follow explicit-ID inserts. Call after committing (setval isn't transactional)."""
    for table, column in (("submissions", "submission_id"), ("attempts", "attempt_id"), ("worksheets", "worksheet_id")):
        session.execute(
            text(
                f"SELECT setval(pg_get_serial_sequence('{table}', '{column}'), "
                f"COALESCE((SELECT MAX({column}) FROM {table}), 1), (SELECT MAX({column}) FROM {table}) IS NOT NULL)"
            )
        )
    session.commit()


# ---------- reading the legacy database ----------

def _rows(legacy: Connection, sql: str):
    return legacy.execute(text(sql)).mappings().all()


def _question_index(row) -> int | None:
    if row["index"] is not None:
        return int(row["index"])
    payload = row["question_json"] or {}
    return int(payload["index"]) if payload.get("index") is not None else None


@dataclass
class LegacyQuestions:
    by_id: dict[int, tuple[int, int]]  # legacy question_id -> (worksheet_id, index)
    by_key: dict[tuple[int, int], dict]  # (worksheet_id, index) -> {"skill_code", "json"}
    by_worksheet: dict[int, list[dict]]  # worksheet_id -> question_json in index order


def load_legacy_questions(legacy: Connection) -> LegacyQuestions:
    by_id, by_key, by_worksheet = {}, {}, {}
    rows = _rows(
        legacy,
        'SELECT question_id, worksheet_id, skill_code, question_json, "index" FROM questions ORDER BY worksheet_id, "index", question_id',
    )
    for row in rows:
        index = _question_index(row)
        if index is None or row["worksheet_id"] is None:
            raise MigrationError(f"legacy question {row['question_id']} has no worksheet_id/index")
        key = (row["worksheet_id"], index)
        by_id[row["question_id"]] = key
        by_key[key] = {"skill_code": row["skill_code"], "json": row["question_json"] or {}}
        by_worksheet.setdefault(row["worksheet_id"], []).append({**(row["question_json"] or {}), "index": index})
    return LegacyQuestions(by_id, by_key, by_worksheet)


# ---------- migration ----------

def migrate(legacy: Connection, session: Session, *, skills_path: Path | None = None) -> MigrationReport:
    """Copy everything and verify it. Does not commit; see module docstring."""
    report = MigrationReport()
    questions = load_legacy_questions(legacy)

    _migrate_simple_tables(legacy, session)
    _migrate_worksheets(legacy, session, questions, report)
    new_question_ids = {
        (r[1], r[2]): r[0] for r in session.execute(text('SELECT question_id, worksheet_id, "index" FROM questions'))
    }
    _migrate_submissions(legacy, session)
    _migrate_attempts(legacy, session, questions, new_question_ids, report)
    session.flush()

    verify(legacy, session, questions, report, skills_path=skills_path)
    return report


def _migrate_simple_tables(legacy: Connection, session: Session) -> None:
    for r in _rows(legacy, "SELECT school_code, school_name FROM schools ORDER BY school_code"):
        session.add(School(school_code=r["school_code"], school_name=r["school_name"]))
    for r in _rows(legacy, "SELECT skill_code, skill_name, skill_level, skill_weight FROM skills ORDER BY skill_code"):
        session.add(
            Skill(
                skill_code=r["skill_code"], skill_name=r["skill_name"], skill_level=str(r["skill_level"]),
                skill_weight=float(r["skill_weight"]) if r["skill_weight"] is not None else 1.0,
            )
        )
    session.flush()
    for r in _rows(
        legacy, "SELECT student_id, student_name, student_school_code, current_level, is_active FROM students ORDER BY student_id"
    ):
        session.add(
            Student(
                student_id=r["student_id"], student_name=r["student_name"], student_school_code=r["student_school_code"],
                current_level=r["current_level"], is_active=bool(r["is_active"]),
            )
        )
    session.flush()
    for r in _rows(legacy, "SELECT student_id, skill_code, mastery_score, last_updated FROM student_skill_mastery"):
        row = StudentSkillMastery(
            student_id=r["student_id"], skill_code=r["skill_code"],
            mastery_score=float(r["mastery_score"]) if r["mastery_score"] is not None else None,
        )
        if r["last_updated"] is not None:
            row.last_updated = r["last_updated"]
        session.add(row)
    session.flush()


def _migrate_worksheets(legacy: Connection, session: Session, questions: LegacyQuestions, report: MigrationReport) -> None:
    fallbacks = 0
    for r in _rows(
        legacy,
        "SELECT worksheet_id, worksheet_level, is_test, max_score, lang, worksheet_json, worksheet_category, title "
        "FROM worksheets ORDER BY worksheet_id",
    ):
        payload = r["worksheet_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        has_questions = bool(payload if isinstance(payload, list) else (payload or {}).get("questions"))
        if not has_questions:
            # Blob missing/empty: rebuild from the normalized questions table.
            rebuilt = questions.by_worksheet.get(r["worksheet_id"])
            if not rebuilt:
                raise MigrationError(f"worksheet {r['worksheet_id']} has no worksheet_json and no questions rows")
            payload = {
                "level": r["worksheet_level"], "language": r["lang"], "title": r["title"],
                "worksheet_category": r["worksheet_category"], "questions": rebuilt,
            }
            fallbacks += 1

        insert_worksheet(
            session, payload, worksheet_id=r["worksheet_id"], worksheet_category=r["worksheet_category"], commit=False
        )
        worksheet = session.get(Worksheet, r["worksheet_id"])
        # The legacy columns are authoritative over whatever the JSON blob says.
        worksheet.worksheet_level = r["worksheet_level"]
        worksheet.lang = r["lang"]
        worksheet.title = r["title"]
        if r["max_score"] is not None:
            worksheet.max_score = r["max_score"]
        if r["is_test"]:
            worksheet.worksheet_metadata = {**worksheet.worksheet_metadata, "legacy_is_test": True}
        session.add(worksheet)
        session.flush()
        session.expunge_all()  # ~100k questions + ~400k options would otherwise pile up in the identity map
    if fallbacks:
        report.notes.append(f"{fallbacks} worksheets had no usable worksheet_json and were rebuilt from the questions table")


def _migrate_submissions(legacy: Connection, session: Session) -> None:
    for r in _rows(
        legacy,
        "SELECT submission_id, student_id, worksheet_id, score, from_number, answers_json, submitted_at, worksheet_category "
        "FROM submissions ORDER BY submission_id",
    ):
        submission = Submission(
            submission_id=r["submission_id"], student_id=r["student_id"], worksheet_id=r["worksheet_id"],
            worksheet_category=r["worksheet_category"], score=r["score"], from_number=r["from_number"],
            answers_json=convert_answers(r["answers_json"]), state=ProcessingState.GRADED.value,
        )
        if r["submitted_at"] is not None:
            submission.submitted_at = r["submitted_at"]
        session.add(submission)
    session.flush()


def _migrate_attempts(
    legacy: Connection, session: Session, questions: LegacyQuestions,
    new_question_ids: dict[tuple[int, int], int], report: MigrationReport,
) -> None:
    unmapped = 0
    for r in _rows(
        legacy,
        "SELECT attempt_id, student_id, submission_id, question_id, worksheet_id, is_correct, skill_code, attempted_at "
        "FROM attempts ORDER BY attempt_id",
    ):
        key = questions.by_id.get(r["question_id"])
        new_id = new_question_ids.get(key) if key else None
        if key is None or new_id is None or key[0] != r["worksheet_id"]:
            unmapped += 1
            report.problems.append(
                f"attempt {r['attempt_id']}: legacy question {r['question_id']} doesn't map to a v2 question of worksheet {r['worksheet_id']}"
            )
            continue
        attempt = Attempt(
            attempt_id=r["attempt_id"], student_id=r["student_id"], submission_id=r["submission_id"], question_id=new_id,
            worksheet_id=r["worksheet_id"], is_correct=r["is_correct"], skill_code=r["skill_code"],
        )
        if r["attempted_at"] is not None:
            attempt.attempted_at = r["attempted_at"]
        session.add(attempt)
    session.flush()


# ---------- verification ----------

def _count(session: Session, table: str) -> int:
    return session.execute(text(f'SELECT count(*) FROM "{table}"')).scalar_one()


def verify(
    legacy: Connection, session: Session, questions: LegacyQuestions, report: MigrationReport,
    *, skills_path: Path | None = None,
) -> None:
    """Row counts, answer-key fidelity, and a re-grade of every legacy submission through the v2
    answer-key path. Anything in report.problems means the migration must not be committed."""
    # 1. counts
    for table in ("schools", "skills", "students", "worksheets", "questions", "submissions", "attempts", "student_skill_mastery"):
        legacy_count = legacy.execute(text(f'SELECT count(*) FROM "{table}"')).scalar_one()
        report.counts[table] = (legacy_count, _count(session, table))
    expected_options = sum(len(q["json"].get("options") or []) for q in questions.by_key.values())
    report.counts["question_options"] = (expected_options, _count(session, "question_options"))
    for table, (legacy_count, target_count) in report.counts.items():
        if legacy_count != target_count:
            report.problems.append(f"{table}: legacy has {legacy_count} rows, target has {target_count}")

    # 2. the answer key: v2's question_options.is_correct must equal legacy correct_option, per question
    target_key = {
        (r[0], r[1]): r[2]
        for r in session.execute(
            text(
                'SELECT q.worksheet_id, q."index", o.option_label FROM question_options o '
                "JOIN questions q ON q.question_id = o.question_id WHERE o.is_correct"
            )
        )
    }
    target_skill = {
        (r[0], r[1]): r[2] for r in session.execute(text('SELECT worksheet_id, "index", skill_code FROM questions'))
    }
    key_mismatches = skill_mismatches = missing = 0
    for key, legacy_q in questions.by_key.items():
        expected = (legacy_q["json"].get("correct_option") or "").strip().upper() or None
        if key not in target_skill:
            missing += 1
            continue
        if target_key.get(key) != expected:
            key_mismatches += 1
        if target_skill[key] != legacy_q["skill_code"]:
            skill_mismatches += 1
    if missing:
        report.problems.append(f"{missing} legacy questions are missing from the target")
    if key_mismatches:
        report.problems.append(f"{key_mismatches} questions have a different correct answer in the target than in legacy")
    if skill_mismatches:
        report.notes.append(f"{skill_mismatches} questions have a different skill_code than the legacy questions table")

    # 3. re-grade every legacy submission with the v2 key
    graded = answer_mismatch = score_vs_legacy = legacy_internal = attempts_mismatch = 0
    for r in _rows(legacy, "SELECT submission_id, worksheet_id, score, answers_json FROM submissions ORDER BY submission_id"):
        answers = convert_answers(r["answers_json"])
        graded += 1
        new_flags = [
            bool(a["selected_option"]) and a["selected_option"] == target_key.get((r["worksheet_id"], a["question_index"]))
            for a in answers
        ]
        answer_mismatch += sum(1 for a, flag in zip(answers, new_flags) if a["is_correct"] != flag)
        if r["score"] != sum(new_flags):
            score_vs_legacy += 1
        if r["score"] != sum(1 for a in answers if a["is_correct"]):
            legacy_internal += 1
        attempt_count = session.execute(
            text("SELECT count(*) FROM attempts WHERE submission_id = :s"), {"s": r["submission_id"]}
        ).scalar_one()
        if attempt_count != len(answers):
            attempts_mismatch += 1
    report.notes.append(f"re-graded {graded} legacy submissions with the v2 answer key")
    if answer_mismatch:
        report.problems.append(f"{answer_mismatch} answers would be graded differently by v2 than they were by legacy")
    if score_vs_legacy:
        report.problems.append(f"{score_vs_legacy} submissions would get a different total score in v2")
    if legacy_internal:
        report.notes.append(f"{legacy_internal} legacy submissions' stored score != their own count of correct answers (pre-existing)")
    if attempts_mismatch:
        report.notes.append(f"{attempts_mismatch} submissions have a different number of attempts rows than answers")

    # 4. the skill catalog vs the copy shipped with v2
    try:
        catalog = json.loads((skills_path or DEFAULT_SKILLS_PATH).read_text(encoding="utf-8"))
        entries = catalog if isinstance(catalog, list) else list(catalog.values())
        shipped = {e["code"]: (e["skill"], str(e["difficulty_level"])) for e in entries}
        migrated = {s.skill_code: (s.skill_name, s.skill_level) for s in session.exec(select(Skill)).all()}
        if shipped != migrated:
            differing = sorted(set(shipped) ^ set(migrated) | {c for c in shipped if c in migrated and shipped[c] != migrated[c]})
            report.notes.append(f"skills differ from app/data/skills.json for: {', '.join(differing)}")
    except (OSError, KeyError, json.JSONDecodeError):
        report.notes.append("could not compare skills against app/data/skills.json")
