"""Legacy-DB migration (app/domain/legacy_migration.py). A miniature legacy database is built in a
throwaway schema of the dev DB; the migration runs into the real v2 tables inside ONE transaction
(after wipe_target, which is itself transactional) and is always rolled back -- so nothing in
the dev DB changes, and the tests can compare exact row counts.
"""

import pytest
from sqlalchemy import text
from sqlmodel import Session, select

from app.db.session import engine
from app.domain import legacy_migration as lm
from app.models import Attempt, Question, QuestionOption, Student, Submission, Worksheet

SCHEMA = "legacy_mig_test"

DDL = f"""
CREATE SCHEMA {SCHEMA};
CREATE TABLE {SCHEMA}.schools (school_code text PRIMARY KEY, school_name text NOT NULL);
CREATE TABLE {SCHEMA}.skills (skill_code text PRIMARY KEY, skill_name text NOT NULL, skill_level text NOT NULL, skill_weight numeric DEFAULT 1.0);
CREATE TABLE {SCHEMA}.students (student_id text PRIMARY KEY, student_name text NOT NULL, student_school_code text, current_level text, is_active boolean NOT NULL DEFAULT true);
CREATE TABLE {SCHEMA}.worksheets (worksheet_id int PRIMARY KEY, worksheet_level text, is_test boolean NOT NULL DEFAULT false, max_score int, lang text, worksheet_json jsonb, worksheet_category text NOT NULL DEFAULT 'practice', title text);
CREATE TABLE {SCHEMA}.questions (question_id int PRIMARY KEY, worksheet_id int, skill_code text NOT NULL, question_json jsonb, "index" int);
CREATE TABLE {SCHEMA}.submissions (submission_id int PRIMARY KEY, student_id text NOT NULL, worksheet_id int NOT NULL, score int, from_number text, answers_json jsonb, submitted_at timestamptz DEFAULT now(), worksheet_category text NOT NULL DEFAULT 'practice');
CREATE TABLE {SCHEMA}.attempts (attempt_id int PRIMARY KEY, student_id text NOT NULL, submission_id int NOT NULL, question_id int NOT NULL, worksheet_id int NOT NULL, is_correct boolean, skill_code text NOT NULL, attempted_at timestamptz DEFAULT now());
CREATE TABLE {SCHEMA}.student_skill_mastery (student_id text, skill_code text, mastery_score numeric, last_updated timestamptz DEFAULT now(), PRIMARY KEY (student_id, skill_code));
"""


def qjson(index: int, correct: str, skill: str = "LMT1") -> dict:
    return {"index": index, "options": ["w", "x", "y", "z"], "skill_code": skill, "question_text": f"Q{index}", "correct_option": correct}


def answers(*pairs) -> list[dict]:
    """(question_index, answer, is_correct) -> legacy answers_json entries."""
    return [{"question_index": i, "answer": a, "is_correct": c} for i, a, c in pairs]


class LegacyDb:
    def __init__(self, connection):
        self.c = connection

    def run(self, sql: str, **params):
        self.c.execute(text(sql), params)


@pytest.fixture
def legacy_db():
    with engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
        conn.execute(text(DDL))
    connection = engine.connect()
    connection.execute(text(f"SET search_path TO {SCHEMA}"))  # legacy queries resolve ONLY here, never in public
    db = LegacyDb(connection)

    import json
    ws1_questions = [qjson(1, "A"), qjson(2, "B"), qjson(3, "C")]
    ws2_questions = [qjson(1, "A", "LMT2"), qjson(2, "B", "LMT2")]
    db.run("INSERT INTO schools VALUES ('LMT', 'Legacy Migration Test School')")
    db.run("INSERT INTO skills VALUES ('LMT1', 'Test skill one', '1', 1.0), ('LMT2', 'Test skill two', '2', 2.5)")
    db.run("INSERT INTO students VALUES ('9701', 'Student One', 'LMT', 'A', true), ('9702', 'Student Two', 'LMT', 'B', false)")
    # worksheet 1: has its blob; worksheet 2: blob missing -> must be rebuilt from the questions table
    db.run("INSERT INTO worksheets VALUES (970001, 'A', false, 3, 'mr', CAST(:j AS jsonb), 'homework', 'Legacy title')",
           j=json.dumps({"level": "A", "language": "mr", "title": "Blob title", "questions": ws1_questions}))
    db.run("INSERT INTO worksheets VALUES (970002, 'B', true, 2, 'en', NULL, 'homework', NULL)")
    for qid, ws, index, correct, skill in [(101, 970001, 1, "A", "LMT1"), (102, 970001, 2, "B", "LMT1"), (103, 970001, 3, "C", "LMT1"),
                                            (201, 970002, 1, "A", "LMT2"), (202, 970002, 2, "B", "LMT2")]:
        db.run('INSERT INTO questions VALUES (:q, :w, :s, CAST(:j AS jsonb), :i)', q=qid, w=ws, s=skill, j=json.dumps(qjson(index, correct, skill)), i=index)
    # submission 1: Q1 correct, Q2 wrong (picked C), Q3 unanswered -> score 1.  submission 2: both correct.
    db.run("INSERT INTO submissions VALUES (970001, '9701', 970001, 1, '+910000000001', CAST(:a AS jsonb), '2026-08-24 09:37:09+00', 'homework')",
           a=json.dumps(answers((1, "A", True), (2, "C", False), (3, "", False))))
    db.run("INSERT INTO submissions VALUES (970002, '9702', 970002, 2, '+910000000002', CAST(:a AS jsonb), '2026-08-25 09:37:09+00', 'homework')",
           a=json.dumps(answers((1, "A", True), (2, "B", True))))
    for attempt_id, sub, qid, ws, correct, skill in [(970001, 970001, 101, 970001, True, "LMT1"), (970002, 970001, 102, 970001, False, "LMT1"),
                                                     (970003, 970001, 103, 970001, False, "LMT1"), (970004, 970002, 201, 970002, True, "LMT2"),
                                                     (970005, 970002, 202, 970002, True, "LMT2")]:
        student = "9701" if sub == 970001 else "9702"
        db.run("INSERT INTO attempts VALUES (:a, :st, :s, :q, :w, :c, :k, now())", a=attempt_id, st=student, s=sub, q=qid, w=ws, c=correct, k=skill)
    db.run("INSERT INTO student_skill_mastery VALUES ('9701', 'LMT1', 0.5, now())")
    connection.commit()

    yield db

    connection.rollback()
    connection.execute(text("RESET search_path"))  # the connection goes back to the pool; don't leak the legacy schema into other tests
    connection.commit()
    connection.close()  # must close before DROP SCHEMA, or its open transaction blocks it
    with engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))


@pytest.fixture
def session():
    with Session(engine) as s:
        yield s
        s.rollback()  # the whole point: nothing these tests do to the dev DB is ever committed


def run_migration(legacy_db, session):
    lm.wipe_target(session)
    assert lm.target_is_empty(session) == []
    return lm.migrate(legacy_db.c, session)


# ---------- pure conversion ----------

def test_convert_answers_renames_answer_and_normalizes():
    assert lm.convert_answers(
        [{"question_index": 1, "answer": "b", "is_correct": True}, {"question_index": 2, "answer": "", "is_correct": False},
         {"question_index": 3, "answer": None, "is_correct": None}]
    ) == [
        {"question_index": 1, "selected_option": "B", "is_correct": True},
        {"question_index": 2, "selected_option": "", "is_correct": False},
        {"question_index": 3, "selected_option": "", "is_correct": False},
    ]
    # already-v2 entries pass through
    assert lm.convert_answers([{"question_index": 4, "selected_option": "D", "is_correct": True}])[0]["selected_option"] == "D"


@pytest.mark.parametrize("bad", [{"not": "a list"}, [{"answer": "A"}], ["A"]])
def test_convert_answers_rejects_unexpected_shapes(bad):
    with pytest.raises(lm.MigrationError):
        lm.convert_answers(bad)


# ---------- the migration ----------

def test_migration_copies_everything_and_verifies(legacy_db, session):
    report = run_migration(legacy_db, session)

    assert report.ok, report.problems
    assert all(legacy == target for legacy, target in report.counts.values()), report.counts
    assert report.counts["question_options"] == (20, 20)  # 5 questions x 4 options
    assert any("rebuilt from the questions table" in n for n in report.notes)  # worksheet 970002 had no blob
    assert any("re-graded 2 legacy submissions" in n for n in report.notes)

    # worksheets: IDs preserved, legacy columns win over the blob, is_test kept, template/page_count set by insert_worksheet
    ws1, ws2 = session.get(Worksheet, 970001), session.get(Worksheet, 970002)
    assert (ws1.title, ws1.worksheet_level, ws1.lang, ws1.max_score) == ("Legacy title", "A", "mr", 3)  # not "Blob title"
    assert ws2.title is None and ws2.worksheet_metadata == {"legacy_is_test": True}
    assert ws1.template_id is not None and ws1.page_count == 1

    # the answer key was derived from correct_option
    labels = {
        (q.worksheet_id, q.index): o.option_label
        for q, o in session.exec(select(Question, QuestionOption).join(QuestionOption, QuestionOption.question_id == Question.question_id).where(QuestionOption.is_correct)).all()
    }
    assert labels == {(970001, 1): "A", (970001, 2): "B", (970001, 3): "C", (970002, 1): "A", (970002, 2): "B"}

    # submissions: IDs kept, answers renamed, graded, timestamp preserved
    sub = session.get(Submission, 970001)
    assert sub.answers_json == [
        {"question_index": 1, "selected_option": "A", "is_correct": True},
        {"question_index": 2, "selected_option": "C", "is_correct": False},
        {"question_index": 3, "selected_option": "", "is_correct": False},
    ]
    assert (sub.state, sub.score, sub.student_id) == ("graded", 1, "9701")
    assert sub.submitted_at.year == 2026 and sub.submitted_at.month == 8

    # attempts re-pointed at the *new* question ids via (worksheet, index)
    attempt = session.get(Attempt, 970002)
    question = session.get(Question, attempt.question_id)
    assert (question.worksheet_id, question.index, attempt.is_correct, attempt.submission_id) == (970001, 2, False, 970001)

    assert session.get(Student, "9702").is_active is False  # is_active carried over, not defaulted


def test_the_target_is_left_untouched_after_rollback(legacy_db, session):
    before = session.execute(text("SELECT count(*) FROM students")).scalar_one()
    run_migration(legacy_db, session)
    session.rollback()
    assert session.execute(text("SELECT count(*) FROM students")).scalar_one() == before
    assert session.get(Worksheet, 970001) is None


def test_target_is_empty_reports_occupied_tables(legacy_db, session):
    lm.wipe_target(session)
    session.add(Student(student_id="9799", student_name="Occupier"))
    session.flush()
    assert lm.target_is_empty(session) == ["students"]


def test_answer_key_disagreement_blocks_the_migration(legacy_db, session):
    """If a worksheet's JSON blob and its questions rows disagree about the right answer, v2 would
    grade differently than legacy did -- that must surface as a problem, not pass silently."""
    legacy_db.run("""UPDATE questions SET question_json = jsonb_set(question_json, '{correct_option}', '"D"') WHERE question_id = 101""")
    report = run_migration(legacy_db, session)
    assert not report.ok
    assert any("different correct answer" in p for p in report.problems)


def test_attempt_pointing_at_a_missing_question_is_reported(legacy_db, session):
    legacy_db.run("INSERT INTO attempts VALUES (970099, '9701', 970001, 999, 970001, true, 'LMT1', now())")
    report = run_migration(legacy_db, session)
    assert not report.ok
    assert any("attempt 970099" in p for p in report.problems)
