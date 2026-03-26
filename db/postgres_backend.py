"""
Postgres Backend — Database access layer for PaperPlus.

Provides functions for all database operations described in flows.md:
- School and student management
- Worksheet and question CRUD
- Assignment lifecycle
- Submission processing and attempt recording
- Skill mastery tracking
- Media references
- Schema migrations
"""

import sys
import json
import logging
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SETTINGS

logger = logging.getLogger(__name__)

DATABASE_URL = SETTINGS.DATABASE_URL
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

@contextmanager
def get_connection():
    """Yield a psycopg connection that auto-commits on success and rolls back on error."""
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_cursor(conn=None):
    """Yield a dict-row cursor. If *conn* is None a new connection is created."""
    if conn is not None:
        with conn.cursor(row_factory=dict_row) as cur:
            yield cur
    else:
        with get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                yield cur


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------

def run_migrations():
    """Apply all SQL migration files that have not yet been applied."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Ensure the tracking table exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version text PRIMARY KEY,
                    applied_at timestamp with time zone DEFAULT now()
                )
            """)
            cur.execute("SELECT version FROM schema_migrations")
            applied = {row["version"] for row in cur.fetchall()}

        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        for mf in migration_files:
            version = mf.stem  # e.g. "001_create_tables"
            if version in applied:
                continue
            logger.info("Applying migration %s", version)
            sql = mf.read_text()
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)",
                    (version,),
                )
        conn.commit()


# ---------------------------------------------------------------------------
# Schools
# ---------------------------------------------------------------------------

def upsert_school(school_code: str, school_name: str):
    """Insert a school or do nothing on conflict."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO schools (school_code, school_name)
                   VALUES (%s, %s)
                   ON CONFLICT (school_code) DO UPDATE SET school_name = EXCLUDED.school_name""",
                (school_code, school_name),
            )


def get_school(school_code: str) -> dict | None:
    """Return a school row or None."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM schools WHERE school_code = %s", (school_code,))
            return cur.fetchone()


def list_schools() -> list[dict]:
    """Return all schools."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM schools ORDER BY school_name")
            return cur.fetchall()


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------

def upsert_student(student_id: str, student_name: str, school_code: str, current_level: str = "A"):
    """Insert a student or do nothing on conflict."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO students (student_id, student_name, student_school_code, current_level)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (student_id) DO NOTHING""",
                (student_id, student_name, school_code, current_level),
            )


def get_student(student_id: str) -> dict | None:
    """Return a student row or None."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM students WHERE student_id = %s", (student_id,))
            return cur.fetchone()



def update_student_level(student_id: str, new_level: str):
    """Change a student's current worksheet level."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE students SET current_level = %s WHERE student_id = %s",
                (new_level, student_id),
            )


def deactivate_student(student_id: str):
    """Mark a student as inactive."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE students SET is_active = false WHERE student_id = %s",
                (student_id,),
            )


def list_students(school_code: str = None, active_only: bool = True) -> list[dict]:
    """Return students, optionally filtered by school and active status."""
    clauses, params = [], []
    if school_code:
        clauses.append("student_school_code = %s")
        params.append(school_code)
    if active_only:
        clauses.append("is_active = true")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM students {where} ORDER BY student_name", params)
            return cur.fetchall()


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

def upsert_skill(skill_code: str, skill_name: str, skill_level: str, skill_weight: float = 1.0):
    """Insert or update a skill."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO skills (skill_code, skill_name, skill_level, skill_weight)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (skill_code) DO UPDATE
                       SET skill_name = EXCLUDED.skill_name,
                           skill_level = EXCLUDED.skill_level,
                           skill_weight = EXCLUDED.skill_weight""",
                (skill_code, skill_name, skill_level, skill_weight),
            )


def get_skill(skill_code: str) -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM skills WHERE skill_code = %s", (skill_code,))
            return cur.fetchone()


def list_skills(level: str = None) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            if level:
                cur.execute("SELECT * FROM skills WHERE skill_level = %s ORDER BY skill_code", (level,))
            else:
                cur.execute("SELECT * FROM skills ORDER BY skill_code")
            return cur.fetchall()


def import_skills_from_json(json_path: str | Path):
    """Bulk-import skills from the services/skills.json file."""
    data = json.loads(Path(json_path).read_text())
    skills = data if isinstance(data, list) else list(data.values())
    with get_connection() as conn:
        with conn.cursor() as cur:
            for s in skills:
                cur.execute(
                    """INSERT INTO skills (skill_code, skill_name, skill_level)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (skill_code) DO UPDATE
                           SET skill_name = EXCLUDED.skill_name,
                               skill_level = EXCLUDED.skill_level""",
                    (s["code"], s["skill"], s["difficulty_level"]),
                )


# ---------------------------------------------------------------------------
# Worksheets
# ---------------------------------------------------------------------------

def create_worksheet(worksheet_json: dict,
                     is_test: bool = False) -> int:
    """
    Insert a new worksheet row and return the generated worksheet_id.

    Flow: "add a new worksheet to the database"
    - Receives the generated worksheet JSON
    - Inserts the worksheet record
    - Returns worksheet_id so it can be embedded in the JSON / used for questions
    """
    lang = worksheet_json.get("language")
    level = worksheet_json.get("level")
    max_score = len(worksheet_json.get("questions", []))
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO worksheets (worksheet_level, is_test, max_score, lang, worksheet_json)
                   VALUES (%s, %s, %s, %s, %s)
                   RETURNING worksheet_id""",
                (level, is_test, max_score, lang, json.dumps(worksheet_json)),
            )
            return cur.fetchone()["worksheet_id"]


def get_worksheet(worksheet_id: int) -> dict | None:
    """Return a worksheet row (including its JSON) by ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM worksheets WHERE worksheet_id = %s", (worksheet_id,))
            return cur.fetchone()


def get_worksheet_json(worksheet_id: int) -> dict | list | None:
    """Return only the worksheet_json for PDF generation."""
    row = get_worksheet(worksheet_id)
    return row["worksheet_json"] if row else None


def get_answer_key(worksheet_id: int) -> list | None:
    """
    Extract the answer key from a worksheet's JSON.

    The JSON stores questions as a list; each question has a 'correct_option' field.
    Returns e.g. ['A', 'C', 'B', ...] or None.
    """
    ws_json = get_worksheet_json(worksheet_id)
    if ws_json is None:
        return None
    questions = ws_json if isinstance(ws_json, list) else ws_json.get("questions", [])
    return [q["correct_option"] for q in questions]


def list_worksheets(level: str = None, lang: str = None, is_test: bool = None) -> list[dict]:
    """List worksheets with optional filters."""
    clauses, params = [], []
    if level:
        clauses.append("worksheet_level = %s")
        params.append(level)
    if lang:
        clauses.append("lang = %s")
        params.append(lang)
    if is_test is not None:
        clauses.append("is_test = %s")
        params.append(is_test)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM worksheets {where} ORDER BY worksheet_id", params)
            return cur.fetchall()


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

def insert_questions_for_worksheet(worksheet_id: int, questions: list[dict]) -> list[int]:
    """
    Bulk-insert questions extracted from a worksheet JSON.

    Each dict should contain at minimum: skill_code, and optionally the full
    question JSON (question_text, options, correct_option, index …).

    Returns the list of generated question_ids.
    """
    question_ids = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            for position, q in enumerate(questions, start=1):
                skill_code = q["skill_code"]
                index = q.get("index", position)
                cur.execute(
                    """INSERT INTO questions (worksheet_id, skill_code, index, question_json)
                       VALUES (%s, %s, %s, %s)
                       RETURNING question_id""",
                    (worksheet_id, skill_code, index, json.dumps(q)),
                )
                question_ids.append(cur.fetchone()["question_id"])
    return question_ids


def get_questions_for_worksheet(worksheet_id: int) -> list[dict]:
    """Return all question rows for a worksheet, ordered by index."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM questions WHERE worksheet_id = %s ORDER BY index",
                (worksheet_id,),
            )
            return cur.fetchall()


def get_question(question_id: int) -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM questions WHERE question_id = %s", (question_id,))
            return cur.fetchone()


# ---------------------------------------------------------------------------
# Assignments
# ---------------------------------------------------------------------------

def create_assignment(student_id: str, worksheet_id: int) -> int:
    """
    Assign a worksheet to a student.  Returns the new assignment_id.

    Flow: "assign a worksheet to a student"
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO assignments (student_id, worksheet_id)
                   VALUES (%s, %s)
                   RETURNING assignment_id""",
                (student_id, worksheet_id),
            )
            return cur.fetchone()["assignment_id"]


def get_assignment(assignment_id: int) -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM assignments WHERE assignment_id = %s", (assignment_id,))
            return cur.fetchone()


def get_assignment_by_student_worksheet(student_id: str, worksheet_id: int) -> dict | None:
    """
    Find the most recent unsubmitted assignment for a student + worksheet pair.

    Flow: "retrieve the assignment_id by querying assignments with worksheet_id"
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM assignments
                   WHERE student_id = %s AND worksheet_id = %s AND submitted_at IS NULL
                   ORDER BY assigned_at DESC LIMIT 1""",
                (student_id, worksheet_id),
            )
            return cur.fetchone()


def get_assignment_by_worksheet(worksheet_id: int) -> dict | None:
    """
    Find the most recent unsubmitted assignment for a worksheet.

    Valid for regular (non-test) worksheets where each worksheet is assigned
    to exactly one student at a time.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM assignments
                   WHERE worksheet_id = %s AND submitted_at IS NULL
                   ORDER BY assigned_at DESC LIMIT 1""",
                (worksheet_id,),
            )
            return cur.fetchone()


def mark_assignment_submitted(assignment_id: int):
    """Set the submitted_at timestamp on an assignment."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE assignments SET submitted_at = now() WHERE assignment_id = %s",
                (assignment_id,),
            )


def list_assignments(student_id: str = None, pending_only: bool = False) -> list[dict]:
    """List assignments, optionally filtered."""
    clauses, params = [], []
    if student_id:
        clauses.append("student_id = %s")
        params.append(student_id)
    if pending_only:
        clauses.append("submitted_at IS NULL")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM assignments {where} ORDER BY assigned_at DESC", params
            )
            return cur.fetchall()


# ---------------------------------------------------------------------------
# Submissions
# ---------------------------------------------------------------------------

def create_submission(student_id: str, assignment_id: int, worksheet_id: int,
                      score: int, from_number: str, answers_json: dict | list) -> int:
    """
    Record a graded submission.

    Flow: "receive a submission from a student"
    - Inserts the submission row
    - Returns submission_id for linking attempts
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO submissions
                       (student_id, assignment_id, worksheet_id, score, from_number, answers_json)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   RETURNING submission_id""",
                (student_id, assignment_id, worksheet_id, score, from_number,
                 json.dumps(answers_json)),
            )
            return cur.fetchone()["submission_id"]


def get_submission(submission_id: int) -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM submissions WHERE submission_id = %s", (submission_id,))
            return cur.fetchone()


def get_submissions_for_worksheet(worksheet_id: int, student_id: str = None) -> list[dict]:
    """Find submissions for a worksheet, optionally filtered by student."""
    clauses = ["worksheet_id = %s"]
    params = [worksheet_id]
    if student_id:
        clauses.append("student_id = %s")
        params.append(student_id)
    where = "WHERE " + " AND ".join(clauses)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM submissions {where} ORDER BY submitted_at DESC", params
            )
            return cur.fetchall()


def get_latest_submission(student_id: str, worksheet_id: int) -> dict | None:
    """Return the most recent submission for a student + worksheet."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM submissions
                   WHERE student_id = %s AND worksheet_id = %s
                   ORDER BY submitted_at DESC LIMIT 1""",
                (student_id, worksheet_id),
            )
            return cur.fetchone()


def get_latest_submission_by_worksheet(worksheet_id: int) -> dict | None:
    """Return the most recent submission for a worksheet (any student)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM submissions
                   WHERE worksheet_id = %s
                   ORDER BY submitted_at DESC LIMIT 1""",
                (worksheet_id,),
            )
            return cur.fetchone()


def overwrite_submission(submission_id: int, score: int, answers_json: dict | list):
    """
    Overwrite an existing submission (manual correction flow).

    Flow: "if already exists, overwrite it (overwrite all the attempt entries as well)"
    - Updates answers and score on the submission
    - Caller should also delete + re-insert attempts
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE submissions
                   SET score = %s, answers_json = %s, submitted_at = now()
                   WHERE submission_id = %s""",
                (score, json.dumps(answers_json), submission_id),
            )


# ---------------------------------------------------------------------------
# Attempts
# ---------------------------------------------------------------------------

def insert_attempts(student_id: str, submission_id: int, worksheet_id: int,
                    attempts: list[dict]):
    """
    Bulk-insert attempt rows from processed answers.

    Each dict in *attempts* must contain: question_id, is_correct, skill_code.

    Flow: "process answers_json into attempt records"
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            for a in attempts:
                cur.execute(
                    """INSERT INTO attempts
                           (student_id, submission_id, question_id, worksheet_id, is_correct, skill_code)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (student_id, submission_id, a["question_id"],
                     worksheet_id, a["is_correct"], a["skill_code"]),
                )


def delete_attempts_for_submission(submission_id: int):
    """Delete all attempts tied to a submission (used before overwrite)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM attempts WHERE submission_id = %s", (submission_id,))


def get_attempts_for_submission(submission_id: int) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM attempts WHERE submission_id = %s ORDER BY question_id",
                (submission_id,),
            )
            return cur.fetchall()


# ---------------------------------------------------------------------------
# Student skill mastery
# ---------------------------------------------------------------------------

def update_skill_mastery(student_id: str, skill_code: str, mastery_score: float):
    """Upsert a mastery score for a student × skill pair."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO student_skill_mastery (student_id, skill_code, mastery_score)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (student_id, skill_code) DO UPDATE
                       SET mastery_score = EXCLUDED.mastery_score,
                           last_updated = now()""",
                (student_id, skill_code, mastery_score),
            )


def get_skill_mastery(student_id: str) -> list[dict]:
    """Return all mastery rows for a student."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT m.*, s.skill_name, s.skill_level
                   FROM student_skill_mastery m
                   JOIN skills s USING (skill_code)
                   WHERE m.student_id = %s
                   ORDER BY s.skill_level, s.skill_code""",
                (student_id,),
            )
            return cur.fetchall()


def recalculate_skill_mastery(student_id: str, skill_code: str):
    """
    Recalculate mastery for one student × skill from attempt history.

    Simple accuracy model: mastery = correct / total for that skill.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT COUNT(*) AS total,
                          SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) AS correct
                   FROM attempts
                   WHERE student_id = %s AND skill_code = %s""",
                (student_id, skill_code),
            )
            row = cur.fetchone()
            total = row["total"]
            if total == 0:
                return
            mastery = row["correct"] / total
    update_skill_mastery(student_id, skill_code, mastery)


# ---------------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------------

def create_media(owner_type: str, owner_id: int, storage_path: str) -> int:
    """Insert a media reference and return its media_id."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO media (owner_type, owner_id, storage_path)
                   VALUES (%s, %s, %s)
                   RETURNING media_id""",
                (owner_type, owner_id, storage_path),
            )
            return cur.fetchone()["media_id"]


def get_media(owner_type: str, owner_id: int) -> list[dict]:
    """Return all media rows for a given owner."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM media WHERE owner_type = %s AND owner_id = %s ORDER BY created_at",
                (owner_type, owner_id),
            )
            return cur.fetchall()


# ---------------------------------------------------------------------------
# Composite / high-level flows
# ---------------------------------------------------------------------------

def add_worksheet_to_db(worksheet_json: dict | list,
                        is_test: bool = False) -> dict:
    """
    Full flow: "add a new worksheet to the database"

    1. Insert worksheet row → get worksheet_id
    2. Embed worksheet_id into the JSON
    3. Extract each question, insert into questions table
    4. Return {worksheet_id, question_ids}
    """
    # Parse questions from JSON
    questions = worksheet_json if isinstance(worksheet_json, list) else worksheet_json.get("questions", [])

    # Insert worksheet
    worksheet_id = create_worksheet(worksheet_json, is_test=is_test)

    # Insert questions
    question_ids = insert_questions_for_worksheet(worksheet_id, questions)

    # Update the stored JSON with worksheet_id and question_ids
    with get_connection() as conn:
        with conn.cursor() as cur:
            if isinstance(worksheet_json, list):
                enriched = {"worksheet_id": worksheet_id, "questions": []}
                for q, qid in zip(questions, question_ids):
                    q_copy = dict(q)
                    q_copy["question_id"] = qid
                    enriched["questions"].append(q_copy)
            else:
                enriched = dict(worksheet_json)
                enriched["worksheet_id"] = worksheet_id
                for q, qid in zip(enriched.get("questions", []), question_ids):
                    q["question_id"] = qid
            cur.execute(
                "UPDATE worksheets SET worksheet_json = %s WHERE worksheet_id = %s",
                (json.dumps(enriched), worksheet_id),
            )

    print(f"Inserted worksheet {worksheet_id} with {len(question_ids)} questions")
    return {"worksheet_id": worksheet_id, "question_ids": question_ids}


def process_submission(worksheet_id: int, score: int,
                       from_number: str, answers_json: list[dict]) -> dict:
    """
    Full flow: "receive a submission from a student / process the worksheet"

    1. Look up assignment by worksheet_id (valid since 1 assignment per non-test worksheet)
    2. Derive student_id from the assignment
    3. Create submission record
    4. Mark assignment as submitted
    5. Build attempt records from answers_json
    6. Insert attempts
    7. Recalculate skill mastery for affected skills
    8. Return {submission_id, assignment_id, student_id, attempts_count}

    answers_json is a list like:
        [{"question_index": 1, "selected": "B", "is_correct": true}, ...]
    """
    # Find the assignment by worksheet — derive student from it
    assignment = get_assignment_by_worksheet(worksheet_id)
    if assignment is None:
        raise ValueError(f"No open assignment found for worksheet_id={worksheet_id}")

    assignment_id = assignment["assignment_id"]
    student_id = assignment["student_id"]

    # Create submission
    submission_id = create_submission(
        student_id, assignment_id, worksheet_id, score, from_number, answers_json
    )

    # Mark assignment submitted
    mark_assignment_submitted(assignment_id)

    # Build and insert attempts
    questions = get_questions_for_worksheet(worksheet_id)
    attempts = []
    affected_skills = set()
    for ans in answers_json:
        q_index = ans.get("question_index", 1)  # 1-based
        if 0 < q_index <= len(questions):
            q = questions[q_index - 1]
            skill = q["skill_code"]
            attempts.append({
                "question_id": q["question_id"],
                "is_correct": ans.get("is_correct", False),
                "skill_code": skill,
            })
            affected_skills.add(skill)

    if attempts:
        insert_attempts(student_id, submission_id, worksheet_id, attempts)

    # Recalculate mastery for each affected skill
    for skill_code in affected_skills:
        recalculate_skill_mastery(student_id, skill_code)

    return {
        "submission_id": submission_id,
        "assignment_id": assignment_id,
        "student_id": student_id,
        "attempts_count": len(attempts),
    }


def overwrite_submission_flow(worksheet_id: int, score: int,
                              answers_json: list[dict]) -> dict:
    """
    Manual correction flow: "if already exists, overwrite it"

    1. Derive student_id from the assignment (same as process_submission)
    2. Find existing submission by worksheet
    3. Delete old attempts
    4. Overwrite submission score/answers
    5. Re-insert new attempts
    6. Recalculate mastery
    """
    # Derive student_id from the assignment
    assignment = get_assignment_by_worksheet(worksheet_id)
    if assignment is None:
        raise ValueError(f"No assignment found for worksheet_id={worksheet_id}")
    student_id = assignment["student_id"]

    existing = get_latest_submission_by_worksheet(worksheet_id)
    if existing is None:
        # No prior submission — fall back to normal flow
        return process_submission(worksheet_id, score, "", answers_json)

    submission_id = existing["submission_id"]

    # Wipe old attempts
    delete_attempts_for_submission(submission_id)

    # Overwrite submission
    overwrite_submission(submission_id, score, answers_json)

    # Re-insert attempts
    questions = get_questions_for_worksheet(worksheet_id)
    attempts = []
    affected_skills = set()
    for ans in answers_json:
        q_index = ans.get("question_index", 0) - 1
        if 0 <= q_index < len(questions):
            q = questions[q_index]
            skill = q["skill_code"]
            attempts.append({
                "question_id": q["question_id"],
                "is_correct": ans.get("is_correct", False),
                "skill_code": skill,
            })
            affected_skills.add(skill)

    if attempts:
        insert_attempts(student_id, submission_id, worksheet_id, attempts)

    for skill_code in affected_skills:
        recalculate_skill_mastery(student_id, skill_code)

    return {
        "submission_id": submission_id,
        "attempts_count": len(attempts),
        "overwritten": True,
    }


def validate_sender(from_number: str) -> bool:
    """
    Flow: "check whether from_number is in users or not"

    Returns True if the phone number belongs to an active user, else False.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM users WHERE phone_number = %s AND is_active = true",
                (from_number,),
            )
            return cur.fetchone() is not None
