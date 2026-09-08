"""Worksheet/skill/student ingestion — ports the old standalone admin scripts
(insert_single_worksheet.py, bulk_insert_worksheets.py, sample_insert_omr_answer_variants.py,
import_students_from_csv.py) onto the new normalized schema.

Deliberately excludes worksheet *generation* (PDF/JSON authoring) — that stays with the old
scripts for now (plan items 4.2 covers generation, not insertion). This module only inserts
already-generated worksheet JSON (as produced by the old `worksheet_json_generator.py`) into
the new DB: `worksheets`/`questions`/`question_options` rather than the old
`worksheets.worksheet_json` blob + implicit `correct_option` field.
"""

import json
import re
from pathlib import Path
from string import ascii_uppercase

from sqlmodel import Session, select

from app.domain.errors import InvalidSubmissionDataError
from app.models import Question, QuestionOption, School, Skill, Student, Worksheet
from app.models.worksheet import QuestionPaperVariant, WorksheetCategory

DEFAULT_SKILLS_PATH = Path(__file__).resolve().parent.parent / "data" / "skills.json"

FILENAME_RE = re.compile(
    r"^(?:"
    r"(?P<worksheet_id>\d+)_(?P<language>[a-z]{2})|"
    r"(?P<language2>[a-z]{2})_(?P<category>practice|homework)_(?P<level>.+?)(?:_(?P<index>\d+))?|"
    r"(?P<category2>homework|practice)_(?P<level2>.+)|"
    r"(?P<worksheet_id2>\d+)_(?P<language3>[a-z]{2})_(?P<category3>practice|homework)"
    r")\.json$"
)


def parse_generated_filename(filename: str) -> dict:
    """Infer worksheet_id/language/category from a generated worksheet JSON filename."""
    name = Path(filename).name
    match = FILENAME_RE.match(name)
    if not match:
        raise ValueError(f"Unsupported generated worksheet filename: {filename!r}")

    data: dict = {"worksheet_id": None, "language": None, "worksheet_category": None}
    if match.group("worksheet_id"):
        data["worksheet_id"] = int(match.group("worksheet_id"))
        data["language"] = match.group("language")
    elif match.group("language2"):
        data["language"] = match.group("language2")
        data["worksheet_category"] = match.group("category")
    elif match.group("category2"):
        data["worksheet_category"] = match.group("category2")
    elif match.group("worksheet_id2"):
        data["worksheet_id"] = int(match.group("worksheet_id2"))
        data["language"] = match.group("language3")
        data["worksheet_category"] = match.group("category3")

    return data


def infer_worksheet_category(worksheet_json: dict, filename: str | None = None) -> str:
    """Determine worksheet category from the JSON payload, falling back to the filename."""
    if isinstance(worksheet_json, dict):
        category = worksheet_json.get("worksheet_category")
        if category in {c.value for c in WorksheetCategory}:
            return category

    if filename:
        name = Path(filename).name.lower()
        if "practice" in name:
            return "practice"
        if "homework" in name:
            return "homework"

    return "practice"


def worksheet_exists(session: Session, worksheet_id: int) -> bool:
    return session.get(Worksheet, worksheet_id) is not None


def seed_skills(session: Session, skills_path: Path | None = None) -> int:
    """Upsert the canonical skill catalog. Returns the number of skills processed.

    Required before inserting worksheets whose questions reference a `skill_code` not yet
    in the `skills` table, since `questions.skill_code` is a foreign key.
    """
    path = skills_path or DEFAULT_SKILLS_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else list(data.values())

    for entry in entries:
        skill_code = entry["code"]
        existing = session.get(Skill, skill_code)
        if existing is None:
            session.add(
                Skill(
                    skill_code=skill_code,
                    skill_name=entry["skill"],
                    skill_level=str(entry["difficulty_level"]),
                )
            )
        else:
            existing.skill_name = entry["skill"]
            existing.skill_level = str(entry["difficulty_level"])
            session.add(existing)

    session.commit()
    return len(entries)


def _ensure_placeholder_skill(session: Session, skill_code: str) -> None:
    """Auto-create an unrecognized skill_code as a placeholder so the FK insert doesn't fail.

    Mirrors the old system's `omr` placeholder-skill behavior for OMR sheets, which don't
    carry per-question skill codes.
    """
    if session.get(Skill, skill_code) is not None:
        return
    session.add(Skill(skill_code=skill_code, skill_name=f"Placeholder for {skill_code}", skill_level="1"))


def insert_worksheet(
    session: Session,
    worksheet_json: dict | list,
    worksheet_id: int | None = None,
    worksheet_category: str | None = None,
) -> dict:
    """Insert a worksheet and its questions/options. Returns {worksheet_id, question_ids}.

    Raises InvalidSubmissionDataError if worksheet_json has no questions.
    """
    questions = worksheet_json if isinstance(worksheet_json, list) else worksheet_json.get("questions", [])
    if not questions:
        raise InvalidSubmissionDataError("worksheet_json must contain at least one question.")

    payload = worksheet_json if isinstance(worksheet_json, dict) else {"questions": worksheet_json}
    category = worksheet_category or infer_worksheet_category(payload)
    if category not in {c.value for c in WorksheetCategory}:
        raise ValueError(f"worksheet_category must be one of {[c.value for c in WorksheetCategory]}, got {category!r}")

    worksheet = Worksheet(
        worksheet_id=worksheet_id,
        worksheet_level=payload.get("level"),
        lang=payload.get("language"),
        title=payload.get("title"),
        worksheet_category=category,
        max_score=len(questions),
        total_question_count=len(questions),
        worksheet_json=payload,
    )
    session.add(worksheet)
    session.flush()  # populate worksheet.worksheet_id (auto or explicit)

    if worksheet_id is not None:
        # Explicit-PK insert bypasses the sequence; keep it in sync for future auto-inserts.
        session.connection().exec_driver_sql(
            "SELECT setval(pg_get_serial_sequence('worksheets', 'worksheet_id'), "
            "(SELECT MAX(worksheet_id) FROM worksheets))"
        )

    question_ids = []
    for position, q in enumerate(questions, start=1):
        skill_code = q.get("skill_code") or q.get("skill") or "omr"
        _ensure_placeholder_skill(session, skill_code)

        question = Question(
            worksheet_id=worksheet.worksheet_id,
            skill_code=skill_code,
            index=q.get("index", position),
            question_json=q,
        )
        session.add(question)
        session.flush()  # populate question.question_id

        options = q.get("options") or []
        correct_option = q.get("correct_option")
        for offset, option_value in enumerate(options):
            label = ascii_uppercase[offset]
            session.add(
                QuestionOption(
                    question_id=question.question_id,
                    option_label=label,
                    option_value=str(option_value),
                    is_correct=(label == correct_option),
                )
            )

        question_ids.append(question.question_id)

    session.commit()
    return {"worksheet_id": worksheet.worksheet_id, "question_ids": question_ids}


def insert_question_paper_variant(
    session: Session,
    worksheet_id: int,
    question_paper_code: str,
    correct_labels: list[str],
) -> list[int]:
    """Seed a per-variant answer key: correct_labels[i] is the correct option for the
    question at index i+1 of the worksheet. Returns the created QuestionPaperVariant ids.

    basic_omr templates print fixed A-F variants sharing one answer key per copy — this is
    not a per-copy bubble-position shuffle, just an override on top of the canonical
    question_options.is_correct answer.
    """
    code = question_paper_code.strip().upper()
    if len(code) != 1 or code not in set(ascii_uppercase[:6]):
        raise ValueError("question_paper_code must be a single uppercase letter A-F")

    questions = session.exec(
        select(Question).where(Question.worksheet_id == worksheet_id).order_by(Question.index)
    ).all()
    if not questions:
        raise InvalidSubmissionDataError(f"No questions found for worksheet_id={worksheet_id}")
    if len(questions) != len(correct_labels):
        raise InvalidSubmissionDataError(
            f"correct_labels has {len(correct_labels)} entries but worksheet_id={worksheet_id} has {len(questions)} questions"
        )

    variant_ids = []
    for question, label in zip(questions, correct_labels):
        existing = session.exec(
            select(QuestionPaperVariant).where(
                QuestionPaperVariant.worksheet_id == worksheet_id,
                QuestionPaperVariant.question_paper_code == code,
                QuestionPaperVariant.question_id == question.question_id,
            )
        ).first()
        if existing is not None:
            existing.correct_option_label = label.strip().upper()
            session.add(existing)
            variant_ids.append(existing.id)
            continue

        variant = QuestionPaperVariant(
            worksheet_id=worksheet_id,
            question_paper_code=code,
            question_id=question.question_id,
            correct_option_label=label.strip().upper(),
        )
        session.add(variant)
        session.flush()
        variant_ids.append(variant.id)

    session.commit()
    return variant_ids


def upsert_school(session: Session, school_code: str, school_name: str) -> School:
    existing = session.get(School, school_code)
    if existing is not None:
        existing.school_name = school_name
        session.add(existing)
        return existing

    school = School(school_code=school_code, school_name=school_name)
    session.add(school)
    session.flush()
    return school


_FOUR_DIGIT_ID_RE = re.compile(r"^\d{4}$")


def next_student_id(session: Session) -> str:
    """Return the next default 4-digit student ID, starting at 0002 (0001 is reserved)."""
    existing = session.exec(select(Student.student_id)).all()
    numeric_ids = [int(sid) for sid in existing if sid and _FOUR_DIGIT_ID_RE.match(sid)]
    if not numeric_ids:
        return "0002"
    return str(max(numeric_ids) + 1).zfill(4)


def upsert_student(session: Session, student_id: str, student_name: str, school_code: str, current_level: str = "A") -> Student:
    normalized_id = student_id.strip().zfill(4)
    existing = session.get(Student, normalized_id)
    if existing is not None:
        return existing

    student = Student(
        student_id=normalized_id,
        student_name=student_name,
        student_school_code=school_code,
        current_level=current_level,
    )
    session.add(student)
    session.flush()
    return student
