"""Composite / high-level database workflows.

These orchestrate multiple domain modules in a single logical operation.
"""

import json

from .connection import get_connection
from .worksheets import create_worksheet, get_worksheet_json
from .questions import insert_questions_for_worksheet, get_questions_for_worksheet
from .submissions import (
    create_submission,
    get_latest_submission_by_worksheet,
    overwrite_submission,
)
from .attempts import insert_attempts, delete_attempts_for_submission
from .mastery import recalculate_skill_mastery


def add_worksheet_to_db(worksheet_json: dict | list,
                        is_test: bool = False,
                        worksheet_id: int = None) -> dict:
    """
    Full flow: "add a new worksheet to the database"

    1. Insert worksheet row → get worksheet_id
    2. Embed worksheet_id into the JSON
    3. Extract each question, insert into questions table
    4. Return {worksheet_id, question_ids}
    """
    questions = worksheet_json if isinstance(worksheet_json, list) else worksheet_json.get("questions", [])

    worksheet_id = create_worksheet(worksheet_json, is_test=is_test, worksheet_id=worksheet_id)
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


# note: this function does not do any checking.
def process_submission(student_id: str, worksheet_id: int, score: int,
                       from_number: str, answers_json: list[dict]) -> dict:
    """
    Full flow: "receive a submission from a student / process the worksheet"

    1. Create submission record
    2. Build attempt records from answers_json
    3. Insert attempts
    4. Recalculate skill mastery for affected skills
    5. Return {submission_id, student_id, attempts_count}
    """
    submission_id = create_submission(
        student_id, worksheet_id, score, from_number, answers_json
    )

    questions = get_questions_for_worksheet(worksheet_id)
    attempts = []
    affected_skills = set()
    for ans in answers_json:
        q_index = ans.get("question_index", 1)
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

    for skill_code in affected_skills:
        recalculate_skill_mastery(student_id, skill_code)

    return {
        "submission_id": submission_id,
        "student_id": student_id,
        "attempts_count": len(attempts),
    }


def overwrite_submission_flow(student_id: str, worksheet_id: int, score: int,
                              answers_json: list[dict]) -> dict:
    """
    Manual correction flow: "if already exists, overwrite it"

    1. Find existing submission by worksheet
    2. Delete old attempts
    3. Overwrite submission score/answers
    4. Re-insert new attempts
    5. Recalculate mastery
    """
    existing = get_latest_submission_by_worksheet(worksheet_id)
    if existing is None:
        return process_submission(student_id, worksheet_id, score, "", answers_json)

    submission_id = existing["submission_id"]

    delete_attempts_for_submission(submission_id)
    overwrite_submission(submission_id, score, answers_json)

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
    Check whether from_number belongs to an active user.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM users WHERE phone_number = %s AND is_active = true",
                (from_number,),
            )
            return cur.fetchone() is not None
