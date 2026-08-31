"""Worksheet CRUD operations."""

import json

from .connection import get_connection


def ensure_worksheet_page_support() -> None:
    """Apply the additive OMR metadata columns if they are missing."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE worksheets ADD COLUMN IF NOT EXISTS sheet_version text NOT NULL DEFAULT 'legacy'")
            cur.execute("ALTER TABLE worksheets ADD COLUMN IF NOT EXISTS page_count integer NOT NULL DEFAULT 1")
            cur.execute("ALTER TABLE worksheets ADD COLUMN IF NOT EXISTS total_question_count integer")
            cur.execute("ALTER TABLE worksheets ADD COLUMN IF NOT EXISTS worksheet_metadata jsonb NOT NULL DEFAULT '{}'::jsonb")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS worksheet_pages (
                    worksheet_page_id serial PRIMARY KEY,
                    worksheet_id integer NOT NULL REFERENCES worksheets(worksheet_id) ON DELETE CASCADE,
                    page_no integer NOT NULL CHECK (page_no >= 1),
                    first_question_index integer NOT NULL CHECK (first_question_index >= 1),
                    last_question_index integer NOT NULL CHECK (last_question_index >= first_question_index),
                    expected_row_tag_count integer NOT NULL DEFAULT 10,
                    page_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
                    UNIQUE (worksheet_id, page_no)
                )
            """)
            cur.execute("UPDATE worksheets SET sheet_version = 'legacy' WHERE sheet_version IS NULL")
            cur.execute("UPDATE worksheets SET page_count = 1 WHERE page_count IS NULL")
            cur.execute("UPDATE worksheets SET total_question_count = COALESCE(total_question_count, max_score) WHERE total_question_count IS NULL")


def create_worksheet(worksheet_json: dict, is_test: bool = False,
                    worksheet_id: int = None,
                    worksheet_category: str = "practice") -> int:
    """Insert a new worksheet row and return the worksheet_id.

    Supports the legacy worksheet categories as well as the new OMR category.
    Legacy records may have language/level metadata while OMR records may omit them.
    """
    if worksheet_category not in {"practice", "homework", "omr"}:
        raise ValueError("worksheet_category must be 'practice', 'homework', or 'omr'")

    ensure_worksheet_page_support()

    lang = worksheet_json.get("language")
    level = worksheet_json.get("level")
    title = worksheet_json.get("title")
    question_count = len(worksheet_json.get("questions", []))
    max_score = question_count or worksheet_json.get("question_count") or 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            if worksheet_id is not None:
                cur.execute(
                    """INSERT INTO worksheets (worksheet_id, worksheet_level, is_test, worksheet_category, max_score, lang, worksheet_json, total_question_count, sheet_version, page_count)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       RETURNING worksheet_id""",
                    (worksheet_id, level, is_test, worksheet_category, max_score, lang, json.dumps(worksheet_json), question_count, "omr" if worksheet_category == "omr" else "legacy", 1),
                )
                inserted_id = cur.fetchone()["worksheet_id"]
                cur.execute("SELECT setval(pg_get_serial_sequence('worksheets', 'worksheet_id'), (SELECT MAX(worksheet_id) FROM worksheets))")
                return inserted_id
            else:
                cur.execute(
                    """INSERT INTO worksheets (worksheet_level, is_test, worksheet_category, max_score, lang, worksheet_json, total_question_count, sheet_version, page_count)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                       RETURNING worksheet_id""",
                    (level, is_test, worksheet_category, max_score, lang, json.dumps(worksheet_json), question_count, "omr" if worksheet_category == "omr" else "legacy", 1),
                )
                return cur.fetchone()["worksheet_id"]


def check_test(worksheet_id: int) -> bool:
    """Check if a worksheet is a test."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT is_test FROM worksheets WHERE worksheet_id = %s", (worksheet_id,))
            row = cur.fetchone()
            return row["is_test"] if row else False


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


def get_worksheet_page(worksheet_id: int, page_no: int) -> dict | None:
    """Return a single worksheet page record for a worksheet/page pair."""
    ensure_worksheet_page_support()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM worksheet_pages WHERE worksheet_id = %s AND page_no = %s ORDER BY page_no",
                (worksheet_id, page_no),
            )
            row = cur.fetchone()
            return row


def get_worksheet_pages(worksheet_id: int) -> list[dict]:
    """Return all page metadata rows belonging to a worksheet, ordered by page number."""
    ensure_worksheet_page_support()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM worksheet_pages WHERE worksheet_id = %s ORDER BY page_no",
                (worksheet_id,),
            )
            return cur.fetchall()


def get_answer_key(worksheet_id: int) -> list | None:
    """Extract the answer key from a worksheet's JSON."""
    ws_json = get_worksheet_json(worksheet_id)
    if ws_json is None:
        return None
    questions = ws_json if isinstance(ws_json, list) else ws_json.get("questions", [])
    return [q["correct_option"] for q in questions]


def save_omr_answer_key(template_name: str, question_paper_code: str, answer_key: list[str], worksheet_id: int | None = None) -> int:
    """Persist a template-specific answer key for a given OMR paper code."""
    if not isinstance(template_name, str) or not template_name.strip():
        raise ValueError("template_name is required")

    normalized_code = question_paper_code.strip().upper() if isinstance(question_paper_code, str) else ""
    if len(normalized_code) != 1 or normalized_code not in {"A", "B", "C", "D", "E", "F"}:
        raise ValueError("question_paper_code must be a single uppercase letter A-F")

    normalized_key = list(answer_key or [])
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS omr_answer_sets (
                    id serial PRIMARY KEY,
                    template_name text NOT NULL,
                    question_paper_code text NOT NULL,
                    worksheet_id integer,
                    answer_key_json jsonb NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    UNIQUE (template_name, question_paper_code)
                )
                """
            )
            cur.execute(
                """
                INSERT INTO omr_answer_sets (template_name, question_paper_code, worksheet_id, answer_key_json)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (template_name, question_paper_code)
                DO UPDATE SET
                    worksheet_id = EXCLUDED.worksheet_id,
                    answer_key_json = EXCLUDED.answer_key_json
                RETURNING id
                """,
                (template_name.strip().lower(), normalized_code, worksheet_id, json.dumps(normalized_key)),
            )
            row = cur.fetchone()
            return int(row["id"])


def get_omr_answer_key(template_name: str, question_paper_code: str) -> list[str] | None:
    """Return the stored answer key for a template/paper code pair, if present."""
    if not template_name or not question_paper_code:
        return None

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT answer_key_json FROM omr_answer_sets WHERE template_name = %s AND question_paper_code = %s",
                (str(template_name).strip().lower(), str(question_paper_code).strip().upper()),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return list(row["answer_key_json"])


def resolve_answer_key_for_template(template_name: str, worksheet_id: int | None, question_paper_code: str | None = None) -> list[str] | None:
    """Resolve the answer key for a worksheet, preferring the paper-code mapping for OMR templates."""
    if template_name and str(template_name).strip().lower() == "basic_omr":
        if question_paper_code:
            return get_omr_answer_key(template_name, question_paper_code)
        if worksheet_id is not None:
            return get_answer_key(worksheet_id)
        return None

    if worksheet_id is None:
        return None
    return get_answer_key(worksheet_id)


def upsert_worksheet_page(worksheet_id: int, page_no: int, first_question_index: int,
                          last_question_index: int | None = None,
                          expected_row_tag_count: int = 10,
                          page_metadata: dict | None = None) -> dict:
    """Insert or replace a worksheet page record for OMR print flows."""
    ensure_worksheet_page_support()
    last_question_index = last_question_index or first_question_index
    page_metadata = page_metadata or {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO worksheet_pages (worksheet_id, page_no, first_question_index, last_question_index, expected_row_tag_count, page_metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (worksheet_id, page_no)
                DO UPDATE SET
                    first_question_index = EXCLUDED.first_question_index,
                    last_question_index = EXCLUDED.last_question_index,
                    expected_row_tag_count = EXCLUDED.expected_row_tag_count,
                    page_metadata = EXCLUDED.page_metadata
                RETURNING *
                """,
                (worksheet_id, page_no, first_question_index, last_question_index, expected_row_tag_count, json.dumps(page_metadata)),
            )
            row = cur.fetchone()
            cur.execute(
                "UPDATE worksheets SET page_count = GREATEST(COALESCE(page_count, 1), %s), worksheet_metadata = COALESCE(worksheet_metadata, '{}'::jsonb) || %s WHERE worksheet_id = %s",
                (page_no, json.dumps({"last_page_no": page_no, "last_question_index": last_question_index}), worksheet_id),
            )
            return row


def record_worksheet_page_metadata(worksheet_id: int, page_no: int | None, first_question_index: int | None,
                                  page_metadata: dict | None = None, expected_row_tag_count: int = 10) -> None:
    """Persist page metadata for a scanned OMR sheet if the sheet is multi-page or page-aware."""
    if worksheet_id is None or page_no is None or first_question_index is None:
        return
    try:
        upsert_worksheet_page(
            worksheet_id=worksheet_id,
            page_no=int(page_no),
            first_question_index=int(first_question_index),
            last_question_index=int(first_question_index) + 39,
            expected_row_tag_count=expected_row_tag_count,
            page_metadata=page_metadata or {},
        )
    except Exception:
        # Keep scanning tolerant of missing or unavailable database services.
        pass


def list_worksheets(level: str = None, lang: str = None,
                   is_test: bool = None,
                   worksheet_category: str = None,
                   sheet_version: str = None,
                   page_count: int = None) -> list[dict]:
    """List worksheets with optional filters, including OMR compatibility fields."""
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
    if worksheet_category:
        clauses.append("worksheet_category = %s")
        params.append(worksheet_category)
    if sheet_version:
        clauses.append("sheet_version = %s")
        params.append(sheet_version)
    if page_count is not None:
        clauses.append("page_count = %s")
        params.append(page_count)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM worksheets {where} ORDER BY worksheet_id", params)
            return cur.fetchall()
