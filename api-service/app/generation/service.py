"""Worksheet generation end to end: compose -> insert -> render -> write -> commit.

The worksheet row is inserted *before* its PDF is rendered, so the PDF's row tags always carry
the id the database actually holds (the old generator printed a placeholder id 0 unless an id was
passed, and inserting the JSON was a separate, forgettable step). The batch is committed only
after every PDF has rendered and been written; on any failure the DB is rolled back and the
files written so far are deleted, so a worksheet never exists only on paper or only in the DB.

Each worksheet gets its own seed (batch seed + position), stored in worksheet_metadata, so any
single worksheet can be regenerated identically. Reprints don't need it: render_existing()
renders straight from the stored worksheet_json.
"""

import logging
import random
import secrets
from dataclasses import dataclass, field
from pathlib import Path

import pypdfium2 as pdfium
from sqlmodel import Session, select

from app.domain.worksheet_import import insert_worksheet
from app.generation.composition import compose_omr_worksheet, compose_worksheet
from app.generation.render import page_specs, render_worksheet_pdfs
from app.models import Skill, Worksheet, WorksheetPage, WorksheetTemplate
from shared.row_tags import MAX_WORKSHEET_ID
from shared.worksheet_templates import QUESTIONS_PER_PAGE

logger = logging.getLogger(__name__)

# Bump when a change would make the same seed produce a different worksheet.
GENERATOR_VERSION = 1
WORKSHEET_TYPES = ("homework", "practice", "omr")
DEFAULT_OMR_QUESTION_COUNT = 2 * QUESTIONS_PER_PAGE["basic_omr"]


class GenerationError(Exception):
    """A batch was refused before anything was written."""


@dataclass
class RenderedWorksheet:
    worksheet_id: int
    pdfs: list[bytes]
    pdf_paths: list[Path] = field(default_factory=list)


@dataclass
class GenerationResult:
    worksheets: list[RenderedWorksheet]
    batch_seed: int | None
    merged_path: Path | None = None


def _template_name(worksheet_type: str) -> str:
    return "basic_omr" if worksheet_type == "omr" else "regular"


def _pdf_filename(worksheet_id: int, worksheet_json: dict, page_no: int, page_count: int) -> str:
    """Same naming as the old generator: <id>_<lang>_<type>_<LEVEL>[_page<n>].pdf."""
    language = worksheet_json.get("language") or "en"
    category = worksheet_json.get("worksheet_category") or "practice"
    level = "BASIC_OMR" if category == "omr" else str(worksheet_json.get("level") or "").upper()
    page_suffix = f"_page{page_no}" if page_count > 1 else ""
    return f"{worksheet_id}_{language}_{category}_{level}{page_suffix}.pdf"


def _check_ids_free(session: Session, start_id: int, count: int) -> None:
    ids = list(range(start_id, start_id + count))
    if start_id < 0 or ids[-1] > MAX_WORKSHEET_ID:
        raise GenerationError(f"Worksheet ids must be between 0 and {MAX_WORKSHEET_ID} (what the row tags can encode).")
    taken = session.exec(select(Worksheet.worksheet_id).where(Worksheet.worksheet_id.in_(ids))).all()
    if taken:
        raise GenerationError(
            f"Worksheet id(s) already exist, nothing generated: {sorted(taken)}. "
            "Pick a free --start-id, or reprint existing sheets with render_worksheet_pdf.py."
        )


def _check_skills_seeded(session: Session, worksheets: list[dict]) -> None:
    """insert_worksheet would silently create placeholder skills for unknown codes; generated
    sheets must only use real catalog skills, so refuse instead."""
    needed = {q["skill_code"] for sheet in worksheets for q in sheet["questions"] if q.get("skill_code")}
    if not needed:
        return
    known = set(session.exec(select(Skill.skill_code).where(Skill.skill_code.in_(sorted(needed)))).all())
    missing = sorted(needed - known)
    if missing:
        raise GenerationError(f"Skills missing from the database: {missing}. Run scripts/seed_skills.py first.")


def _write_merged(pdf_pages: list[bytes], path: Path) -> None:
    merged = pdfium.PdfDocument.new()
    for pdf_bytes in pdf_pages:
        source = pdfium.PdfDocument(pdf_bytes)
        merged.import_pages(source)
    merged.save(str(path))


def _write_outputs(rendered: list[RenderedWorksheet], worksheet_jsons: list[dict], output_dir: Path,
                   merge_name: str | None, written: list[Path]) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for item, worksheet_json in zip(rendered, worksheet_jsons):
        for page_no, pdf_bytes in enumerate(item.pdfs, start=1):
            path = output_dir / _pdf_filename(item.worksheet_id, worksheet_json, page_no, len(item.pdfs))
            path.write_bytes(pdf_bytes)
            written.append(path)
            item.pdf_paths.append(path)
    if merge_name is None:
        return None
    merged_path = output_dir / merge_name
    _write_merged([pdf for item in rendered for pdf in item.pdfs], merged_path)
    written.append(merged_path)
    return merged_path


def _merge_name(rendered: list[RenderedWorksheet], worksheet_json: dict) -> str:
    first, last = rendered[0].worksheet_id, rendered[-1].worksheet_id
    language = worksheet_json.get("language") or "en"
    category = worksheet_json.get("worksheet_category") or "practice"
    level = "BASIC_OMR" if category == "omr" else str(worksheet_json.get("level") or "").upper()
    return f"{first}to{last}_{language}_{category}_{level}_print.pdf"


def _render_inserted(session: Session, worksheet_id: int, worksheet_json: dict, template_name: str) -> list[bytes]:
    pages = session.exec(select(WorksheetPage).where(WorksheetPage.worksheet_id == worksheet_id)).all()
    specs = page_specs(template_name, len(worksheet_json["questions"]), pages)
    return render_worksheet_pdfs(worksheet_id, worksheet_json, template_name, specs)


def generate_worksheets(
    session: Session,
    *,
    worksheet_type: str,
    language: str,
    count: int,
    output_dir: Path,
    level: str | None = None,
    start_id: int | None = None,
    seed: int | None = None,
    title: str | None = None,
    omr_question_count: int = DEFAULT_OMR_QUESTION_COUNT,
    merge: bool = False,
    dry_run: bool = False,
) -> GenerationResult:
    """Generate `count` worksheets, insert them, and write their PDFs to output_dir.

    start_id=None lets the database assign ids. dry_run renders everything, then rolls back and
    writes no files. Raises GenerationError (nothing written) for an invalid request.
    """
    if worksheet_type not in WORKSHEET_TYPES:
        raise GenerationError(f"worksheet_type must be one of {WORKSHEET_TYPES}, got {worksheet_type!r}")
    if count < 1:
        raise GenerationError("count must be at least 1")
    if worksheet_type != "omr" and not level:
        raise GenerationError("level is required for homework and practice worksheets")
    if worksheet_type == "omr" and omr_question_count < 1:
        raise GenerationError("omr_question_count must be at least 1")
    if start_id is not None:
        _check_ids_free(session, start_id, count)

    template_name = _template_name(worksheet_type)
    batch_seed = None
    worksheet_jsons: list[dict] = []
    seeds: list[int | None] = []
    if worksheet_type == "omr":
        worksheet_jsons = [compose_omr_worksheet(omr_question_count, language=language, title=title) for _ in range(count)]
        seeds = [None] * count
    else:
        batch_seed = seed if seed is not None else secrets.randbits(32)
        for index in range(count):
            sheet_seed = batch_seed + index
            try:
                worksheet_jsons.append(compose_worksheet(worksheet_type, level, language, random.Random(sheet_seed), title=title))
            except ValueError as exc:
                raise GenerationError(str(exc)) from exc
            seeds.append(sheet_seed)
        _check_skills_seeded(session, worksheet_jsons)

    written: list[Path] = []
    try:
        rendered = []
        for index, worksheet_json in enumerate(worksheet_jsons):
            inserted = insert_worksheet(
                session,
                worksheet_json,
                worksheet_id=None if start_id is None else start_id + index,
                worksheet_category=worksheet_type,
                template_name=template_name,
                commit=False,
            )
            worksheet_id = inserted["worksheet_id"]
            worksheet = session.get(Worksheet, worksheet_id)
            generator_info = {"version": GENERATOR_VERSION}
            if seeds[index] is not None:
                generator_info["seed"] = seeds[index]
            worksheet.worksheet_metadata = {**(worksheet.worksheet_metadata or {}), "generator": generator_info}
            session.add(worksheet)
            session.flush()
            rendered.append(RenderedWorksheet(worksheet_id, _render_inserted(session, worksheet_id, worksheet_json, template_name)))

        if dry_run:
            session.rollback()
            return GenerationResult(rendered, batch_seed)

        merged_path = _write_outputs(
            rendered, worksheet_jsons, output_dir,
            _merge_name(rendered, worksheet_jsons[0]) if merge else None, written,
        )
        session.commit()
    except Exception:
        session.rollback()
        for path in written:
            path.unlink(missing_ok=True)
        raise

    logger.info(
        "Generated %s %s worksheet(s) %s-%s (seed=%s)",
        len(rendered), worksheet_type, rendered[0].worksheet_id, rendered[-1].worksheet_id, batch_seed,
    )
    return GenerationResult(rendered, batch_seed, merged_path)


def render_existing(session: Session, worksheet_ids: list[int], output_dir: Path, merge: bool = False) -> GenerationResult:
    """Reprint worksheets already in the database from their stored worksheet_json."""
    found = {w.worksheet_id: w for w in session.exec(select(Worksheet).where(Worksheet.worksheet_id.in_(worksheet_ids))).all()}
    missing = [i for i in worksheet_ids if i not in found]
    if missing:
        raise GenerationError(f"Worksheet id(s) not in the database: {missing}")

    rendered, worksheet_jsons = [], []
    for worksheet_id in worksheet_ids:
        worksheet = found[worksheet_id]
        if not worksheet.worksheet_json or not worksheet.worksheet_json.get("questions"):
            raise GenerationError(f"Worksheet {worksheet_id} has no stored questions to render.")
        template = session.get(WorksheetTemplate, worksheet.template_id) if worksheet.template_id else None
        template_name = template.name if template else _template_name(worksheet.worksheet_category)
        worksheet_json = {"worksheet_category": worksheet.worksheet_category, **worksheet.worksheet_json}
        rendered.append(RenderedWorksheet(worksheet_id, _render_inserted(session, worksheet_id, worksheet_json, template_name)))
        worksheet_jsons.append(worksheet_json)

    written: list[Path] = []
    try:
        merged_path = _write_outputs(
            rendered, worksheet_jsons, output_dir,
            _merge_name(rendered, worksheet_jsons[0]) if merge else None, written,
        )
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return GenerationResult(rendered, None, merged_path)
