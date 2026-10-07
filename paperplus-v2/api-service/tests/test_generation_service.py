"""generate_worksheets / render_existing against the real dev DB (ids from a reserved high range,
cleaned up after each test)."""

import random

import pypdfium2 as pdfium
import pytest
from sqlmodel import Session, delete, select

import app.generation.service as service
from app.db.session import engine
from app.generation.composition import compose_worksheet
from app.generation.service import GenerationError, generate_worksheets, render_existing
from app.models import Question, QuestionOption, Worksheet, WorksheetPage

TEST_ID = 52_400_000  # far above real ids, still encodable in row tags (max 52,521,874)


@pytest.fixture
def session():
    with Session(engine) as s:
        yield s


@pytest.fixture
def cleanup(session):
    ids: list[int] = []
    yield ids
    session.rollback()
    for worksheet_id in ids:
        session.exec(delete(WorksheetPage).where(WorksheetPage.worksheet_id == worksheet_id))
        for question in session.exec(select(Question).where(Question.worksheet_id == worksheet_id)).all():
            session.exec(delete(QuestionOption).where(QuestionOption.question_id == question.question_id))
        session.exec(delete(Question).where(Question.worksheet_id == worksheet_id))
        session.exec(delete(Worksheet).where(Worksheet.worksheet_id == worksheet_id))
    session.commit()


def _answer_key(session: Session, worksheet_id: int) -> dict[int, str]:
    key = {}
    for question in session.exec(select(Question).where(Question.worksheet_id == worksheet_id)).all():
        options = session.exec(select(QuestionOption).where(QuestionOption.question_id == question.question_id)).all()
        (correct,) = [o.option_label for o in options if o.is_correct]
        key[question.index] = correct
    return key


def test_homework_batch_is_inserted_seeded_and_written(session, cleanup, tmp_path):
    cleanup += [TEST_ID, TEST_ID + 1]
    result = generate_worksheets(
        session, worksheet_type="homework", level="C", language="mr", count=2,
        start_id=TEST_ID, seed=500, output_dir=tmp_path, merge=True,
    )
    assert [w.worksheet_id for w in result.worksheets] == [TEST_ID, TEST_ID + 1]
    for offset, item in enumerate(result.worksheets):
        expected = compose_worksheet("homework", "C", "mr", random.Random(500 + offset))
        worksheet = session.get(Worksheet, item.worksheet_id)
        assert worksheet.worksheet_json == expected
        assert worksheet.worksheet_metadata["generator"] == {"version": service.GENERATOR_VERSION, "seed": 500 + offset}
        assert _answer_key(session, item.worksheet_id) == {q["index"]: q["correct_option"] for q in expected["questions"]}
        assert [p.name for p in item.pdf_paths] == [f"{item.worksheet_id}_mr_homework_C.pdf"]
    assert result.merged_path.name == f"{TEST_ID}to{TEST_ID + 1}_mr_homework_C_print.pdf"
    assert len(pdfium.PdfDocument(str(result.merged_path))) == 2


def test_omr_worksheets_get_two_pages_each(session, cleanup, tmp_path):
    cleanup.append(TEST_ID)
    result = generate_worksheets(session, worksheet_type="omr", language="en", count=1, start_id=TEST_ID, output_dir=tmp_path)
    (item,) = result.worksheets
    assert [p.name for p in item.pdf_paths] == [f"{TEST_ID}_en_omr_BASIC_OMR_page1.pdf", f"{TEST_ID}_en_omr_BASIC_OMR_page2.pdf"]
    pages = session.exec(select(WorksheetPage).where(WorksheetPage.worksheet_id == TEST_ID)).all()
    assert sorted((p.page_no, p.first_question_index, p.last_question_index) for p in pages) == [(1, 1, 39), (2, 40, 78)]
    assert "seed" not in session.get(Worksheet, TEST_ID).worksheet_metadata["generator"]


def test_database_assigned_id_is_the_one_printed(session, cleanup, tmp_path, monkeypatch):
    printed_ids = []
    real_render = service.render_worksheet_pdfs
    monkeypatch.setattr(service, "render_worksheet_pdfs", lambda wid, *a: printed_ids.append(wid) or real_render(wid, *a))
    result = generate_worksheets(session, worksheet_type="practice", level="A2", language="en", count=1, output_dir=tmp_path)
    cleanup.append(result.worksheets[0].worksheet_id)
    assert printed_ids == [result.worksheets[0].worksheet_id]
    assert session.get(Worksheet, printed_ids[0]) is not None


def test_taken_ids_are_refused_with_nothing_written(session, cleanup, tmp_path):
    cleanup.append(TEST_ID + 1)
    generate_worksheets(session, worksheet_type="omr", language="en", count=1, start_id=TEST_ID + 1, output_dir=tmp_path / "first")
    with pytest.raises(GenerationError, match="already exist"):
        generate_worksheets(session, worksheet_type="homework", level="A", language="en", count=3, start_id=TEST_ID, output_dir=tmp_path / "second")
    assert session.get(Worksheet, TEST_ID) is None
    assert not (tmp_path / "second").exists()


def test_render_failure_rolls_back_the_whole_batch(session, cleanup, tmp_path, monkeypatch):
    cleanup += [TEST_ID, TEST_ID + 1]
    real_render = service.render_worksheet_pdfs

    def fail_on_second(worksheet_id, *args):
        if worksheet_id == TEST_ID + 1:
            raise RuntimeError("weasyprint exploded")
        return real_render(worksheet_id, *args)

    monkeypatch.setattr(service, "render_worksheet_pdfs", fail_on_second)
    with pytest.raises(RuntimeError):
        generate_worksheets(session, worksheet_type="homework", level="A", language="en", count=2, start_id=TEST_ID, output_dir=tmp_path)
    assert session.get(Worksheet, TEST_ID) is None and session.get(Worksheet, TEST_ID + 1) is None
    assert list(tmp_path.iterdir()) == []


def test_write_failure_removes_files_already_written(session, cleanup, tmp_path, monkeypatch):
    cleanup += [TEST_ID, TEST_ID + 1]
    monkeypatch.setattr(service, "_write_merged", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        generate_worksheets(session, worksheet_type="homework", level="B", language="en", count=2,
                            start_id=TEST_ID, output_dir=tmp_path, merge=True)
    assert session.get(Worksheet, TEST_ID) is None
    assert list(tmp_path.iterdir()) == []


def test_dry_run_renders_but_writes_nothing(session, cleanup, tmp_path):
    cleanup.append(TEST_ID)
    result = generate_worksheets(session, worksheet_type="homework", level="G", language="en", count=1,
                                 start_id=TEST_ID, output_dir=tmp_path / "out", dry_run=True)
    assert result.worksheets[0].pdfs[0].startswith(b"%PDF")
    assert session.get(Worksheet, TEST_ID) is None
    assert not (tmp_path / "out").exists()


def test_reprint_renders_existing_worksheets_from_the_db(session, cleanup, tmp_path):
    cleanup += [TEST_ID, TEST_ID + 1]
    generate_worksheets(session, worksheet_type="homework", level="E", language="en", count=1, start_id=TEST_ID, seed=1, output_dir=tmp_path / "a")
    generate_worksheets(session, worksheet_type="omr", language="en", count=1, start_id=TEST_ID + 1, output_dir=tmp_path / "a")
    result = render_existing(session, [TEST_ID, TEST_ID + 1], tmp_path / "b", merge=True)
    assert {p.name for p in (tmp_path / "b").iterdir()} == {p.name for p in (tmp_path / "a").iterdir()} | {result.merged_path.name}
    assert len(pdfium.PdfDocument(str(result.merged_path))) == 3
    with pytest.raises(GenerationError, match="not in the database"):
        render_existing(session, [TEST_ID + 2], tmp_path / "c")


def test_unseeded_skills_and_bad_requests_are_refused(session):
    with pytest.raises(GenerationError, match="seed_skills"):
        service._check_skills_seeded(session, [{"questions": [{"skill_code": "NOT_A_SKILL"}]}])
    for kwargs in [dict(worksheet_type="homework", level="Q"), dict(worksheet_type="essay", level="A"),
                   dict(worksheet_type="practice", level=None), dict(worksheet_type="homework", level="A", count=0)]:
        with pytest.raises(GenerationError):
            generate_worksheets(session, language="en", output_dir=None, **{"count": 1, **kwargs})
    with pytest.raises(GenerationError, match="between 0"):
        generate_worksheets(session, worksheet_type="omr", language="en", count=2, start_id=52_521_874, output_dir=None)
