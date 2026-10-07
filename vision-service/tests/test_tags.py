from app.vision.tags import (
    checksum,
    decode_row_tag_metadata,
    encode_worksheet_id_rows,
    validate_question_paper_code,
    worksheet_id_to_rows,
)


def test_encode_decode_roundtrip_legacy():
    worksheet_id = 12345
    rows = worksheet_id_to_rows(worksheet_id)
    assert len(rows) == 10
    metadata = decode_row_tag_metadata(rows)
    assert metadata == {
        "worksheet_id": worksheet_id,
        "page_no": 1,
        "first_question_index": 1,
        "format": "legacy",
    }


def test_encode_decode_roundtrip_omr_v2():
    worksheet_id = 999
    rows = worksheet_id_to_rows(worksheet_id, page_no=2, first_question_index=40)
    assert len(rows) == 13
    metadata = decode_row_tag_metadata(rows)
    assert metadata == {
        "worksheet_id": worksheet_id,
        "page_no": 2,
        "first_question_index": 40,
        "format": "omr_v2",
    }


def test_page_1_collapses_to_legacy_format():
    rows = worksheet_id_to_rows(42, page_no=1, first_question_index=1)
    assert len(rows) == 10


def test_bad_checksum_returns_none_worksheet_id():
    rows = worksheet_id_to_rows(555)
    corrupted = rows[:]
    corrupted[5] = (corrupted[5] + 1) % 35
    metadata = decode_row_tag_metadata(corrupted)
    assert metadata["worksheet_id"] is None
    assert metadata["format"] == "legacy"


def test_decode_row_tag_metadata_rejects_bad_length():
    import pytest

    with pytest.raises(ValueError):
        decode_row_tag_metadata([1, 2, 3])


def test_checksum_is_deterministic():
    ids = encode_worksheet_id_rows(100)
    assert checksum(ids) == checksum(ids)


def test_validate_question_paper_code():
    assert validate_question_paper_code("a") == "A"
    assert validate_question_paper_code(" F ") == "F"
    assert validate_question_paper_code("G") == ""
    assert validate_question_paper_code("AB") == ""
    assert validate_question_paper_code(None) == ""
    assert validate_question_paper_code(123) == ""


def test_shared_row_tags_roundtrip_across_id_range_and_pages():
    """The generator (api-service) and the scanner (vision-service) both use shared.row_tags --
    every printable id/page must decode back to itself and only use valid 25h9 tag ids (0..34)."""
    import random

    from shared.row_tags import MAX_WORKSHEET_ID

    rng = random.Random(0)
    ids = [0, 1, 34, 35, 4810, 99999, MAX_WORKSHEET_ID] + [rng.randint(0, MAX_WORKSHEET_ID) for _ in range(2000)]
    for worksheet_id in ids:
        for page_no, first_question_index in ((None, None), (1, 1), (2, 40)):
            rows = worksheet_id_to_rows(worksheet_id, page_no=page_no, first_question_index=first_question_index)
            assert all(0 <= tag <= 34 for tag in rows), (worksheet_id, rows)
            decoded = decode_row_tag_metadata(rows)
            assert decoded["worksheet_id"] == worksheet_id
            assert decoded["page_no"] == (page_no or 1)
            assert decoded["first_question_index"] == (first_question_index or 1)


def test_vision_tags_reexports_shared_implementation():
    import shared.row_tags as shared_row_tags
    from app.vision import tags

    assert tags.worksheet_id_to_rows is shared_row_tags.worksheet_id_to_rows
    assert tags.decode_row_tag_metadata is shared_row_tags.decode_row_tag_metadata
