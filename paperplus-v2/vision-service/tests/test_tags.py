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
