"""Row-tag helpers and misc validation for the scan pipeline. The 25h9 row-tag encoder/decoder
itself lives in shared/row_tags.py (also used by api-service's worksheet PDF generation) and is
re-exported here so existing imports keep working.
"""

from shared.row_tags import (  # noqa: F401 -- re-exported; the encoder/decoder lives in shared/
    checksum,
    decode_row_tag_metadata,
    decode_row_tags,
    encode_worksheet_id_rows,
    worksheet_id_to_rows,
)

ORIENTATION_ID = 0


def rotate(lst: list, n: int) -> list:
    return lst[-n:] + lst[:-n]


def validate_question_paper_code(raw_value: str | None) -> str:
    """A question paper code must be a single uppercase letter in A-F; anything else -> ""."""
    if not isinstance(raw_value, str):
        return ""
    normalized = raw_value.strip().upper()
    if len(normalized) == 1 and normalized in {"A", "B", "C", "D", "E", "F"}:
        return normalized
    return ""
