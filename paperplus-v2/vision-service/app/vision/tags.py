"""Row-tag (25h9) encode/decode logic and misc validation. Pure functions, ported faithfully
from the old repo's services/image_service.py so worksheet_id/page metadata decoding stays
bit-compatible with already-printed worksheets.
"""

import hashlib

ORIENTATION_ID = 0


def encode_worksheet_id_rows(n: int) -> list[int]:
    """Encode a worksheet ID into 5 base-35 tag IDs (25h9 tags only allow IDs 0..34)."""
    digits = []
    for _ in range(5):
        digits.append(n % 35)
        n //= 35
    digits.reverse()
    return digits


def checksum(ids: list[int]) -> list[int]:
    """SHA256-based checksum of 5 tag IDs, returned as 5 base-35 digits."""
    digest = hashlib.sha256(bytes(ids)).digest()
    return [b % 35 for b in digest[:5]]


def worksheet_id_to_rows(n: int, page_no: int | None = None, first_question_index: int | None = None) -> list[int]:
    """Encode a worksheet ID into row tags, optionally including OMR v2 page metadata."""
    data_tags = encode_worksheet_id_rows(n)
    check = checksum(data_tags)
    legacy_tags = data_tags + check

    if page_no is None and first_question_index is None:
        return legacy_tags

    page_no_value = max(1, int(page_no or 1))
    first_question_index_value = max(1, int(first_question_index or 1))
    if page_no_value == 1:
        return legacy_tags

    low = first_question_index_value % 35
    high = first_question_index_value // 35
    return legacy_tags + [page_no_value, low, high]


def decode_row_tag_metadata(tags: list[int]) -> dict:
    """Decode worksheet_id + optional OMR v2 page metadata from row tags.

    Legacy format: exactly 10 tags (5 data + 5 checksum).
    OMR v2 format: exactly 13 tags (legacy packet + page_no + first_question_index base-35 pair).
    """
    if len(tags) == 10:
        data = tags[:5]
        check = tags[5:]
        if checksum(data) != check:
            return {"worksheet_id": None, "page_no": None, "first_question_index": None, "format": "legacy"}
        value = 0
        for d in data:
            value = value * 35 + d
        return {"worksheet_id": value, "page_no": 1, "first_question_index": 1, "format": "legacy"}

    if len(tags) == 13:
        legacy_data = tags[:5]
        legacy_check = tags[5:10]
        if checksum(legacy_data) != legacy_check:
            return {"worksheet_id": None, "page_no": None, "first_question_index": None, "format": "omr_v2"}

        value = 0
        for d in legacy_data:
            value = value * 35 + d

        page_no = int(tags[10]) if 0 <= tags[10] <= 9 else 1
        low = int(tags[11]) if 0 <= tags[11] <= 34 else 0
        high = int(tags[12]) if 0 <= tags[12] <= 34 else 0
        first_question_index = low + high * 35
        if page_no <= 0:
            page_no = 1
        if page_no == 1 and first_question_index != 1:
            first_question_index = 1
        if first_question_index <= 0:
            first_question_index = 1
        return {
            "worksheet_id": value,
            "page_no": page_no,
            "first_question_index": first_question_index,
            "format": "omr_v2",
        }

    raise ValueError("Expected 10 tags (legacy) or 13 tags (OMR v2 metadata format)")


def decode_row_tags(tags: list[int]) -> int | None:
    return decode_row_tag_metadata(tags).get("worksheet_id")


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
