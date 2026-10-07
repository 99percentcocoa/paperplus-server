"""Properties of generated worksheets that must hold for every level, language and seed --
independent of the old code (see test_generation_parity.py for exact equivalence with it)."""

import random
import re

import pytest

from app.generation.composition import (
    HOMEWORK_LEVELS,
    PRACTICE_THEME_SKILLS,
    QUESTIONS_PER_WORKSHEET,
    compose_omr_worksheet,
    compose_worksheet,
    load_skill_catalog,
    parse_practice_level,
)
from app.generation.distractors import DISTRACTORS
from app.generation.localization import arabic_to_devanagari
from app.generation.questions import GENERATORS

ALL_SHEETS = [("homework", level) for level in HOMEWORK_LEVELS] + [
    ("practice", f"{theme}{level}") for theme, levels in PRACTICE_THEME_SKILLS.items() for level in levels
]
_FROM_DEVANAGARI = str.maketrans("०१२३४५६७८९", "0123456789")
_QUESTION = re.compile(r"^(\d+) ([+\-×÷]) (\d+)$")


def expected_answer(question_text: str) -> str:
    """Recompute the answer from the printed question text alone."""
    a, op, b = _QUESTION.match(question_text.translate(_FROM_DEVANAGARI)).groups()
    a, b = int(a), int(b)
    if op == "+":
        return str(a + b)
    if op == "-":
        return str(a - b)
    if op == "×":
        return str(a * b)
    return str(a // b) if a % b == 0 else f"{a // b}R{a % b}"


def test_every_skill_has_a_generator_and_distractors():
    catalog = {s["code"] for s in load_skill_catalog()}
    assert set(GENERATORS) == catalog
    assert set(DISTRACTORS) == catalog
    practice_skills = {code for levels in PRACTICE_THEME_SKILLS.values() for codes in levels.values() for code in codes}
    assert practice_skills <= catalog


@pytest.mark.parametrize("language", ["en", "mr"])
@pytest.mark.parametrize("worksheet_type,level", ALL_SHEETS)
def test_generated_sheets_are_well_formed_and_answer_keys_are_right(worksheet_type, level, language):
    for seed in range(25):
        sheet = compose_worksheet(worksheet_type, level, language, random.Random(seed))
        assert sheet["worksheet_category"] == worksheet_type
        questions = sheet["questions"]
        assert [q["index"] for q in questions] == list(range(1, QUESTIONS_PER_WORKSHEET + 1))
        for q in questions:
            assert len(q["options"]) == 4 and len(set(q["options"])) == 4, q
            assert q["correct_option"] in "ABCD"
            marked = q["options"]["ABCD".index(q["correct_option"])]
            assert marked.translate(_FROM_DEVANAGARI) == expected_answer(q["question_text"]), q
            if language == "mr":
                assert not re.search(r"[0-9]", q["question_text"] + "".join(q["options"])), q


def test_same_seed_same_sheet_and_different_seed_different_sheet():
    first = compose_worksheet("homework", "E", "mr", random.Random(7))
    assert compose_worksheet("homework", "E", "mr", random.Random(7)) == first
    assert compose_worksheet("homework", "E", "mr", random.Random(8)) != first


def test_homework_level_mixes_the_configured_difficulties():
    difficulty = {s["code"]: int(s["difficulty_level"]) for s in load_skill_catalog()}
    for seed in range(20):
        sheet = compose_worksheet("homework", "C", "en", random.Random(seed))
        counts = {}
        for q in sheet["questions"]:
            counts[difficulty[q["skill_code"]]] = counts.get(difficulty[q["skill_code"]], 0) + 1
        assert counts == {1: 5, 2: 5, 3: 10}


def test_practice_levels_use_only_their_own_and_previous_level_skills():
    for theme, levels in PRACTICE_THEME_SKILLS.items():
        for level, skills in levels.items():
            allowed = set(skills) | set(levels.get(level - 1, []))
            sheet = compose_worksheet("practice", f"{theme}{level}", "en", random.Random(level))
            assert {q["skill_code"] for q in sheet["questions"]} <= allowed


@pytest.mark.parametrize("bad", ["H", "", "Z1"])
def test_invalid_levels_are_refused(bad):
    with pytest.raises(ValueError):
        compose_worksheet("homework" if len(bad) < 2 else "practice", bad, "en", random.Random(0))


def test_practice_level_parsing():
    assert parse_practice_level(" m4 ") == ("M", 4)
    for bad in ["M", "M6", "X1", "Ma"]:
        with pytest.raises(ValueError):
            parse_practice_level(bad)


def test_unsupported_language_is_refused():
    with pytest.raises(ValueError):
        compose_worksheet("homework", "A", "hi", random.Random(0))


def test_marathi_digits_fix_handles_operand_substrings():
    # The old sequential-replace conversion turned this into "३ + 1३".
    assert arabic_to_devanagari("3 + 13") == "३ + १३"


def test_blank_omr_sheet_shape():
    sheet = compose_omr_worksheet(78)
    assert sheet["template_name"] == "basic_omr" and sheet["worksheet_category"] == "omr"
    assert [q["index"] for q in sheet["questions"]] == list(range(1, 79))
    assert all(q["options"] == ["", "", "", ""] and q["correct_option"] == "" for q in sheet["questions"])
