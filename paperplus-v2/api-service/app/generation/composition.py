"""Worksheet composition: which skills a worksheet covers, and its 20 questions -- ported from the
old repo's worksheet_json_generator.py.

Pure functions over an injected `random.Random`; no DB access. The skill catalog is read from
app/data/skills.json (the same file scripts/seed_skills.py seeds into the `skills` table) rather
than the DB, because catalog *order* determines the sequence of random draws, and the old
generator iterated skills in this file's order -- DB row order isn't guaranteed.
"""

import json
import random
from functools import lru_cache
from pathlib import Path

from app.generation.distractors import build_distractors, choose_options
from app.generation.localization import SUPPORTED_LANGUAGES, localize_text, localize_value
from app.generation.questions import gen_questions, number_to_letter

SKILLS_PATH = Path(__file__).resolve().parent.parent / "data" / "skills.json"

QUESTIONS_PER_WORKSHEET = 20

# Homework level -> {difficulty level: share of the 20 questions}. Copied from the old
# config.SETTINGS.WORKSHEET_LEVEL_DISTRIBUTIONS.
WORKSHEET_LEVEL_DISTRIBUTIONS: dict[str, dict[int, float]] = {
    "A": {1: 1.0},
    "B": {1: 0.5, 2: 0.5},
    "C": {1: 0.25, 2: 0.25, 3: 0.5},
    "D": {1: 0.125, 2: 0.125, 3: 0.25, 4: 0.5},
    "E": {1: 1 / 12, 2: 1 / 12, 3: 1 / 12, 4: 0.25, 5: 0.5},
    "F": {1: 1 / 16, 2: 1 / 16, 3: 1 / 16, 4: 1 / 16, 5: 0.25, 6: 0.5},
    "G": {1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.05, 6: 0.25, 7: 0.5},
}
HOMEWORK_LEVELS = tuple(WORKSHEET_LEVEL_DISTRIBUTIONS)

# Practice theme -> level (1-5) -> skills practised at that level.
PRACTICE_THEME_SKILLS: dict[str, dict[int, list[str]]] = {
    "A": {
        1: ["1A", "2A1", "2A2"],
        2: ["1AC", "2A1C", "2A2", "3A"],
        3: ["2A1C", "2A2C", "3A", "3AC"],
        4: ["2A2C", "3A", "3AC", "3AC2"],
        5: ["3A", "3AC", "3AC2", "2A2C"],
    },
    "S": {
        1: ["1S", "2S1"],
        2: ["1S", "2S1", "2S2"],
        3: ["2S1B", "2S2", "3S", "2S2B"],
        4: ["2S1B", "2S2B", "3S", "3SB"],
        5: ["3SB", "3SB2", "2S2B", "3S"],
    },
    "M": {
        1: ["T5", "2M1"],
        2: ["T10", "2M1", "3M1"],
        3: ["2M1", "2M1C", "3M1", "3M1C"],
        4: ["2M1C", "3M1C", "3M1C2", "2M2"],
        5: ["3M1C2", "2M2", "2M2C", "3M2C"],
    },
    "D": {
        1: ["2D1"],
        2: ["2D1", "3D1"],
        3: ["2D1", "2D1R", "3D1"],
        4: ["2D1R", "3D1R", "3D1Z", "4D1R"],
        5: ["2D1R", "3D1R", "3D1Z", "4D1R"],
    },
}
PRACTICE_RECYCLED_SHARE = 0.30  # share of a practice sheet drawn from the previous level's skills


@lru_cache(maxsize=1)
def load_skill_catalog() -> tuple[dict, ...]:
    return tuple(json.loads(SKILLS_PATH.read_text(encoding="utf-8")))


def _skill_codes_at_difficulty(difficulty_level: int) -> list[str]:
    codes = [s["code"] for s in load_skill_catalog() if s["difficulty_level"] == str(difficulty_level)]
    if not codes:
        raise ValueError(f"No skills found at difficulty level {difficulty_level}")
    return codes


def parse_practice_level(level_label: str) -> tuple[str, int]:
    """Parse a practice label like 'A1' or 'M4' into (theme, level)."""
    if not isinstance(level_label, str):
        raise ValueError(f"Practice level must be a string, got {type(level_label).__name__}")

    label = level_label.strip().upper()
    if len(label) < 2:
        raise ValueError(f"Invalid practice level: {level_label!r}. Expected a theme plus a level, e.g. A1 or D3.")

    theme = label[0]
    if theme not in PRACTICE_THEME_SKILLS:
        raise ValueError(f"Invalid practice theme: {theme!r}. Must be one of: {', '.join(PRACTICE_THEME_SKILLS)}")

    try:
        level = int(label[1:])
    except ValueError as exc:
        raise ValueError(f"Invalid practice level: {level_label!r}. Expected a number after the theme.") from exc

    if level not in range(1, 6):
        raise ValueError(f"Practice level must be between 1 and 5, got {level}.")
    return theme, level


def parse_homework_level(level_label: str) -> str:
    level = str(level_label).strip().upper()
    if level not in WORKSHEET_LEVEL_DISTRIBUTIONS:
        raise ValueError(f"Homework level must be one of {''.join(HOMEWORK_LEVELS)}; got {level_label!r}")
    return level


def _assign_questions_to_skills(skill_codes: list[str], total_questions: int, rng: random.Random) -> dict[str, int]:
    """Distribute a fixed number of questions across a skill list without leaving any unused."""
    if not skill_codes:
        return {}
    if len(skill_codes) == 1:
        return {skill_codes[0]: total_questions}

    distribution = {}
    remaining = total_questions
    for i, skill_code in enumerate(skill_codes[:-1]):
        max_for_skill = remaining - (len(skill_codes) - i - 1)
        count = rng.randint(1, max(1, max_for_skill))
        distribution[skill_code] = count
        remaining -= count
    distribution[skill_codes[-1]] = remaining
    return distribution


def practice_distribution(theme: str, level: int, rng: random.Random) -> dict[str, int]:
    """20-question skill distribution for a practice sheet; levels 2-5 recycle 30% from the level below."""
    current_skills = PRACTICE_THEME_SKILLS[theme][level]
    if level == 1:
        return _assign_questions_to_skills(current_skills, QUESTIONS_PER_WORKSHEET, rng)

    previous_skills = PRACTICE_THEME_SKILLS[theme][level - 1]
    recycled_questions = max(1, round(QUESTIONS_PER_WORKSHEET * PRACTICE_RECYCLED_SHARE))
    current_questions = QUESTIONS_PER_WORKSHEET - recycled_questions

    distribution = dict(_assign_questions_to_skills(previous_skills, recycled_questions, rng))
    for skill_code, count in _assign_questions_to_skills(current_skills, current_questions, rng).items():
        distribution[skill_code] = distribution.get(skill_code, 0) + count

    total = sum(distribution.values())
    if total != QUESTIONS_PER_WORKSHEET:
        last_skill = list(distribution)[-1]
        distribution[last_skill] += QUESTIONS_PER_WORKSHEET - total
    return distribution


def homework_distribution(level: str, rng: random.Random) -> dict[str, int]:
    """20-question skill distribution for a homework level (A-G).

    Ported as-is, including the old `- 2` in max_for_skill (its sibling functions use `- 1`):
    it lets earlier skills take more questions, so later skills at a difficulty level can get
    none. That shapes what's printed, so changing it is a content decision, not a port fix.
    """
    difficulty_distribution = WORKSHEET_LEVEL_DISTRIBUTIONS[level]

    # Questions per difficulty level; the last level takes whatever rounding left over.
    difficulty_question_counts = {}
    total_allocated = 0
    sorted_difficulties = sorted(difficulty_distribution.items())
    for i, (difficulty_level, proportion) in enumerate(sorted_difficulties):
        if i == len(sorted_difficulties) - 1:
            num_questions = QUESTIONS_PER_WORKSHEET - total_allocated
        else:
            num_questions = round(QUESTIONS_PER_WORKSHEET * proportion)
            total_allocated += num_questions
        difficulty_question_counts[difficulty_level] = num_questions

    skill_distribution: dict[str, int] = {}
    for difficulty_level, num_questions in difficulty_question_counts.items():
        if num_questions == 0:
            continue
        skill_codes = _skill_codes_at_difficulty(difficulty_level)

        remaining = num_questions
        for i, skill_code in enumerate(skill_codes[:-1]):
            if remaining <= 0:
                break
            max_for_skill = remaining - (len(skill_codes) - i - 2)
            questions_for_skill = rng.randint(1, max(1, max_for_skill))
            if questions_for_skill > 0:
                skill_distribution[skill_code] = questions_for_skill
                remaining -= questions_for_skill

        if remaining > 0:
            last_skill = skill_codes[-1]
            skill_distribution[last_skill] = skill_distribution.get(last_skill, 0) + remaining

    return skill_distribution


def build_questions(skill_distribution: dict[str, int], language: str, rng: random.Random) -> list[dict]:
    """Generate the questions for a skill distribution, in the worksheet-JSON question shape."""
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"language must be one of {SUPPORTED_LANGUAGES}, got {language!r}")
    total = sum(skill_distribution.values())
    if total != QUESTIONS_PER_WORKSHEET:
        raise ValueError(f"Skill distribution must sum to {QUESTIONS_PER_WORKSHEET}, got {total}")

    questions = []
    for skill_code, num_questions in skill_distribution.items():
        for question_text, correct_ans in gen_questions(skill_code, num_questions, rng):
            if isinstance(correct_ans, tuple):
                quotient, remainder = correct_ans
                correct_ans = f"{quotient}R{remainder}"

            possible_distractors = build_distractors(skill_code, question_text, correct_ans, rng, needed=3)
            options, answer_position = choose_options(
                localize_value(correct_ans, language),
                [localize_value(d, language) for d in possible_distractors],
                rng,
            )
            questions.append(
                {
                    "index": len(questions) + 1,
                    "question_text": localize_text(question_text, language),
                    "skill_code": skill_code,
                    "options": [str(opt) for opt in options],
                    "correct_option": number_to_letter(answer_position),
                }
            )
    return questions


def compose_worksheet(worksheet_type: str, level: str, language: str, rng: random.Random, title: str | None = None) -> dict:
    """Worksheet JSON (same shape the old generator wrote to files/json/) for a homework or practice sheet."""
    if worksheet_type == "homework":
        level_label = parse_homework_level(level)
        distribution = homework_distribution(level_label, rng)
        title = title or f"Worksheet Level {level_label}"
    elif worksheet_type == "practice":
        theme, level_num = parse_practice_level(level)
        level_label = f"{theme}{level_num}"
        distribution = practice_distribution(theme, level_num, rng)
        title = title or f"Practice Worksheet {level_label}"
    else:
        raise ValueError(f"worksheet_type must be 'homework' or 'practice', got {worksheet_type!r}")

    return {
        "title": title,
        "level": level_label,
        "language": language,
        "worksheet_category": worksheet_type,
        "questions": build_questions(distribution, language, rng),
    }


def compose_omr_worksheet(question_count: int, language: str = "en", title: str | None = None) -> dict:
    """Blank basic_omr worksheet JSON: numbered questions with no text or options (answer keys
    come from omr_answer_sets / question_paper_variants, keyed by the printed question-paper code)."""
    return {
        "title": title or "Basic OMR Sheet",
        "worksheet_category": "omr",
        "template_name": "basic_omr",
        "language": language,
        "questions": [
            {"index": idx, "question_text": "", "options": ["", "", "", ""], "correct_option": ""}
            for idx in range(1, question_count + 1)
        ],
    }
