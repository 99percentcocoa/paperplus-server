"""
Create a 20-question worksheet with questions and distractors.
Saves the worksheet to a JSON file.
"""

import json
import random
import argparse
from pathlib import Path
from services.question_generator_service import gen_questions, number_to_letter, question_to_marathi, _load_skills
from services.distractor_generator_service import build_distractors
from models import Question
from config import SETTINGS

# Worksheet levels map to difficulty level distributions
# Keys are difficulty levels, values are proportions of 20 questions
# WORKSHEET_LEVEL_DISTRIBUTIONS = {
#     "A": {1: 1.0},
#     "B": {1: 0.5, 2: 0.5},
#     "C": {1: 0.25, 2: 0.25, 3: 0.5},
#     "D": {1: 0.125, 2: 0.125, 3: 0.25, 4: 0.5},  # 1-2 (25%) split evenly
#     "E": {1: 1/12, 2: 1/12, 3: 1/12, 4: 0.25, 5: 0.5},  # 1-3 (25%) split evenly
#     "F": {1: 1/16, 2: 1/16, 3: 1/16, 4: 1/16, 5: 0.25, 6: 0.5},  # 1-4 (25%) split evenly
#     "G": {1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.05, 6: 0.25, 7: 0.5},  # 1-5 (25%) split evenly
# }
WORKSHEET_LEVEL_DISTRIBUTIONS = SETTINGS.WORKSHEET_LEVEL_DISTRIBUTIONS

PRACTICE_THEME_SKILLS = {
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


def parse_practice_level(level_label: str) -> tuple[str, int]:
    """Parse a practice label like 'A1' or 'M4' into (theme, level)."""
    if not isinstance(level_label, str):
        raise ValueError(f"Practice level must be a string, got {type(level_label).__name__}")

    label = level_label.strip().upper()
    if len(label) < 2:
        raise ValueError(f"Invalid practice level: {level_label!r}. Expected a theme plus a level, e.g. A1 or D3.")

    theme = label[0]
    if theme not in PRACTICE_THEME_SKILLS:
        valid = ", ".join(PRACTICE_THEME_SKILLS.keys())
        raise ValueError(f"Invalid practice theme: {theme!r}. Must be one of: {valid}")

    try:
        level = int(label[1:])
    except ValueError as exc:
        raise ValueError(f"Invalid practice level: {level_label!r}. Expected a number after the theme.") from exc

    if level not in range(1, 6):
        raise ValueError(f"Practice level must be between 1 and 5, got {level}.")

    return theme, level


def _assign_questions_to_skills(skill_codes: list[str], total_questions: int) -> dict:
    """Distribute a fixed number of questions across a skill list without leaving any unused."""
    if not skill_codes:
        return {}

    distribution = {}
    if len(skill_codes) == 1:
        distribution[skill_codes[0]] = total_questions
        return distribution

    remaining = total_questions
    for i, skill_code in enumerate(skill_codes[:-1]):
        max_for_skill = remaining - (len(skill_codes) - i - 1)
        count = random.randint(1, max(1, max_for_skill))
        distribution[skill_code] = count
        remaining -= count

    distribution[skill_codes[-1]] = remaining
    return distribution


def create_practice_worksheet_level_distribution(theme: str, level: int | str) -> dict:
    """Create a 20-question distribution for a practice sheet using theme + level."""
    if isinstance(level, str):
        theme, level = parse_practice_level(f"{theme}{level}") if not theme else parse_practice_level(level)
    theme = theme.upper()
    if theme not in PRACTICE_THEME_SKILLS:
        valid = ", ".join(PRACTICE_THEME_SKILLS.keys())
        raise ValueError(f"Invalid practice theme: {theme!r}. Must be one of: {valid}")

    level = int(level)
    if level not in range(1, 6):
        raise ValueError(f"Practice level must be between 1 and 5, got {level}.")

    current_skills = PRACTICE_THEME_SKILLS[theme][level]
    distribution = {}

    if level == 1:
        distribution.update(_assign_questions_to_skills(current_skills, 20))
        return distribution

    previous_skills = PRACTICE_THEME_SKILLS[theme][level - 1]
    recycled_questions = max(1, round(20 * 0.30))
    current_questions = 20 - recycled_questions

    distribution.update(_assign_questions_to_skills(previous_skills, recycled_questions))
    current_distribution = _assign_questions_to_skills(current_skills, current_questions)
    for skill_code, count in current_distribution.items():
        distribution[skill_code] = distribution.get(skill_code, 0) + count

    # Ensure the final total is exactly 20.
    total = sum(distribution.values())
    if total != 20:
        diff = 20 - total
        last_skill = list(distribution.keys())[-1]
        distribution[last_skill] += diff

    return distribution


def _normalize_skills(skills_data) -> list:
    """Return a list of skill records from either dict or list input."""
    if isinstance(skills_data, dict):
        return list(skills_data.values())
    if isinstance(skills_data, list):
        return skills_data
    return []


def create_worksheet(skill_distribution: dict = None, language: str = "en") -> list:
    """
    Create a 20-question worksheet with questions and distractors.
    
    Args:
        skill_distribution: Dict mapping skill_code to number of questions.
                           If None, uses a default distribution.
    
    Returns:
        List of Question objects with chosen distractors.
    """
    
    # Default distribution if not provided
    if skill_distribution is None:
        skill_distribution = {
            "1A": 2,
            "2A1": 3,
            "2A2": 2,
            "2S1": 2,
            "T5": 2,
            "T10": 2,
            "3A": 2,
            "3S": 2,
            "2M1": 1,
            "2D1": 0,
        }
    
    # Verify total is 20
    total = sum(skill_distribution.values())
    if total != 20:
        raise ValueError(f"Skill distribution must sum to 20, got {total}")
    
    worksheet = []
    question_index = 1
    
    for skill_code, num_questions in skill_distribution.items():
        # Generate raw questions
        raw_questions = gen_questions(skill_code, num_questions)
        
        for question_text, correct_ans in raw_questions:
            # Handle tuple answers (quotient, remainder) for division problems
            if isinstance(correct_ans, tuple):
                quotient, remainder = correct_ans
                correct_ans = f"{quotient}R{remainder}"
            
            possible_distractors = build_distractors(
                skill_code=skill_code,
                question=question_text,
                correct_ans=correct_ans,
                needed=3,
            )
            
            # Create Question object
            question = Question(
                index=question_index,
                question_text=question_text,
                skill_code=skill_code,
                options=[correct_ans],  # Will be replaced by choose_distractors
                answer=1,  # Will be updated by choose_distractors
                possible_distractors=possible_distractors
            )

            if language == "mr":
                question = question_to_marathi(question)
            
            # Choose distractors and randomize positions
            question.choose_distractors()
            
            # Convert answer from 1-4 to A-D
            question.correct_option = number_to_letter(question.answer)
            
            worksheet.append(question)
            question_index += 1
    
    return worksheet


def create_difficulty_distribution(difficulty_level: int) -> dict:
    """
    Create a random skill distribution for a given difficulty level.
    
    Args:
        difficulty_level: Difficulty level (1-7) as specified in skills.json
    
    Returns:
        Dict mapping skill_code to number of questions, summing to 20.
    """
    # Load skills 
    skills = _normalize_skills(_load_skills())
    
    # Filter skills by difficulty level (stored as string in skills.json)
    skills_at_level = [s for s in skills if s["difficulty_level"] == str(difficulty_level)]
    
    if not skills_at_level:
        raise ValueError(f"No skills found at difficulty level {difficulty_level}")
    
    # Get skill codes
    skill_codes = [s["code"] for s in skills_at_level]
    
    # Create random distribution summing to 20
    distribution = {}
    remaining = 20
    
    # Randomly assign questions to each skill code except the last
    for i, skill_code in enumerate(skill_codes[:-1]):
        # Ensure at least 1 question per remaining skill
        max_questions = remaining - (len(skill_codes) - i - 1)
        num_questions = random.randint(1, max_questions)
        distribution[skill_code] = num_questions
        remaining -= num_questions
    
    # Assign remaining questions to the last skill
    if skill_codes:
        distribution[skill_codes[-1]] = remaining
    
    return distribution


def create_worksheet_level_distribution(worksheet_level: str) -> dict:
    """
    Create a skill distribution for a worksheet level (A-G).
    
    Each level mixes skills from different difficulty levels:
    - A: level 1 (100%)
    - B: level 1 (50%), level 2 (50%)
    - C: level 1 (25%), level 2 (25%), level 3 (50%)
    - D: level 1-2 (25% total), level 3 (25%), level 4 (50%)
    - E: level 1-3 (25% total), level 4 (25%), level 5 (50%)
    - F: level 1-4 (25% total), level 5 (25%), level 6 (50%)
    - G: level 1-5 (25% total), level 6 (25%), level 7 (50%)
    
    Args:
        worksheet_level: Worksheet level letter (A-G)
    
    Returns:
        Dict mapping skill_code to number of questions, summing to 20.
    """
    # Load skills
    skills = _normalize_skills(_load_skills())
    
    if worksheet_level not in WORKSHEET_LEVEL_DISTRIBUTIONS:
        valid_levels = ", ".join(WORKSHEET_LEVEL_DISTRIBUTIONS.keys())
        raise ValueError(f"Invalid worksheet level: {worksheet_level}. Must be one of: {valid_levels}")
    
    difficulty_distribution = WORKSHEET_LEVEL_DISTRIBUTIONS[worksheet_level]
    skill_distribution = {}
    
    # First pass: calculate questions per difficulty level
    # Use rounding but ensure the last level gets remaining questions to sum to 20
    difficulty_question_counts = {}
    total_allocated = 0
    sorted_difficulties = sorted(difficulty_distribution.items())
    
    for i, (difficulty_level, proportion) in enumerate(sorted_difficulties):
        if i == len(sorted_difficulties) - 1:
            # Last difficulty level gets remaining questions
            num_questions = 20 - total_allocated
        else:
            num_questions = round(20 * proportion)
            total_allocated += num_questions
        
        difficulty_question_counts[difficulty_level] = num_questions
    
    # Second pass: for each difficulty level, randomly distribute questions among skills
    for difficulty_level, num_questions in difficulty_question_counts.items():
        if num_questions == 0:
            continue
        
        # Get skills at this difficulty level
        skills_at_level = [s for s in skills if s["difficulty_level"] == str(difficulty_level)]
        
        if not skills_at_level:
            raise ValueError(f"No skills found at difficulty level {difficulty_level}")
        
        skill_codes = [s["code"] for s in skills_at_level]
        
        # Randomly distribute questions among skills at this level
        remaining = num_questions
        
        for i, skill_code in enumerate(skill_codes[:-1]):
            if remaining <= 0:
                break
            
            # Ensure at least 1 question per remaining skill
            max_for_skill = remaining - (len(skill_codes) - i - 2)
            questions_for_skill = random.randint(1, max(1, max_for_skill))
            
            if questions_for_skill > 0:
                skill_distribution[skill_code] = questions_for_skill
                remaining -= questions_for_skill
        
        # Remaining goes to last skill
        if skill_codes and remaining > 0:
            last_skill = skill_codes[-1]
            if last_skill in skill_distribution:
                skill_distribution[last_skill] += remaining
            else:
                skill_distribution[last_skill] = remaining
    
    return skill_distribution


def worksheet_to_json(name: str, worksheet: list, level: str, language: str) -> dict:
    """
    Convert worksheet to JSON-serializable object.
    
    Args:
        worksheet: List of Question objects
    
    Returns:
        Worksheet object with title, level, language, and questions.
    """
    questions = []
    answer_key = []
    
    for q in worksheet:
        # answer is already a letter (A-D) from create_worksheet
        answer_letter = q.correct_option
        
        questions.append({
            "index": q.index,
            "question_text": q.question_text,
            "skill_code": q.skill_code,
            "options": [str(opt) for opt in q.options],
            "correct_option": answer_letter
        })
        answer_key.append(answer_letter)
    
    return {
        "title": name,
        "level": level,
        "language": language,
        "questions": questions
    }


def save_worksheet(worksheet_data: dict, filepath: str = "worksheet.json"):
    """
    Save worksheet object to JSON file.
    
    Args:
        worksheet_data: Worksheet object
        filepath: Output file path
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(worksheet_data, f, indent=2, ensure_ascii=False)
    print(f"Worksheet saved to {filepath}")

def create_worksheet_json(title: str, level: str, language: str) -> dict:
    """
    Create a worksheet JSON structure from level and language.
    
    Args:
        title: Title of the worksheet
        level: Worksheet level (A-G)
        language: Language code (e.g., "en", "mr")
    
    Returns:
        Worksheet object as per worksheet JSON schema.
    """
    distribution = create_worksheet_level_distribution(level)
    worksheet = create_worksheet(skill_distribution=distribution, language=language)
    worksheet_json = worksheet_to_json(name=title, worksheet=worksheet, level=level, language=language)
    return worksheet_json


def create_practice_worksheet_json(title: str, theme: str, level: int | str, language: str) -> dict:
    """Create a practice worksheet JSON using the explicit theme + level system (A1..D5)."""
    theme = str(theme).upper()
    if isinstance(level, str):
        parsed_theme, parsed_level = parse_practice_level(level)
        if theme not in {parsed_theme, ""}:
            theme = parsed_theme
        level = parsed_level

    level_label = f"{theme}{level}"
    distribution = create_practice_worksheet_level_distribution(theme, level)
    worksheet = create_worksheet(skill_distribution=distribution, language=language)
    worksheet_json = worksheet_to_json(name=title, worksheet=worksheet, level=level_label, language=language)
    return worksheet_json


if __name__ == "__main__":
    # Usage:
    # python3 worksheet_json_generator.py --level A --language en
    # python3 worksheet_json_generator.py --level A --language en --filename custom_name
    # python3 worksheet_json_generator.py --all-levels --language mr --output-dir files/json
    parser = argparse.ArgumentParser(description="Generate worksheet JSON files.")
    parser.add_argument(
        "--level",
        default="A",
        help="Worksheet level (A-G). Ignored when --all-levels is set.",
    )
    parser.add_argument(
        "--theme",
        choices=["A", "S", "M", "D"],
        help="Practice theme for a practice worksheet (A, S, M, D).",
    )
    parser.add_argument(
        "--practice-level",
        type=str,
        help="Practice worksheet level like A1, S2, M4, or D3.",
    )
    parser.add_argument(
        "--all-levels",
        action="store_true",
        help="Generate worksheets for all levels A-G.",
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=["en", "mr"],
        help="Worksheet language.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "files" / "json"),
        help="Directory to save generated worksheet JSON files.",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Optional output filename for single-level generation (without .json).",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_levels and args.filename:
        raise ValueError("--filename cannot be used with --all-levels")

    if args.practice_level:
        theme, level = parse_practice_level(args.practice_level)
        worksheet_json = create_practice_worksheet_json(
            title=f"Practice Worksheet {theme}{level}",
            theme=theme,
            level=level,
            language=args.language,
        )
        filename = args.filename if args.filename else f"{args.language}_practice_{theme}{level}.json"
        if not filename.lower().endswith(".json"):
            filename = f"{filename}.json"
        save_worksheet(worksheet_json, output_dir / filename)
        print(f"Created practice worksheet {theme}{level} with {len(worksheet_json['questions'])} questions.")
        raise SystemExit(0)

    levels = "ABCDEFG" if args.all_levels else args.level.upper()
    if not args.all_levels and levels not in "ABCDEFG":
        raise ValueError("--level must be one of A, B, C, D, E, F, G")

    print(f"Creating 20-question worksheet JSON for level(s): {', '.join(levels)}")

    for level in levels:
        worksheet_json = create_worksheet_json(
            title=f"Worksheet Level {level}",
            level=level,
            language=args.language,
        )

        if args.filename:
            filename = args.filename if args.filename.lower().endswith(".json") else f"{args.filename}.json"
        else:
            filename = f"{args.language}_level_{level}.json"
        filepath = output_dir / filename
        save_worksheet(worksheet_json, filepath)

        print("Preview of first 2 questions:")
        questions = worksheet_json.get("questions", [])
        for q in questions[:2]:
            print(f"\n{q['question_text']}")
            for i, opt in enumerate(q['options']):
                print(f"   {chr(65 + i)}) {opt}")
