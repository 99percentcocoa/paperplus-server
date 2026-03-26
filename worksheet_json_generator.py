"""
Create a 20-question worksheet with questions and distractors.
Saves the worksheet to a JSON file.
"""

import json
import random
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


if __name__ == "__main__":
    print("Creating 20-question worksheets for levels A-G...")
    
    # Create worksheets for each level (A-G)
    for level in "ABCDEFG":
        
        worksheet_json = create_worksheet_json(title=f"Worksheet Level {level}", level=level, language="mr")
        
        # Save to file
        wsname = f"mr_level_{level}"
        filename = f"{wsname}.json"
        filepath = Path(__file__).parent / "files" / "json" / filename

        save_worksheet(worksheet_json, filepath)
        
        # Print a preview
        print(f"Preview of first 2 questions:")
        questions = worksheet_json.get("questions", [])
        for q in questions[:2]:
            print(f"\n{q['question_text']}")
            for i, opt in enumerate(q['options']):
                print(f"   {chr(65 + i)}) {opt}")
