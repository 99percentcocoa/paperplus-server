"""
Grading Utilities - Helper functions for result checking and scoring.

This module contains utility functions for comparing student answers
against answer keys and calculating scores.
"""

import logging

logger = logging.getLogger(__name__)


def check_results(results, ans_key):
    """Compare student answers with correct answer key and calculate score.

    Args:
        results (list): List of student answers
        ans_key (list): List of correct answers

    Returns:
        int or str: Number of correct answers, or error message if mismatch
    """
    logger.info("Checking results.")
    if len(results) != len(ans_key):
        return "An error occurred. Please try again. ⟳"
    else:
        marked_lowercase = [item.lower() for item in results]
        anskey_lowercase = [item.lower() for item in ans_key]

        marks = 0

        for marked_char, anskey_char in zip(marked_lowercase, anskey_lowercase):
            if marked_char == anskey_char:
                marks += 1

        return marks
