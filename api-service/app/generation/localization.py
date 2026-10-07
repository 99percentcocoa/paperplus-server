"""Marathi rendering of generated questions: Arabic digits -> Devanagari digits.

The old services/question_generator_service.question_to_marathi replaced the first and last
operand with sequential substring str.replace calls, which leaves ASCII digits behind when the
first operand is a substring of the second (e.g. "3 + 13" -> "३ + 1३"). No current generator
produces such a question, so translating every digit in the string gives identical output for
everything that can be generated today, without the latent bug.
"""

_DEVANAGARI_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")

SUPPORTED_LANGUAGES = ("en", "mr")


def arabic_to_devanagari(text: str) -> str:
    return text.translate(_DEVANAGARI_DIGITS)


def localize_value(value, language: str):
    """Option/answer value for `language`: unchanged for en, a Devanagari string for mr."""
    if language == "mr":
        return arabic_to_devanagari(str(value))
    return value


def localize_text(text: str, language: str) -> str:
    return arabic_to_devanagari(text) if language == "mr" else text
