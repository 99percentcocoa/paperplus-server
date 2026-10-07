"""Misconception-based wrong answers ("distractors") per skill -- ported from the old repo's
services/distractor_generator_service.py, plus models.Question.choose_distractors.

Changes from the old code, none of which change what can be generated:
  * Randomness goes through an injected `random.Random` (see questions.py).
  * The old off_by_one_generic/one_table_off/division_errors shuffled their *mutable default*
    `offsets=[-1, 1]` list in place, so its order leaked from one call into the next as hidden
    module state. Each call now shuffles its own fresh copy. (In practice this never changed an
    output -- the resulting small-int sets iterate in the same order either way, and the parity
    test against the unmodified old code passes -- but it no longer relies on that.)
  * print() calls became logger.debug().
"""

import logging
import random
from typing import Callable

logger = logging.getLogger(__name__)

DEFAULT_OFFSETS = (-1, 1)
WIDE_OFFSETS = (-2, -1, 1, 2)


def _coerce_numeric_answer(value):
    """Normalize a correct answer value to a numeric quotient for distractor logic."""
    if isinstance(value, tuple):
        return int(value[0])

    if isinstance(value, str):
        s = value.strip().replace(" ", "")
        if "R" in s:
            q, sep, r = s.partition("R")
            if sep and q and r:
                return int(q)
        try:
            return int(s)
        except ValueError as exc:
            raise ValueError(f"Cannot convert answer to int: {value!r}") from exc

    if isinstance(value, int):
        return value

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert answer to int: {value!r}") from exc


def is_non_negative_option(value) -> bool:
    """Return True only for non-negative answer/option values (ints, digit strings, 'QRr')."""
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return False
        if "R" in s:
            q, sep, r = s.partition("R")
            if not sep:
                return False
            try:
                return int(q) >= 0 and int(r) >= 0
            except ValueError:
                return False
        try:
            return int(s) >= 0
        except ValueError:
            return False
    return False


def get_terms(question: str, correct_ans) -> list[int]:
    """Extract [larger operand, smaller operand, numeric answer] from 'a op b'."""
    terms = question.split()
    num1 = int(terms[0])
    num2 = int(terms[-1])
    if num1 < num2:
        num1, num2 = num2, num1
    return [num1, num2, _coerce_numeric_answer(correct_ans)]


def get_nth_digit_string(number, n: int) -> int | None:
    """The nth digit (1-based, from the left) of an int or numeric string, else None."""
    if isinstance(number, int):
        num_str = str(abs(number))
    elif isinstance(number, str):
        num_str = number.strip().lstrip("+-")
    else:
        raise TypeError("number must be int or str")

    if not (0 < n <= len(num_str)):
        return None

    ch = num_str[n - 1]
    return int(ch) if ch.isdigit() else None


def off_by_one_generic(question, correct_ans, rng: random.Random, offsets=DEFAULT_OFFSETS):
    """Off by one (all operations except multi-digit addition/subtraction)."""
    offsets = list(offsets)
    rng.shuffle(offsets)
    return list({int(correct_ans) + offset for offset in offsets})


def off_by_one_multidigit(question, correct_ans, offsets=DEFAULT_OFFSETS):
    """Off by one in any place value of a multi-digit addition/subtraction answer -- covers
    adding/subtracting at the wrong place value and forgetting a carry."""
    num1, num2, correct_ans = get_terms(question, correct_ans)

    if len(str(num1)) == 1 or len(str(num2)) == 1:
        raise ValueError("Both numbers must be multi-digit for off_by_one_multidigit.")

    distractors = set()
    max_digits = len(str(correct_ans))
    for i in range(1, max_digits + 1):
        digit = get_nth_digit_string(correct_ans, i)
        if digit is not None:
            for offset in offsets:
                new_digit = digit + offset
                if 0 <= new_digit <= 9:
                    distractors.add(correct_ans + (offset * (10 ** (max_digits - i))))
    return list(distractors)


def add_instead_of_multiply(question, correct_ans):
    """Long multiplication, but adding each digit instead of multiplying by it."""
    num1, num2, correct_ans = get_terms(question, correct_ans)
    distractors = set()

    total = 0
    for idx, ch in enumerate(reversed(str(abs(num2)))):
        partial = num1 + int(ch)
        total += partial * (10**idx)

    if total != correct_ans:
        distractors.add(total)
    return list(distractors)


def add_instead_of(question, correct_ans):
    """Add instead of subtract."""
    num1, num2, _ = get_terms(question, correct_ans)
    return [num1 + num2]


def one_table_off(question, correct_ans, rng: random.Random, offsets=DEFAULT_OFFSETS):
    """Read off the neighbouring multiplication table row (T5, T10)."""
    num1, num2, _ = get_terms(question, correct_ans)
    offsets = list(offsets)
    rng.shuffle(offsets)
    distractors = set()
    for offset in offsets:
        distractor = num1 * (num2 + offset)
        if distractor >= 0:
            distractors.add(distractor)
    return list(distractors)


def add_wrong_place_value_addition(question, correct_ans):
    """Add the second number at the wrong place value."""
    num1, num2, _ = get_terms(question, correct_ans)
    digits_difference = len(str(num1)) - len(str(num2))
    return list({num1 + (num2 * 10 ** (i + 1)) for i in range(digits_difference)})


def division_errors(question, correct_ans, rng: random.Random, offsets=DEFAULT_OFFSETS):
    """Off by one in the quotient, and per-digit slips during long division."""
    num1, num2, correct_ans = get_terms(question, correct_ans)
    distractors = set()

    offsets = list(offsets)
    rng.shuffle(offsets)
    for offset in offsets:
        distractor = int(correct_ans) + offset
        if distractor >= 0:
            distractors.add(distractor)

    quotient = int(correct_ans)

    # Off by one in each digit of the quotient
    if quotient >= 10:
        quotient_str = str(quotient)
        for i in range(len(quotient_str)):
            for offset in [-1, 1]:
                new_digit = int(quotient_str[i]) + offset
                if 0 <= new_digit <= 9:
                    distractors.add(int(quotient_str[:i] + str(new_digit) + quotient_str[i + 1 :]))

    # Missing / extra step in long division
    if num2 != 0:
        distractors.add(quotient - 1 if quotient > 0 else 0)
        distractors.add(quotient + 1)

    distractors.discard(int(correct_ans))
    return list(distractors)


DistractorFn = Callable[[str, object, random.Random], list]


def _generic(offsets=DEFAULT_OFFSETS) -> DistractorFn:
    return lambda q, ans, rng: off_by_one_generic(q, ans, rng, offsets=offsets)


def _table(offsets=DEFAULT_OFFSETS) -> DistractorFn:
    return lambda q, ans, rng: one_table_off(q, ans, rng, offsets=offsets)


def _multidigit(offsets=DEFAULT_OFFSETS) -> DistractorFn:
    return lambda q, ans, rng: off_by_one_multidigit(q, ans, offsets=offsets)


_place_value: DistractorFn = lambda q, ans, rng: add_wrong_place_value_addition(q, ans)  # noqa: E731
_add_instead: DistractorFn = lambda q, ans, rng: add_instead_of(q, ans)  # noqa: E731
_add_not_multiply: DistractorFn = lambda q, ans, rng: add_instead_of_multiply(q, ans)  # noqa: E731
_division: DistractorFn = lambda q, ans, rng: division_errors(q, ans, rng)  # noqa: E731

DISTRACTORS: dict[str, list[DistractorFn]] = {
    "1A": [_generic(WIDE_OFFSETS)],
    "1S": [_generic(WIDE_OFFSETS)],
    "T5": [_table(WIDE_OFFSETS)],
    "2A1": [_place_value, _generic()],
    "2A2": [_place_value, _multidigit()],
    "2S1": [_add_instead, _generic()],
    "1AC": [_generic(WIDE_OFFSETS)],
    "2A1C": [_place_value, _generic()],
    "2A2C": [_place_value, _multidigit()],
    "2S1B": [_add_instead, _generic()],
    "2S2": [_add_instead, _multidigit()],
    "T10": [_table(WIDE_OFFSETS)],
    "3A": [_place_value, _multidigit()],
    "3AC": [_place_value, _multidigit()],
    "3S": [_add_instead, _multidigit()],
    "2S2B": [_add_instead, _multidigit()],
    "3AC2": [_place_value, _multidigit()],
    "3SB": [_add_instead, _multidigit()],
    "3SB2": [_add_instead, _multidigit()],
    "2M1": [_add_not_multiply, _generic()],
    "3M1": [_add_not_multiply, _generic(WIDE_OFFSETS)],
    "2M1C": [_add_not_multiply, _generic(WIDE_OFFSETS)],
    "3M1C": [_add_not_multiply, _generic(WIDE_OFFSETS)],
    "3M1C2": [_add_not_multiply, _generic(WIDE_OFFSETS)],
    "2M2": [_add_not_multiply, _multidigit()],
    "2D1": [_division, _generic()],
    "3D1": [_division, _generic()],
    "2M2C": [_add_not_multiply, _multidigit(WIDE_OFFSETS)],
    "2D1R": [_division, _generic()],
    "3D1R": [_division, _generic()],
    "3M2C": [_add_not_multiply, _multidigit(WIDE_OFFSETS)],
    "3D1Z": [_division, _generic()],
    "4D1R": [_division, _generic()],
}


def generate_distractors(skill_code: str, question: str, correct_ans, rng: random.Random) -> list:
    """All non-negative distractors the skill's misconception functions produce."""
    if skill_code not in DISTRACTORS:
        raise ValueError(f"Unknown skill code: {skill_code}")

    all_distractors = set()
    for func in DISTRACTORS[skill_code]:
        try:
            all_distractors.update(func(question, correct_ans, rng))
        except Exception as exc:  # one misconception not applying (e.g. 1-digit operand) is normal
            logger.debug("Distractor function skipped for %s %r: %s", skill_code, question, exc)

    return [d for d in all_distractors if is_non_negative_option(d)]


def _build_non_negative_fallback_distractors(correct_ans, existing, needed=3):
    """Fill distractors up to `needed` with guaranteed non-negative candidates."""
    out = list(existing)

    if isinstance(correct_ans, str) and "R" in correct_ans:
        q, _, r = correct_ans.strip().replace(" ", "").partition("R")
        try:
            qv = int(q)
            rv = int(r)
        except ValueError:
            qv, rv = 0, 0
        seed = [f"{qv + 1}R{rv}", f"{qv}R{rv + 1}", f"{qv + 1}R{rv + 1}", f"{qv + 2}R{rv}", f"{qv}R{rv + 2}"]
    else:
        try:
            base = _coerce_numeric_answer(correct_ans)
        except ValueError:
            base = 0
        seed = [base + i for i in (1, 2, 3, 4, 5, 6)]

    for candidate in seed:
        if len(out) >= needed:
            break
        if candidate == correct_ans or not is_non_negative_option(candidate) or candidate in out:
            continue
        out.append(candidate)
    return out


def build_distractors(skill_code: str, question: str, correct_ans, rng: random.Random, needed: int = 3) -> list:
    """Return at least `needed` non-negative distractors for a question."""
    try:
        possible_distractors = generate_distractors(skill_code, question, correct_ans, rng)
    except Exception as exc:
        logger.debug("Error generating distractors for %s: %s", skill_code, exc)
        possible_distractors = []

    possible_distractors = [d for d in possible_distractors if is_non_negative_option(d)]

    if len(possible_distractors) < needed:
        try:
            correct_val = _coerce_numeric_answer(correct_ans)
            candidates = [correct_val + offset for offset in [1, 2, 3, 4, 5, 6, 7, 8]]
            candidates += [correct_val - offset for offset in [1, 2, 3] if correct_val - offset >= 0]
            possible_distractors.extend(candidates)
            possible_distractors = list(set(possible_distractors))
            possible_distractors = [d for d in possible_distractors if d != correct_val][:needed]
        except (ValueError, AttributeError):
            possible_distractors = []

    return _build_non_negative_fallback_distractors(
        correct_ans,
        [d for d in possible_distractors if d != correct_ans],
        needed=needed,
    )


def choose_options(correct_answer, possible_distractors: list, rng: random.Random) -> tuple[list, int]:
    """Pick 3 distractors and shuffle them with the correct answer.

    Returns (options, answer_position) with answer_position 1-based. Port of the old
    models.Question.choose_distractors.
    """
    filtered = [d for d in possible_distractors if is_non_negative_option(d) and d != correct_answer]
    if len(filtered) < 3:
        raise ValueError("Need at least 3 non-negative possible distractors")

    options = [correct_answer] + rng.sample(filtered, 3)
    rng.shuffle(options)
    return options, options.index(correct_answer) + 1
