# generate distractiors
# distractor function: takes a number (correct ans) and generates required number of distractors for that number

import random


def _is_non_negative_option(value):
    """Return True only for non-negative distractor candidates."""
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

# extract terms from the question string.
# also make sure that num1 > num2
def get_terms(question, correct_ans):
    terms = question.split()
    num1 = int(terms[0])
    num2 = int(terms[-1])
    if num1 < num2:
        num1, num2 = num2, num1
    return [num1, num2, int(correct_ans)]

# off by one distractor function
# operations used: all except addition and subtraction
def off_by_one_generic(question, correct_ans, offsets=[-1, 1]):
    distractors = set()
    random.shuffle(offsets)
    for index, offset in enumerate(offsets):
        distractor = int(correct_ans) + offset
        distractors.add(distractor)
    return list(distractors)

# off by one for multi digit addition and subtraction
# covers adding/subtracting at wrong place value
# also covers off by one in units place for addition and subtraction
# also covern forgot to add carry
def off_by_one_multidigit(question, correct_ans, offsets=[-1, 1]):
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
                # check if the new digit is valid (0-9)
                if 0 <= new_digit <= 9:
                    distractor = correct_ans + (offset * (10 ** (max_digits - i)))
                    distractors.add(distractor)

    return list(distractors)

# multiplication: performs the multiplication algorithm but adds intstead of multiplying
def add_instead_of_multiply(question, correct_ans):
    num1, num2, correct_ans = get_terms(question, correct_ans)
    distractors = set()

    # Simulate long multiplication, but add the digit instead of multiplying it
    total = 0
    for idx, ch in enumerate(reversed(str(abs(num2)))):
        d = int(ch)
        partial = num1 + d           # mistaken operation: add digit instead of multiply
        total += partial * (10 ** idx)  # shift by place value

    if total != correct_ans:
        distractors.add(total)

    return list(distractors)

# added wrong place value for second number
def add_wrong_place_value(question, correct_ans):
    num1, num2, correct_ans = get_terms(question, correct_ans)
    print(num1, num2)
    distractors = set()

    # num2 added at wrong place value
    digits_difference = len(str(num1)) - len(str(num2))
    # print(f"{len(str(num2))} - {len(str(num1))} = {digits_difference}")
    for i in range(digits_difference):
        distractors.add(num1 + (num2 * 10**(i+1)))

    return list(distractors)

# add instead of subtract (for 1S, 2S1, 2S1B, 2S2, 3S, 2S2B, 3SB, 3SB2)
def add_instead_of(question, correct_ans):
    num1, num2, correct_ans = get_terms(question, correct_ans)
    distractors = set()
    distractor = num1 + num2
    distractors.add(distractor)
    return list(distractors)

# one table off (for T5, T10)
def one_table_off(question, correct_ans, offsets=[-1, 1]):
    num1, num2, correct_ans = get_terms(question, correct_ans)
    distractors = set()
    random.shuffle(offsets)
    for index, offset in enumerate(offsets):
        distractor = num1 * (num2 + offset)
        if distractor >= 0:
            distractors.add(distractor)
    return list(distractors)

# add second number at wrong place value (for 2A1, 2A2, 2A1C, 2A2C, 3A, 3AC2)
def add_wrong_place_value_addition(question, correct_ans):
    num1, num2, correct_ans = get_terms(question, correct_ans)
    distractors = set()

    # num2 added at wrong place value
    digits_difference = len(str(num1)) - len(str(num2))
    for i in range(digits_difference):
        distractors.add(num1 + (num2 * 10**(i+1)))

    return list(distractors)

# division: off by one and arithmetic errors during long division
def division_errors(question, correct_ans, offsets=[-1, 1]):
    num1, num2, correct_ans = get_terms(question, correct_ans)
    distractors = set()
    
    # Off-by-one errors in the quotient
    random.shuffle(offsets)
    for offset in offsets:
        distractor = int(correct_ans) + offset
        if distractor >= 0:  # quotient should be non-negative
            distractors.add(distractor)
    
    # Arithmetic errors during long division steps
    # Simulate common mistakes when subtracting during division
    quotient = int(correct_ans)
    
    # Error 1: Off by one in each digit of quotient
    if quotient >= 10:
        quotient_str = str(quotient)
        for i in range(len(quotient_str)):
            for offset in [-1, 1]:
                digit = int(quotient_str[i])
                new_digit = digit + offset
                if 0 <= new_digit <= 9:
                    new_quotient_str = quotient_str[:i] + str(new_digit) + quotient_str[i+1:]
                    distractors.add(int(new_quotient_str))
    
    # Error 2: Subtract incorrectly (e.g., subtract smaller from larger regardless of position)
    # This typically results in quotient being off by factors related to the divisor
    if num2 != 0:
        # Common error: missing a step in long division
        distractor = quotient - 1 if quotient > 0 else 0
        distractors.add(distractor)
        
        # Common error: adding an extra step
        distractor = quotient + 1
        distractors.add(distractor)
    
    # Remove the correct answer if it somehow got added
    distractors.discard(int(correct_ans))
    
    return list(distractors)


# HELPER FUNCTIONS
def get_nth_digit_string(number, n):
    """
    Gets the nth digit of a number from the left (1-based index).
    Accepts either int or str. For ints, uses absolute value; for strs,
    uses the numeric characters after stripping an optional leading + or -.
    Returns None if the digit does not exist or is non-numeric.
    """
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

# mapping from question type to distractor functions
# mapping from question type to distractor functions
_distractors_map = {
    "1A": [lambda q, ans: off_by_one_generic(q, ans, offsets=[-2, -1, 1, 2])],
    "1S": [lambda q, ans: off_by_one_generic(q, ans, offsets=[-2, -1, 1, 2])],
    "T5": [lambda q, ans: one_table_off(q, ans, offsets=[-2, -1, 1, 2])],
    "2A1": [lambda q, ans: add_wrong_place_value_addition(q, ans), 
            lambda q, ans: off_by_one_generic(q, ans)],
    "2A2": [lambda q, ans: add_wrong_place_value_addition(q, ans), 
            lambda q, ans: off_by_one_multidigit(q, ans)],
    "2S1": [lambda q, ans: add_instead_of(q, ans), 
            lambda q, ans: off_by_one_generic(q, ans)],
    "1AC": [lambda q, ans: off_by_one_generic(q, ans, offsets=[-2, -1, 1, 2])],
    "2A1C": [lambda q, ans: add_wrong_place_value_addition(q, ans), 
             lambda q, ans: off_by_one_generic(q, ans)],
    "2A2C": [lambda q, ans: add_wrong_place_value_addition(q, ans), 
             lambda q, ans: off_by_one_multidigit(q, ans)],
    "2S1B": [lambda q, ans: add_instead_of(q, ans), 
             lambda q, ans: off_by_one_generic(q, ans)],
    "2S2": [lambda q, ans: add_instead_of(q, ans), 
            lambda q, ans: off_by_one_multidigit(q, ans)],
    "T10": [lambda q, ans: one_table_off(q, ans, offsets=[-2, -1, 1, 2])],
    "3A": [lambda q, ans: add_wrong_place_value_addition(q, ans), 
           lambda q, ans: off_by_one_multidigit(q, ans)],
    "3AC": [lambda q, ans: add_wrong_place_value_addition(q, ans), 
            lambda q, ans: off_by_one_multidigit(q, ans)],
    "3S": [lambda q, ans: add_instead_of(q, ans), 
           lambda q, ans: off_by_one_multidigit(q, ans)],
    "2S2B": [lambda q, ans: add_instead_of(q, ans), 
             lambda q, ans: off_by_one_multidigit(q, ans)],
    "3AC2": [lambda q, ans: add_wrong_place_value_addition(q, ans), 
             lambda q, ans: off_by_one_multidigit(q, ans)],
    "3SB": [lambda q, ans: add_instead_of(q, ans), 
            lambda q, ans: off_by_one_multidigit(q, ans)],
    "3SB2": [lambda q, ans: add_instead_of(q, ans), 
             lambda q, ans: off_by_one_multidigit(q, ans)],
    "2M1": [lambda q, ans: add_instead_of_multiply(q, ans), 
            lambda q, ans: off_by_one_generic(q, ans)],
    "3M1": [lambda q, ans: add_instead_of_multiply(q, ans), 
            lambda q, ans: off_by_one_generic(q, ans, offsets=[-2, -1, 1, 2])],
    "2M1C": [lambda q, ans: add_instead_of_multiply(q, ans), 
             lambda q, ans: off_by_one_generic(q, ans, offsets=[-2, -1, 1, 2])],
    "3M1C": [lambda q, ans: add_instead_of_multiply(q, ans), 
             lambda q, ans: off_by_one_generic(q, ans, offsets=[-2, -1, 1, 2])],
    "3M1C2": [lambda q, ans: add_instead_of_multiply(q, ans), 
              lambda q, ans: off_by_one_generic(q, ans, offsets=[-2, -1, 1, 2])],
    "2M2": [lambda q, ans: add_instead_of_multiply(q, ans), 
            lambda q, ans: off_by_one_multidigit(q, ans)],
    "2D1": [lambda q, ans: division_errors(q, ans),
            lambda q, ans: off_by_one_generic(q, ans)],
    "3D1": [lambda q, ans: division_errors(q, ans),
            lambda q, ans: off_by_one_generic(q, ans)],
    "2M2C": [lambda q, ans: add_instead_of_multiply(q, ans), 
             lambda q, ans: off_by_one_multidigit(q, ans, offsets=[-2, -1, 1, 2])],
    "2D1R": [lambda q, ans: division_errors(q, ans),
             lambda q, ans: off_by_one_generic(q, ans)],
    "3D1R": [lambda q, ans: division_errors(q, ans),
             lambda q, ans: off_by_one_generic(q, ans)],
    "3M2C": [lambda q, ans: add_instead_of_multiply(q, ans), 
              lambda q, ans: off_by_one_multidigit(q, ans, offsets=[-2, -1, 1, 2])],
    "3D1Z": [lambda q, ans: division_errors(q, ans),
             lambda q, ans: off_by_one_generic(q, ans)],
    "4D1R": [lambda q, ans: division_errors(q, ans),
             lambda q, ans: off_by_one_generic(q, ans)]
}

def generate_distractors(skill_code, question, correct_ans):
    """Generate all distractors for a given skill code."""
    if skill_code not in _distractors_map:
        raise ValueError(f"Unknown skill code: {skill_code}")
    
    all_distractors = set()
    for func in _distractors_map[skill_code]:
        try:
            distractors = func(question, correct_ans)
            all_distractors.update(distractors)
        except Exception as e:
            print(f"Error generating distractors: {e}")
    
    return [d for d in all_distractors if _is_non_negative_option(d)]


def _build_non_negative_fallback_distractors(correct_ans, existing, needed=3):
    """Fill distractors up to `needed` with guaranteed non-negative candidates."""
    out = list(existing)

    if isinstance(correct_ans, str) and "R" in correct_ans:
        q, _, r = correct_ans.partition("R")
        try:
            qv = int(q)
            rv = int(r)
        except ValueError:
            qv, rv = 0, 0

        seed = [
            f"{qv + 1}R{rv}",
            f"{qv}R{rv + 1}",
            f"{qv + 1}R{rv + 1}",
            f"{qv + 2}R{rv}",
            f"{qv}R{rv + 2}",
        ]
    else:
        try:
            base = int(correct_ans)
        except (TypeError, ValueError):
            base = 0

        seed = [base + i for i in (1, 2, 3, 4, 5, 6)]

    for candidate in seed:
        if len(out) >= needed:
            break
        if candidate == correct_ans:
            continue
        if not _is_non_negative_option(candidate):
            continue
        if candidate in out:
            continue
        out.append(candidate)

    return out


def build_distractors(skill_code, question, correct_ans, needed=3):
    """Return at least `needed` non-negative distractors for a question."""
    try:
        possible_distractors = generate_distractors(skill_code, question, correct_ans)
    except Exception as e:
        print(f"Error generating distractors for {skill_code}: {e}")
        possible_distractors = []

    possible_distractors = [
        d for d in possible_distractors
        if _is_non_negative_option(d)
    ]

    if len(possible_distractors) < needed:
        # Preserve existing numeric fallback behavior.
        try:
            correct_val = int(correct_ans) if not isinstance(correct_ans, str) else int(correct_ans.split()[0])
            candidates = []
            for offset in [1, 2, 3, 4, 5, 6, 7, 8]:
                candidates.append(correct_val + offset)
            for offset in [1, 2, 3]:
                candidate = correct_val - offset
                if candidate >= 0:
                    candidates.append(candidate)
            possible_distractors.extend(candidates)
            possible_distractors = list(set(possible_distractors))
            possible_distractors = [d for d in possible_distractors if d != correct_val][:needed]
        except (ValueError, AttributeError):
            possible_distractors = []

    possible_distractors = _build_non_negative_fallback_distractors(
        correct_ans,
        [d for d in possible_distractors if d != correct_ans],
        needed=needed,
    )

    return possible_distractors