import random
from typing import Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json

from models import Question

# Type for answers: either an int (for most), or (quotient, remainder) tuple for divisions with remainder
Answer = Union[int, Tuple[int, int]]
# Cache for skills data
_skills_cache = None

# -----------------------
# Addition / Subtraction
# -----------------------

def gen_1A() -> Tuple[str, Answer]:
    """1-digit addition - no carry"""
    a = random.randint(1, 8)
    b = random.randint(0, 9 - a)  # ensure a+b < 10
    q = f"{a} + {b}"
    return q, a + b

def gen_1S() -> Tuple[str, Answer]:
    """1-digit subtraction - no borrow (minuend >= subtrahend)"""
    a = random.randint(1, 9)
    b = random.randint(0, a)
    q = f"{a} - {b}"
    return q, a - b

def gen_T5() -> Tuple[str, Answer]:
    """Multiplication tables - up to 5 (i.e., pick 1..5)"""
    a = random.randint(1, 10)
    b = random.randint(1, 5)
    q = f"{a} × {b}"
    return q, a * b

def gen_2A1() -> Tuple[str, Answer]:
    """2 + 1 digit addition - no carry (two-digit + one-digit, units sum < 10)"""
    tens = random.randint(1, 9)
    units = random.randint(0, 8)
    one = random.randint(0, 9 - units)  # ensure units + one < 10
    a = 10 * tens + units
    q = f"{a} + {one}"
    return q, a + one

def gen_2A2() -> Tuple[str, Answer]:
    """2 + 2 digit addition - no carry (units sum < 10)"""
    t1 = random.randint(1, 9)
    u1 = random.randint(0, 9)
    t2 = random.randint(1, 9)
    u2 = random.randint(0, 9 - u1)  # ensure unit-digit no carry
    a = 10 * t1 + u1
    b = 10 * t2 + u2
    return f"{a} + {b}", a + b

def gen_2S1() -> Tuple[str, Answer]:
    """2 - 1 digit subtraction - no borrow (two-digit minuend minus one-digit subtrahend; units >= subtrahend)"""
    tens = random.randint(1, 9)
    units = random.randint(0, 9)
    one = random.randint(0, units)  # ensure no borrow
    a = 10 * tens + units
    return f"{a} - {one}", a - one

def gen_1AC() -> Tuple[str, Answer]:
    """1-digit addition - carry (sum >= 10)"""
    a = random.randint(1, 9)
    b = random.randint(1, 9)
    # ensure carry
    max_iterations = 1000
    iterations = 0
    while a + b < 10:
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        iterations += 1
        if iterations > max_iterations:
            # Fallback: force a carry
            a, b = 9, 9
            break
    return f"{a} + {b}", a + b

def gen_2A1C() -> Tuple[str, Answer]:
    """2-digit addition - carry (2-digit + ? single carry occurs in units)"""
    tens = random.randint(1, 9)
    u1 = random.randint(0, 9)
    one = random.randint(0, 9)
    # ensure units produce carry
    max_iterations = 1000
    iterations = 0
    while (u1 + one) < 10:
        u1 = random.randint(0, 9)
        one = random.randint(0, 9)
        iterations += 1
        if iterations > max_iterations:
            # Fallback: force a carry
            u1, one = 9, 9
            break
    a = 10 * tens + u1
    return f"{a} + {one}", a + one

def gen_2A2C() -> Tuple[str, Answer]:
    """2 + 2 digit addition - double carry (carry from units to tens AND tens to hundreds)"""
    # select digits such that:
    # units sum >= 10 -> carry1 = 1
    # tens sum + carry1 >= 10 -> carry2 = 1
    max_iterations = 1000
    iterations = 0
    while True:
        u1 = random.randint(0, 9)
        u2 = random.randint(0, 9)
        c1 = 1 if (u1 + u2) >= 10 else 0
        t1 = random.randint(1, 9)
        t2 = random.randint(1, 9)
        if (t1 + t2 + c1) >= 10:
            a = 10 * t1 + u1
            b = 10 * t2 + u2
            return f"{a} + {b}", a + b
        iterations += 1
        if iterations > max_iterations:
            # Fallback: force double carry
            a, b = 99, 99
            return f"{a} + {b}", a + b

def gen_2S1B() -> Tuple[str, Answer]:
    """2 - 1 digit subtraction - borrow (units < subtrahend -> borrow occurs)"""
    tens = random.randint(1, 9)
    units = random.randint(0, 8)
    # choose subtrahend > units to force borrow
    sub = random.randint(units + 1, 9)
    a = 10 * tens + units
    return f"{a} - {sub}", a - sub

def gen_2S2() -> Tuple[str, Answer]:
    """2 - 2 digit subtraction - no borrow (digitwise minuend >= subtrahend)"""
    t1 = random.randint(1, 9)
    u1 = random.randint(0, 9)
    t2 = random.randint(1, t1)  # ensure tens >=
    u2 = random.randint(0, u1)  # ensure units >=
    a = 10 * t1 + u1
    b = 10 * t2 + u2
    return f"{a} - {b}", a - b

def gen_2S2B() -> Tuple[str, Answer]:
    """2 - 2 digit subtraction - single borrow (units place requires borrow, tens does not)"""
    max_iterations = 1000
    iterations = 0
    while True:
        t1 = random.randint(1, 9)
        u1 = random.randint(0, 9)
        t2 = random.randint(1, 9)
        u2 = random.randint(0, 9)
        a = 10 * t1 + u1
        b = 10 * t2 + u2
        if a <= b:
            iterations += 1
            if iterations > max_iterations:
                # Fallback: construct a valid single borrow case
                a, b = 52, 28
                return f"{a} - {b}", a - b
            continue
        # Single borrow: units requires borrow but tens doesn't
        # Units borrow if u1 < u2
        # After borrowing for units, tens has (t1 - 1), which must be >= t2
        if u1 < u2 and (t1 - 1) >= t2:
            return f"{a} - {b}", a - b
        iterations += 1
        if iterations > max_iterations:
            # Fallback: construct a valid single borrow case
            a, b = 52, 28
            return f"{a} - {b}", a - b

def gen_T10() -> Tuple[str, Answer]:
    """Multiplication tables - 5 to 10 (i.e., choose multiplier 5..10)"""
    a = random.randint(6, 10)
    b = random.randint(2, 10)
    return f"{a} × {b}", a * b

def gen_3A() -> Tuple[str, Answer]:
    """3-digit addition - no carry (no digit pair sums >= 10)"""
    d1 = random.randint(1, 9)
    d2 = random.randint(0, 9)
    d3 = random.randint(0, 9)
    max_iterations = 1000
    iterations = 0
    while True:
        e1 = random.randint(1, 9)
        e2 = random.randint(0, 9)
        e3 = random.randint(0, 9)
        if d3 + e3 < 10 and d2 + e2 < 10 and d1 + e1 < 10:
            a = 100 * d1 + 10 * d2 + d3
            b = 100 * e1 + 10 * e2 + e3
            return f"{a} + {b}", a + b
        iterations += 1
        if iterations > max_iterations:
            # Fallback: construct a valid no-carry case
            a, b = 111, 222
            return f"{a} + {b}", a + b

def gen_3AC() -> Tuple[str, Answer]:
    """3-digit addition - single carry (exactly one column causes carry)"""
    # build digits such that exactly one of the three columns has sum >= 10
    max_iterations = 1000
    iterations = 0
    while True:
        A = [random.randint(1, 9), random.randint(0, 9), random.randint(0, 9)]
        B = [random.randint(1, 9), random.randint(0, 9), random.randint(0, 9)]
        carries = 0
        # unit
        if A[2] + B[2] >= 10:
            carries += 1
            c1 = 1
        else:
            c1 = 0
        # tens
        if A[1] + B[1] + c1 >= 10:
            carries += 1
            c2 = 1
        else:
            c2 = 0
        # hundreds
        if A[0] + B[0] + c2 >= 10:
            carries += 1
        if carries == 1:
            a = 100 * A[0] + 10 * A[1] + A[2]
            b = 100 * B[0] + 10 * B[1] + B[2]
            return f"{a} + {b}", a + b
        iterations += 1
        if iterations > max_iterations:
            # Fallback: construct a case with exactly 1 carry
            a, b = 105, 108  # Only units carry: 5+8=13
            return f"{a} + {b}", a + b

def gen_3S() -> Tuple[str, Answer]:
    """3-digit subtraction - no borrow (digitwise minuend >= subtrahend)"""
    A0 = random.randint(1, 9)
    A1 = random.randint(0, 9)
    A2 = random.randint(0, 9)
    B0 = random.randint(1, A0)
    B1 = random.randint(0, A1)
    B2 = random.randint(0, A2)
    a = 100 * A0 + 10 * A1 + A2
    b = 100 * B0 + 10 * B1 + B2
    return f"{a} - {b}", a - b

def gen_3AC2() -> Tuple[str, Answer]:
    """3-digit addition - double carry (exactly two columns cause a carry)"""
    max_iterations = 1000
    iterations = 0
    while True:
        A = [random.randint(1, 9), random.randint(0, 9), random.randint(0, 9)]
        B = [random.randint(1, 9), random.randint(0, 9), random.randint(0, 9)]
        carries = 0
        c1 = 1 if A[2] + B[2] >= 10 else 0
        if c1: carries += 1
        c2 = 1 if (A[1] + B[1] + c1) >= 10 else 0
        if c2: carries += 1
        c3 = 1 if (A[0] + B[0] + c2) >= 10 else 0
        if c3: carries += 1
        # double carry -> exactly 2 columns produced carry
        if carries == 2:
            a = 100 * A[0] + 10 * A[1] + A[2]
            b = 100 * B[0] + 10 * B[1] + B[2]
            return f"{a} + {b}", a + b
        iterations += 1
        if iterations > max_iterations:
            # Fallback: construct a case with exactly 2 carries
            a, b = 195, 108  # Units carry (5+8=13), tens carry (9+0+1=10)
            return f"{a} + {b}", a + b

def gen_3SB() -> Tuple[str, Answer]:
    """3-digit subtraction - single borrow (exactly one borrow occurs)"""
    max_iterations = 1000
    iterations = 0
    while True:
        A = [random.randint(1, 9), random.randint(0, 9), random.randint(0, 9)]
        B = [random.randint(0, 9), random.randint(0, 9), random.randint(0, 9)]
        # require A >= B overall
        a = 100 * A[0] + 10 * A[1] + A[2]
        b = 100 * B[0] + 10 * B[1] + B[2]
        if a <= b:
            iterations += 1
            if iterations > max_iterations:
                # Fallback: construct a valid single borrow case
                a, b = 325, 218  # Only units borrow: 5<8
                return f"{a} - {b}", a - b
            continue
        # compute borrows by simulating column subtraction
        borrows = 0
        A2, A1, A0 = A[2], A[1], A[0]  # units, tens, hundreds (we'll treat reversed)
        # units
        if A2 < B[2]:
            borrows += 1
            # simulate borrow: tens reduced by 1
            t1 = A1 - 1
        else:
            t1 = A1
        # tens
        if t1 < B[1]:
            borrows += 1
            h = A0 - 1
        else:
            h = A0
        # hundreds
        if h < B[0]:
            borrows += 1
        if borrows == 1:
            return f"{a} - {b}", a - b
        iterations += 1
        if iterations > max_iterations:
            # Fallback: construct a valid single borrow case
            a, b = 325, 218  # Only units borrow: 5<8
            return f"{a} - {b}", a - b

def gen_3SB2() -> Tuple[str, Answer]:
    """3-digit subtraction - double borrow (exactly two borrows occur)"""
    max_iterations = 1000
    iterations = 0
    while True:
        A = [random.randint(1, 9), random.randint(0, 9), random.randint(0, 9)]
        B = [random.randint(0, 9), random.randint(0, 9), random.randint(0, 9)]
        a = 100 * A[0] + 10 * A[1] + A[2]
        b = 100 * B[0] + 10 * B[1] + B[2]
        if a <= b:
            iterations += 1
            if iterations > max_iterations:
                # Fallback: construct a valid double borrow case
                a, b = 302,  198  # Both units and tens borrow
                return f"{a} - {b}", a - b
            continue
        borrows = 0
        A2, A1, A0 = A[2], A[1], A[0]
        if A2 < B[2]:
            borrows += 1
            t1 = A1 - 1
        else:
            t1 = A1
        if t1 < B[1]:
            borrows += 1
            h = A0 - 1
        else:
            h = A0
        if h < B[0]:
            borrows += 1
        if borrows == 2:
            return f"{a} - {b}", a - b
        iterations += 1
        if iterations > max_iterations:
            # Fallback: construct a valid double borrow case
            a, b = 302, 198  # Both units and tens borrow
            return f"{a} - {b}", a - b

# -----------------------
# Multiplication
# -----------------------

def gen_2M1() -> Tuple[str, Answer]:
    """2x1 multiplication - no carry (each digit * multiplier < 10)"""
    max_iterations = 1000
    iterations = 0
    while True:
        multiplier = random.randint(2, 9)
        tens = random.randint(1, 9)
        units = random.randint(0, 9)
        if multiplier * units < 10 and multiplier * tens < 10:
            a = 10 * tens + units
            return f"{a} × {multiplier}", a * multiplier
        iterations += 1
        if iterations > max_iterations:
            # Fallback: construct a valid no-carry case
            a, multiplier = 11, 2
            return f"{a} × {multiplier}", a * multiplier

def gen_3M1() -> Tuple[str, Answer]:
    """3x1 multiplication - no carry (each digit * multiplier < 10)"""
    max_iterations = 1000
    iterations = 0
    while True:
        multiplier = random.randint(2, 9)
        h = random.randint(1, 9)
        t = random.randint(0, 9)
        u = random.randint(0, 9)
        if multiplier * u < 10 and multiplier * t < 10 and multiplier * h < 10:
            a = 100 * h + 10 * t + u
            return f"{a} × {multiplier}", a * multiplier
        iterations += 1
        if iterations > max_iterations:
            # Fallback: construct a valid no-carry case
            a, multiplier = 111, 2
            return f"{a} × {multiplier}", a * multiplier

def gen_2M1C() -> Tuple[str, Answer]:
    """2x1 multiplication - carry (at least one digit*multiplier >= 10)"""
    max_iterations = 1000
    iterations = 0
    while True:
        multiplier = random.randint(2, 9)
        tens = random.randint(1, 9)
        units = random.randint(0, 9)
        if multiplier * units >= 10 or multiplier * tens >= 10:
            a = 10 * tens + units
            return f"{a} × {multiplier}", a * multiplier
        iterations += 1
        if iterations > max_iterations:
            # Fallback: force a carry
            a, multiplier = 15, 2
            return f"{a} × {multiplier}", a * multiplier

def gen_3M1C() -> Tuple[str, Answer]:
    """3x1 multiplication - single carry (exactly one digit*multiplier produces carry)"""
    max_iterations = 1000
    iterations = 0
    while True:
        multiplier = random.randint(2, 9)
        h = random.randint(1, 9)
        t = random.randint(0, 9)
        u = random.randint(0, 9)
        prod_flags = [multiplier * u >= 10, multiplier * t >= 10, multiplier * h >= 10]
        if sum(prod_flags) == 1:
            a = 100 * h + 10 * t + u
            return f"{a} × {multiplier}", a * multiplier
        iterations += 1
        if iterations > max_iterations:
            # Fallback: exactly one carry
            a, multiplier = 105, 2
            return f"{a} × {multiplier}", a * multiplier

def gen_3M1C2() -> Tuple[str, Answer]:
    """3x1 multiplication - double carry (exactly two digit*multiplier produces carry)"""
    max_iterations = 1000
    iterations = 0
    while True:
        multiplier = random.randint(2, 9)
        h = random.randint(1, 9)
        t = random.randint(0, 9)
        u = random.randint(0, 9)
        prod_flags = [multiplier * u >= 10, multiplier * t >= 10, multiplier * h >= 10]
        if sum(prod_flags) == 2:
            a = 100 * h + 10 * t + u
            return f"{a} × {multiplier}", a * multiplier
        iterations += 1
        if iterations > max_iterations:
            # Fallback: exactly two carries
            a, multiplier = 566, 2
            return f"{a} × {multiplier}", a * multiplier

def gen_2M2() -> Tuple[str, Answer]:
    """2x2 multiplication - no carry. (each single-digit product < 10)"""
    max_iterations = 1000
    iterations = 0
    while True:
        a1 = random.randint(1, 9)
        a0 = random.randint(0, 9)
        b1 = random.randint(1, 9)
        b0 = random.randint(0, 9)
        if (a0 * b0 < 10) and (a0 * b1 < 10) and (a1 * b0 < 10) and (a1 * b1 < 10):
            a = 10 * a1 + a0
            b = 10 * b1 + b0
            return f"{a} × {b}", a * b
        iterations += 1
        if iterations > max_iterations:
            # Fallback: no carry case
            a, b = 11, 22
            return f"{a} × {b}", a * b

def gen_2M2C() -> Tuple[str, Answer]:
    """2x2 multiplication - carry (at least one single-digit product >= 10)"""
    max_iterations = 1000
    iterations = 0
    while True:
        a1 = random.randint(1, 9)
        a0 = random.randint(0, 9)
        b1 = random.randint(1, 9)
        b0 = random.randint(0, 9)
        cond = (a0 * b0 >= 10) or (a0 * b1 >= 10) or (a1 * b0 >= 10) or (a1 * b1 >= 10)
        if cond:
            a = 10 * a1 + a0
            b = 10 * b1 + b0
            return f"{a} × {b}", a * b
        iterations += 1
        if iterations > max_iterations:
            # Fallback: force a carry
            a, b = 15, 16
            return f"{a} × {b}", a * b

def gen_3M2C() -> Tuple[str, Answer]:
    """3x2 multiplication - carry (three-digit times two-digit where at least one single-digit product >= 10)"""
    max_iterations = 1000
    iterations = 0
    while True:
        a2 = random.randint(1, 9)
        a1 = random.randint(0, 9)
        a0 = random.randint(0, 9)
        b1 = random.randint(1, 9)
        b0 = random.randint(0, 9)
        cond = any(x >= 10 for x in [a0*b0, a1*b0, a2*b0, a0*b1, a1*b1, a2*b1])
        if cond:
            a = 100 * a2 + 10 * a1 + a0
            b = 10 * b1 + b0
            return f"{a} × {b}", a * b
        iterations += 1
        if iterations > max_iterations:
            # Fallback: force a carry
            a, b = 115, 16
            return f"{a} × {b}", a * b

# -----------------------
# Division
# -----------------------

def gen_2D1() -> Tuple[str, Answer]:
    """
    2-digit ÷ 1-digit division WITHOUT remainder.
    e.g., 48 ÷ 6 = 8
    """
    divisor = random.randint(2, 9)
    quotient = random.randint(1, 9)     # ensures dividend remains 2-digit
    tens = random.randint(1, 9)         # create 2-digit quotient indirectly
    dividend = (10 * tens + quotient) * divisor
    if dividend < 10 or dividend > 99:
        return gen_2D1()                # retry if not 2-digit
    return f"{dividend} ÷ {divisor}", dividend // divisor

def gen_3D1() -> Tuple[str, Answer]:
    """3/1 division without remainder (three-digit dividend divided by one-digit divisor evenly)"""
    max_iterations = 1000
    iterations = 0
    while True:
        dividend = random.randint(100, 999)
        divisor = random.randint(2, 9)
        if dividend % divisor == 0:
            return f"{dividend} ÷ {divisor}", dividend // divisor
        iterations += 1
        if iterations > max_iterations:
            # Fallback: construct a valid division
            dividend, divisor = 120, 2
            return f"{dividend} ÷ {divisor}", dividend // divisor

def gen_2D1R() -> Tuple[str, Answer]:
    """2/1 division with remainder (two-digit dividend divided by one-digit divisor with remainder)"""
    max_iterations = 1000
    iterations = 0
    while True:
        dividend = random.randint(10, 99)
        divisor = random.randint(2, 9)
        if dividend % divisor != 0:
            return f"{dividend} ÷ {divisor}", (dividend // divisor, dividend % divisor)
        iterations += 1
        if iterations > max_iterations:
            # Fallback: construct a valid division with remainder
            dividend, divisor = 23, 5
            return f"{dividend} ÷ {divisor}", (dividend // divisor, dividend % divisor)

def gen_3D1R() -> Tuple[str, Answer]:
    """3/1 division with remainder (three-digit dividend divided by one-digit divisor with remainder)"""
    max_iterations = 1000
    iterations = 0
    while True:
        dividend = random.randint(100, 999)
        divisor = random.randint(2, 9)
        if dividend % divisor != 0:
            return f"{dividend} ÷ {divisor}", (dividend // divisor, dividend % divisor)
        iterations += 1
        if iterations > max_iterations:
            # Fallback: construct a valid division with remainder
            dividend, divisor = 123, 5
            return f"{dividend} ÷ {divisor}", (dividend // divisor, dividend % divisor)

def gen_3D1Z() -> Tuple[str, Answer]:
    """3/1 division with 0 in quotient (we produce a division where quotient's middle digit is 0, no remainder)"""
    max_iterations = 1000
    iterations = 0
    while True:
        divisor = random.randint(2, 9)
        # construct quotient as a0b with 3-digit quotient
        a = random.randint(1, 4)  # keep product in 3-digit
        b = random.randint(0, 9)
        quo = 100 * a + 0 * 10 + b  # a0b
        dividend = divisor * quo
        if 100 <= dividend <= 999:
            return f"{dividend} ÷ {divisor}", quo
        iterations += 1
        if iterations > max_iterations:
            # Fallback: construct a valid case
            dividend, divisor, quo = 202, 2, 101
            return f"{dividend} ÷ {divisor}", quo

def gen_4D1R() -> Tuple[str, Answer]:
    """4/1 division with remainder (four-digit dividend divided by one-digit divisor with remainder)"""
    max_iterations = 1000
    iterations = 0
    while True:
        dividend = random.randint(1000, 9999)
        divisor = random.randint(2, 9)
        if dividend % divisor != 0:
            return f"{dividend} ÷ {divisor}", (dividend // divisor, dividend % divisor)
        iterations += 1
        if iterations > max_iterations:
            # Fallback: construct a valid division with remainder
            dividend, divisor = 1234, 5
            return f"{dividend} ÷ {divisor}", (dividend // divisor, dividend % divisor)

# -----------------------
# Dispatcher and helpers
# -----------------------

_gen_map = {
    "1A": gen_1A,
    "1S": gen_1S,
    "T5": gen_T5,
    "2A1": gen_2A1,
    "2A2": gen_2A2,
    "2S1": gen_2S1,
    "1AC": gen_1AC,
    "2A1C": gen_2A1C,
    "2A2C": gen_2A2C,
    "2S1B": gen_2S1B,
    "2S2": gen_2S2,
    "T10": gen_T10,
    "3A": gen_3A,
    "3AC": gen_3AC,
    "3S": gen_3S,
    "2S2B": gen_2S2B,
    "3AC2": gen_3AC2,
    "3SB": gen_3SB,
    "3SB2": gen_3SB2,
    "2M1": gen_2M1,
    "3M1": gen_3M1,
    "2M1C": gen_2M1C,
    "3M1C": gen_3M1C,
    "3M1C2": gen_3M1C2,
    "2M2": gen_2M2,
    "2D1": gen_2D1,
    "3D1": gen_3D1,
    "2M2C": gen_2M2C,
    "2D1R": gen_2D1R,
    "3D1R": gen_3D1R,
    "3M2C": gen_3M2C,
    "3D1Z": gen_3D1Z,
    "4D1R": gen_4D1R,
}

def gen_questions(code: str, n: int):
    """
    Generate n questions for the given skill code.
    Returns a list of (question_str, answer) tuples.
    """
    code = code.strip()
    if code not in _gen_map:
        raise ValueError(f"Unknown code: {code}")
    
    generator = _gen_map[code]
    out = []
    for _ in range(n):
        out.append(generator())
    return out

# Backwards-compatible single-question call
def gen_question(code: str):
    return gen_questions(code, 1)[0]

# utils
# TODO: replace this with database logic
def _load_skills():
    """Load skills data from skills.json file."""
    global _skills_cache
    if _skills_cache is not None:
        return _skills_cache
    
    skills_file = Path(__file__).parent / "skills.json"
    
    try:
        with open(skills_file, 'r') as f:
            data = json.load(f)
            # If data is a list, assume it's [{"answerKey": [...]}, [...skills...]]
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                # Check if first element has answerKey (old format)
                if "answerKey" in data[0]:
                    _skills_cache = {}
                else:
                    # It's a list of skill objects
                    _skills_cache = {skill["code"]: skill for skill in data}
            else:
                # It's a dict or list of skills
                _skills_cache = {skill["code"]: skill for skill in data}
            return _skills_cache
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get_difficulty_level(skill_code):
    """
    Get the difficulty level for a given skill code.
    
    Args:
        skill_code: str representing the skill code (e.g., "1A", "2M1")
    
    Returns:
        int or str: The difficulty level, or None if skill code not found
    
    Raises:
        ValueError: If skill code is not found in skills.json
    """
    skills = _load_skills()
    
    if skill_code not in skills:
        raise ValueError(f"Skill code '{skill_code}' not found in skills.json")
    
    difficulty = skills[skill_code].get("difficulty_level")
    # Convert to int if it's a string
    if isinstance(difficulty, str):
        try:
            return int(difficulty)
        except ValueError:
            return difficulty
    return difficulty


def number_to_letter(num):
    """
    Convert numeric answer (1, 2, 3, 4) to letter (A, B, C, D).
    
    Args:
        num: int or str representing the answer position (1-4)
    
    Returns:
        str: The corresponding letter (A, B, C, or D)
    
    Raises:
        ValueError: If input is not 1-4
    """
    # Convert to int if string
    if isinstance(num, str):
        try:
            num = int(num)
        except ValueError:
            raise ValueError(f"Invalid input: {num}. Must be '1', '2', '3', or '4'")
    
    # Map number to letter
    mapping = {1: "A", 2: "B", 3: "C", 4: "D"}
    
    if num not in mapping:
        raise ValueError(f"Invalid answer number: {num}. Must be 1, 2, 3, or 4")
    
    return mapping[num]


def letter_to_index(letter):
    """
    Convert letter answer (A, B, C, D) to index (0, 1, 2, 3).
    
    Args:
        letter: str representing the answer letter (A-D, case-insensitive)
    
    Returns:
        int: The corresponding index (0, 1, 2, or 3)
    
    Raises:
        ValueError: If input is not A-D
    """
    # Convert to uppercase if needed
    if isinstance(letter, str):
        letter = letter.upper()
    else:
        raise ValueError(f"Invalid input type: {type(letter)}. Must be a string")
    
    # Map letter to index
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    
    if letter not in mapping:
        raise ValueError(f"Invalid answer letter: {letter}. Must be A, B, C, or D")
    
    return mapping[letter]

def question_to_marathi(question: Question) -> Question:
    """
    Convert a Question object to a Marathi string representation.
    
    Args:
        question: Question object to convert
    
    Returns:
        Question: question with question_text and options converted to Marathi
    """
    # Placeholder implementation - this would need actual translation logic
    num1, num2 = [question.question_text.split(" ")[i] for i in [0, -1]]
    marathi_num1 = arabic_to_devanagari(num1)
    marathi_num2 = arabic_to_devanagari(num2)

    # Replace numbers in question text
    marathi_question_text = question.question_text.replace(num1, marathi_num1).replace(num2, marathi_num2)

    # Convert options to Marathi
    marathi_options = [arabic_to_devanagari(str(opt)) for opt in question.options]
    
    # Convert possible_distractors to Marathi
    marathi_possible_distractors = [arabic_to_devanagari(str(opt)) for opt in question.possible_distractors]

    return Question(
        index=question.index,
        question_text=marathi_question_text,
        skill_code=question.skill_code,
        options=marathi_options,
        answer=question.answer,
        correct_option=question.correct_option,
        possible_distractors=marathi_possible_distractors
    )

def arabic_to_devanagari(number_string: str) -> str:
  """Converts a string of Arabic numerals to Devanagari numerals."""
  arabic_numerals = '0123456789'
  devanagari_numerals = '०१२३४५६७८९'
  translation_table = str.maketrans(arabic_numerals, devanagari_numerals)
  return number_string.translate(translation_table)

# -----------------------
# Quick demo when run as script
# -----------------------
if __name__ == "__main__":
    # show one example for each code in the mapping
    for k in sorted(_gen_map.keys()):
        q, a = _gen_map[k]()
        print(f"{k}: {q} -> {a}")

