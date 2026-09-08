class InvalidStudentError(ValueError):
    """Raised when a student_id/roll number does not match a registered student."""


class InvalidWorksheetError(ValueError):
    """Raised when a worksheet_id does not match an existing worksheet."""


class InvalidSubmissionDataError(ValueError):
    """Raised when submission data (score, answers_json, etc.) is malformed."""


class InvalidAnswerKeyError(ValueError):
    """Raised when a worksheet has no resolvable answer key (no question_paper_variant seeded
    for the scanned code, and no canonical question_options.is_correct fallback either) --
    grading would otherwise silently score every question as incorrect.
    """


class VisionClientError(RuntimeError):
    """Raised when the vision-service call fails or returns an unusable result."""
