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


class ProjectMismatchError(ValueError):
    """Raised when a school/student being imported into one project already exists in another
    (student IDs are globally unique across projects, so this would silently cross them)."""


class VisionClientError(RuntimeError):
    """Raised when the vision-service call fails or returns an unusable result."""
