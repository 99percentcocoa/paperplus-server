class InvalidStudentError(ValueError):
    """Raised when a student_id/roll number does not match a registered student."""


class InvalidWorksheetError(ValueError):
    """Raised when a worksheet_id does not match an existing worksheet."""


class InvalidSubmissionDataError(ValueError):
    """Raised when submission data (score, answers_json, etc.) is malformed."""


class VisionClientError(RuntimeError):
    """Raised when the vision-service call fails or returns an unusable result."""
