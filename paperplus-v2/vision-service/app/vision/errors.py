class VisionError(Exception):
    """Base class for all vision-pipeline errors."""


class CornerTagDetectionError(VisionError):
    """Fewer than 4 corner (36h11) tags were detected; image cannot be dewarped."""


class RowTagDetectionError(VisionError):
    """Fewer than the required number of row (25h9) tags were detected."""


class RollNumberError(VisionError):
    """Roll number OCR did not produce a valid 4-digit number."""
