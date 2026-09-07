"""OCR for handwritten fields (roll number, question paper code).

PaddleOCR is a heavy dependency (~500MB+ models downloaded on first use) and this dev
environment is resource-constrained, so it's deferred: a Protocol-based provider interface
lets PaddleOCRProvider be swapped in later without touching predict_ocr's callers.
"""

from typing import Protocol

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None
    np = None


class OCRProvider(Protocol):
    def recognize(self, image_array) -> str:
        """Return the best-guess recognized text for the given image (empty string if none)."""
        ...


class StubOCRProvider:
    """Deterministic fallback used until PaddleOCRProvider is installed/wired in.

    Always returns "" (no text), which callers already treat as "OCR failed" —
    keeps the pipeline runnable end-to-end without the real OCR dependency installed.
    """

    def recognize(self, image_array) -> str:
        return ""


class PaddleOCRProvider:
    """Real provider — TODO(Phase 3 follow-up): install `paddleocr` and enable this.

    Kept here (rather than deferred entirely) so the exact variant-fallback strategy
    ported from the old services/inference.py is preserved and ready to activate.
    """

    def __init__(self):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("paddleocr is not installed; install it to use PaddleOCRProvider.") from exc

        self._ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv6_small_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def recognize(self, image_array) -> str:
        result = self._ocr.predict(image_array)
        if not result:
            return ""
        rec_texts = result[0].get("rec_texts") or []
        rec_scores = result[0].get("rec_scores") or []
        if not rec_texts or not rec_scores:
            return ""
        best_index = max(range(len(rec_scores)), key=lambda i: rec_scores[i])
        return rec_texts[best_index]


def _image_variants(image_array) -> list:
    """Fallback variants tried in order: RGB, grayscale, adaptive-threshold binary, inverted binary."""
    if cv2 is None:
        return [image_array]

    variants = [cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)]

    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    variants.append(gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    variants.append(binary)
    variants.append(cv2.bitwise_not(binary))

    return variants


def predict_ocr(image_array, provider: OCRProvider) -> str:
    """Try OCR variants in order, returning the first non-empty recognized text."""
    for variant in _image_variants(image_array):
        text = provider.recognize(variant)
        if text:
            return text
    return ""
