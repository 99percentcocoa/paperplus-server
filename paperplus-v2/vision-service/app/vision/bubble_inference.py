"""Bubble mark classification via the TFLite models, ported from services/inference.py."""

import math

try:
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter
    from PIL import Image
except ImportError:  # pragma: no cover
    np = None
    Interpreter = None
    Image = None

TARGET_SIZE = 128
MARKED = "Marked"
UNMARKED = "Unmarked"
OPTION_LABELS = ["A", "B", "C", "D"]


class BubbleClassifier:
    """Wraps a single TFLite interpreter instance; preload once at service startup."""

    def __init__(self, model_path: str):
        if Interpreter is None:
            raise RuntimeError("ai-edge-litert is not installed; cannot load the bubble model.")
        self._interpreter = Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    def predict_bubble(self, bubble_image_array) -> tuple[str, float]:
        """Returns (result, confidence_fraction) where result in {"Marked", "Unmarked"}."""
        preprocessed = _preprocess_image(bubble_image_array, TARGET_SIZE)

        self._interpreter.set_tensor(self._input_details[0]["index"], preprocessed)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_details[0]["index"])
        logit = float(output[0][0])

        probability = 1.0 / (1.0 + math.exp(-logit))
        if probability < 0.5:
            return MARKED, (1.0 - probability)
        return UNMARKED, probability


def _preprocess_image(image_array, target_size: int):
    if Image is None:
        raise RuntimeError("Pillow is not installed; cannot preprocess bubble image.")
    pil_image = Image.fromarray(image_array).convert("RGB").resize((target_size, target_size))
    array = np.asarray(pil_image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def get_cropped_bubbles_roi(roi_image_array) -> list:
    """Split a question ROI horizontally into 4 equal bubble images (A, B, C, D order)."""
    roi_width = roi_image_array.shape[1]
    part_width = roi_width // 4
    bubbles = []
    for i in range(4):
        start_x = i * part_width
        end_x = start_x + part_width if i < 3 else roi_width
        bubbles.append(roi_image_array[:, start_x:end_x].copy())
    return bubbles


def detect_bubble_marks(roi_image_array, classifier: BubbleClassifier) -> tuple[str | None, float]:
    """Classify which single option (A-D) is marked in a question ROI.

    Returns (marked_option, confidence_fraction). marked_option is None if zero or
    more than one bubble is marked (unanswered / ambiguous), matching the old system's
    "" sentinel.
    """
    bubbles = get_cropped_bubbles_roi(roi_image_array)
    marked: list[tuple[str, float]] = []
    for label, bubble in zip(OPTION_LABELS, bubbles):
        result, confidence = classifier.predict_bubble(bubble)
        if result == MARKED:
            marked.append((label, confidence))

    if len(marked) == 1:
        return marked[0]
    if len(marked) == 0:
        return None, 0.0
    # Multiple bubbles marked: ambiguous: report None with the lowest confidence among them.
    return None, min(c for _, c in marked)
