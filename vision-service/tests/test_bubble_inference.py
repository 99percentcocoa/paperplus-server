import numpy as np

from app.vision.bubble_inference import MARKED, UNMARKED, detect_bubble_marks, get_cropped_bubbles_roi


def test_get_cropped_bubbles_roi_splits_into_four_equal_parts():
    roi = np.zeros((10, 40, 3), dtype=np.uint8)
    bubbles = get_cropped_bubbles_roi(roi)
    assert len(bubbles) == 4
    assert all(b.shape == (10, 10, 3) for b in bubbles)


class FakeClassifier:
    """Test double: returns a scripted (result, confidence) per call, in order."""

    def __init__(self, responses):
        self._responses = iter(responses)

    def predict_bubble(self, bubble_image_array):
        return next(self._responses)


def test_detect_bubble_marks_single_marked_option():
    roi = np.zeros((10, 40, 3), dtype=np.uint8)
    classifier = FakeClassifier([(UNMARKED, 0.9), (MARKED, 0.95), (UNMARKED, 0.8), (UNMARKED, 0.7)])
    marked_option, confidence = detect_bubble_marks(roi, classifier)
    assert marked_option == "B"
    assert confidence == 0.95


def test_detect_bubble_marks_none_marked():
    roi = np.zeros((10, 40, 3), dtype=np.uint8)
    classifier = FakeClassifier([(UNMARKED, 0.9)] * 4)
    marked_option, confidence = detect_bubble_marks(roi, classifier)
    assert marked_option is None
    assert confidence == 0.0


def test_detect_bubble_marks_multiple_marked_is_ambiguous():
    roi = np.zeros((10, 40, 3), dtype=np.uint8)
    classifier = FakeClassifier([(MARKED, 0.6), (MARKED, 0.55), (UNMARKED, 0.9), (UNMARKED, 0.8)])
    marked_option, confidence = detect_bubble_marks(roi, classifier)
    assert marked_option is None
    assert confidence == 0.55
