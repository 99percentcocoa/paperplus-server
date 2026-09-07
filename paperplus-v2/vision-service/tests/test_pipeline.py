import numpy as np
import pytest

from app.vision import pipeline as pipeline_module
from app.vision.errors import RollNumberError
from app.vision.tags import worksheet_id_to_rows


class FakeDetection:
    def __init__(self, tag_id, center):
        self.tag_id = tag_id
        self.center = center


class FakeDetectionResult:
    def __init__(self, tag_ids, sorted_row_detections=None, sorted_corner_detections=None):
        self.tag_ids = tag_ids
        self.sorted_row_detections = sorted_row_detections or []
        self.sorted_corner_detections = sorted_corner_detections or []


class FakeBubbleClassifier:
    def predict_bubble(self, bubble_image_array):
        return ("Unmarked", 0.9)


class FakeOCRProvider:
    def __init__(self, roll_number="1234", paper_code=""):
        self._roll_number = roll_number
        self._paper_code = paper_code

    def recognize(self, image_array) -> str:
        # First ROI extracted is roll_number in the "regular" template.
        return self._roll_number


@pytest.fixture
def cropped_image():
    # Big enough to contain both handwritten-field ROIs and question ROIs used by "regular".
    return np.zeros((1754, 1240, 3), dtype=np.uint8)


def _patch_detection(monkeypatch, cropped_image, row_tag_ids, row_centers):
    def fake_detect_apriltags(image_array, tag_family):
        if tag_family == "36h11":
            return FakeDetectionResult(tag_ids=[0, 1, 2, 3], sorted_corner_detections=[
                FakeDetection(0, (0, 0)), FakeDetection(1, (10, 0)),
                FakeDetection(2, (10, 10)), FakeDetection(3, (0, 10)),
            ])
        return FakeDetectionResult(
            tag_ids=row_tag_ids,
            sorted_row_detections=[FakeDetection(i, c) for i, c in enumerate(row_centers)],
        )

    monkeypatch.setattr(pipeline_module, "detect_apriltags", fake_detect_apriltags)
    monkeypatch.setattr(pipeline_module, "crop_image", lambda *a, **k: cropped_image)
    monkeypatch.setattr(pipeline_module, "apply_median_blur", lambda image: image)


def test_process_scan_assigns_sequential_question_indices(monkeypatch, cropped_image):
    legacy_rows = worksheet_id_to_rows(42)
    _patch_detection(monkeypatch, cropped_image, legacy_rows, row_centers=[(300, 400), (300, 500)])

    result = pipeline_module.process_scan(
        cropped_image,
        target_width=1240,
        target_height=1754,
        bubble_classifier=FakeBubbleClassifier(),
        ocr_provider=FakeOCRProvider(roll_number="1234"),
    )

    assert result.worksheet_id == 42
    assert result.template_name == "regular"
    assert result.roll_number == "1234"
    # "regular" has 2 question ROI columns per row tag -> 2 row tags * 2 columns = 4 marks.
    assert [m.question_index for m in result.question_marks] == [1, 2, 3, 4]


def test_process_scan_infers_basic_omr_from_omr_v2_row_metadata(monkeypatch, cropped_image):
    omr_v2_rows = worksheet_id_to_rows(7, page_no=2, first_question_index=40)
    _patch_detection(monkeypatch, cropped_image, omr_v2_rows, row_centers=[(300, 400)])

    result = pipeline_module.process_scan(
        cropped_image,
        target_width=1240,
        target_height=1754,
        bubble_classifier=FakeBubbleClassifier(),
        ocr_provider=FakeOCRProvider(roll_number="0001"),
    )

    assert result.template_name == "basic_omr"
    assert result.first_question_index == 40
    # "basic_omr" has 3 question ROI columns per row tag -> 1 row tag * 3 columns = 3 marks.
    assert [m.question_index for m in result.question_marks] == [40, 41, 42]


def test_process_scan_raises_on_invalid_roll_number(monkeypatch, cropped_image):
    legacy_rows = worksheet_id_to_rows(1)
    _patch_detection(monkeypatch, cropped_image, legacy_rows, row_centers=[(300, 400)])

    with pytest.raises(RollNumberError):
        pipeline_module.process_scan(
            cropped_image,
            target_width=1240,
            target_height=1754,
            bubble_classifier=FakeBubbleClassifier(),
            ocr_provider=FakeOCRProvider(roll_number="not-a-number"),
        )


def test_process_scan_template_hint_overrides_inference(monkeypatch, cropped_image):
    legacy_rows = worksheet_id_to_rows(1)
    _patch_detection(monkeypatch, cropped_image, legacy_rows, row_centers=[(300, 400)])

    result = pipeline_module.process_scan(
        cropped_image,
        target_width=1240,
        target_height=1754,
        bubble_classifier=FakeBubbleClassifier(),
        ocr_provider=FakeOCRProvider(roll_number="1234"),
        template_hint="basic_omr",
    )

    assert result.template_name == "basic_omr"
