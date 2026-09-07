import pytest

from app.vision.apriltag_detector import DetectionResult
from app.vision.errors import CornerTagDetectionError, RowTagDetectionError


class FakeDetection:
    def __init__(self, tag_id, center):
        self.tag_id = tag_id
        self.center = center


def _square_corners():
    return [
        FakeDetection(0, (0, 0)),
        FakeDetection(1, (10, 0)),
        FakeDetection(2, (10, 10)),
        FakeDetection(3, (0, 10)),
    ]


def test_corner_detection_requires_at_least_four():
    with pytest.raises(CornerTagDetectionError):
        DetectionResult(detections=_square_corners()[:3], tag_family="36h11")


def test_corner_detection_sorts_clockwise():
    result = DetectionResult(detections=_square_corners(), tag_family="36h11")
    assert len(result.sorted_corner_detections) == 4
    assert result.tag_ids == [0, 1, 2, 3]


def test_row_detection_requires_ten_tags():
    with pytest.raises(RowTagDetectionError):
        DetectionResult(detections=[FakeDetection(i, (0, i)) for i in range(5)], tag_family="25h9")


def test_row_detection_filters_invalid_ids_and_sorts_top_to_bottom():
    detections = [
        FakeDetection(5, (0, 300)),
        FakeDetection(40, (0, 999)),  # out of valid range [0, 34], must be filtered
        *[FakeDetection(i, (0, 100 - i)) for i in range(9)],
    ]
    result = DetectionResult(detections=detections, tag_family="25h9")
    assert 40 not in result.tag_ids
    ys = [d.center[1] for d in result.sorted_row_detections]
    assert ys == sorted(ys)
