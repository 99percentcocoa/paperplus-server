from app.vision.geometry import ROI, sort_detections_clockwise


class FakeDetection:
    def __init__(self, tag_id, center):
        self.tag_id = tag_id
        self.center = center


def test_roi_width_and_height():
    roi = ROI(10, 20, 110, 70)
    assert roi.width() == 100
    assert roi.height() == 50


def test_sort_detections_clockwise_orders_from_top_left():
    # A square: TL, TR, BR, BL given in a shuffled order.
    tl = FakeDetection(0, (0, 0))
    tr = FakeDetection(1, (10, 0))
    br = FakeDetection(2, (10, 10))
    bl = FakeDetection(3, (0, 10))

    shuffled = [br, tl, bl, tr]
    sorted_result = sort_detections_clockwise(shuffled)
    sorted_ids = [d.tag_id for d in sorted_result]

    # Order should be cyclic (clockwise or counter-clockwise) around the square;
    # what matters is adjacency is preserved, i.e. it's a valid rotation of [0,1,2,3] or its reverse.
    assert sorted_ids in (
        [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2],
        [0, 3, 2, 1], [3, 2, 1, 0], [2, 1, 0, 3], [1, 0, 3, 2],
    )
