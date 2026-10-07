import numpy as np

from app.vision.ocr import StubOCRProvider, predict_ocr


def test_stub_provider_returns_empty_string():
    assert StubOCRProvider().recognize(np.zeros((5, 5, 3), dtype=np.uint8)) == ""


class FakeProvider:
    """Test double: succeeds only on the Nth variant tried."""

    def __init__(self, succeed_on_call: int, text: str = "1234"):
        self._call_count = 0
        self._succeed_on_call = succeed_on_call
        self._text = text

    def recognize(self, image_array) -> str:
        self._call_count += 1
        if self._call_count == self._succeed_on_call:
            return self._text
        return ""


def test_predict_ocr_returns_first_non_empty_variant_result():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    provider = FakeProvider(succeed_on_call=3)
    assert predict_ocr(image, provider) == "1234"


def test_predict_ocr_returns_empty_when_all_variants_fail():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    provider = FakeProvider(succeed_on_call=999)
    assert predict_ocr(image, provider) == ""
