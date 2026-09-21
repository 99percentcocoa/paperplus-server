"""HTTPVisionClient: the photo goes to vision-service as base64 (vision-service never sees a path),
and the dewarped page that comes back is stored on api-service's own disk."""

import base64
import io

import httpx
import pytest
from PIL import Image

from app.core.config import settings
from app.domain.errors import VisionClientError
from app.services import vision_client
from app.services.vision_client import HTTPVisionClient
from shared.contracts import ProcessingResult, QuestionMark


def _jpeg_bytes(color: str = "white") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _result(**overrides) -> ProcessingResult:
    fields = dict(
        worksheet_id=1, page_no=1, first_question_index=1, template_name="regular", roll_number="0001",
        roll_number_confidence=None, question_paper_code="",
        question_marks=[QuestionMark(question_index=1, marked_option="A", confidence=0.9, roi_x1=1, roi_y1=2, roi_x2=3, roi_y2=4)],
    )
    fields.update(overrides)
    return ProcessingResult(**fields)


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=httpx.Request("POST", "http://vision"), response=httpx.Response(self.status_code))

    def json(self):
        return self._payload


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_root", str(tmp_path / "storage"))
    return tmp_path / "storage"


@pytest.fixture
def upload(tmp_path):
    path = tmp_path / "photo.jpg"
    path.write_bytes(_jpeg_bytes("blue"))
    return path


def _client_with_response(monkeypatch, payload: dict, sent: list | None = None, status: int = 200) -> HTTPVisionClient:
    def fake_post(url, json=None, headers=None, timeout=None):
        if sent is not None:
            sent.append({"url": url, "json": json, "headers": headers})
        return FakeResponse(payload, status)

    monkeypatch.setattr(vision_client.httpx, "post", fake_post)
    return HTTPVisionClient(base_url="http://vision:8100", shared_secret="s3cret")


def test_sends_the_photo_as_base64_and_never_a_path(monkeypatch, storage, upload):
    sent: list = []
    client = _client_with_response(monkeypatch, _result().model_dump(), sent)

    client.process(str(upload), "corr-vc-1", template_hint="basic_omr")

    request = sent[0]
    assert request["url"] == "http://vision:8100/process"
    assert request["headers"] == {"x-service-secret": "s3cret"}
    assert set(request["json"]) == {"correlation_id", "image_b64", "template_hint"}  # no image_path
    assert base64.b64decode(request["json"]["image_b64"]) == upload.read_bytes()
    assert request["json"]["template_hint"] == "basic_omr"


def test_stores_the_returned_dewarped_image_locally_and_drops_the_base64(monkeypatch, storage, upload):
    dewarped = _jpeg_bytes("green")
    payload = _result(dewarped_image_b64=base64.b64encode(dewarped).decode("ascii")).model_dump()
    client = _client_with_response(monkeypatch, payload)

    result = client.process(str(upload), "corr-vc-2")

    expected = storage / "dewarped" / "corr-vc-2_dewarped.jpg"
    assert result.dewarped_image_path == str(expected)
    assert expected.read_bytes() == dewarped
    # The blob must not survive into what handle_incoming_image persists as scans.vision_result.
    assert result.dewarped_image_b64 is None
    assert "dewarped_image_b64" in result.model_dump() and result.model_dump()["dewarped_image_b64"] is None
    assert result.question_marks[0].roi_x2 == 3  # the ROI data passes through untouched


def test_response_without_a_dewarped_image_is_fine(monkeypatch, storage, upload):
    result = _client_with_response(monkeypatch, _result().model_dump()).process(str(upload), "corr-vc-3")
    assert result.dewarped_image_path is None and not (storage / "dewarped").exists()


def test_undecodable_dewarped_image_degrades_instead_of_failing_the_scan(monkeypatch, storage, upload):
    """Grading already succeeded on vision-service's side; a bad image payload should only cost the
    checked image, not the whole submission."""
    client = _client_with_response(monkeypatch, _result(dewarped_image_b64="***not base64***").model_dump())
    result = client.process(str(upload), "corr-vc-4")
    assert result.dewarped_image_path is None and result.dewarped_image_b64 is None
    assert result.roll_number == "0001"


def test_missing_upload_is_a_vision_client_error_without_calling_vision(monkeypatch, storage, tmp_path):
    sent: list = []
    client = _client_with_response(monkeypatch, _result().model_dump(), sent)
    with pytest.raises(VisionClientError, match="could not read upload"):
        client.process(str(tmp_path / "does-not-exist.jpg"), "corr-vc-5")
    assert sent == []


def test_http_errors_still_become_vision_client_errors(monkeypatch, storage, upload):
    client = _client_with_response(monkeypatch, {"detail": "no corner tags"}, status=422)
    with pytest.raises(VisionClientError, match="vision-service call failed"):
        client.process(str(upload), "corr-vc-6")
