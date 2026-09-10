"""Mock-webhook fixture harness (Phase 6): drives the real POST /webhook route end-to-end
through FastAPI's TestClient -- message parsing, image download, per-message dispatch to
handle_incoming_image, and the Phase 5 batch-resilience guard -- rather than calling
handle_incoming_image directly like test_submission_service.py does. VisionClient/
CommunicationClient are swapped for fakes via app.dependency_overrides (get_vision_client/
get_comm_client, added to webhook.py specifically to make this possible); the DB session
override points at this test's own Session so assertions/cleanup can use it directly; image
download is faked by monkeypatching httpx.get so no real network call happens.

Uses an unregistered roll number ("0000") to reach the InvalidStudentError path deliberately --
that's enough to exercise the whole webhook plumbing (parse -> download -> dispatch -> reply)
without needing a full worksheet/questions fixture, since grading logic itself is already
covered by test_submission_service.py.
"""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, delete, select

from app.db.session import engine, get_session
from app.main import app
from app.models import ScanReview
from app.routes.webhook import get_comm_client, get_vision_client
from shared.contracts import ProcessingResult
from tests.fakes import FakeCommunicationClient, FakeVisionClient

UNREGISTERED_STUDENT_RESULT = ProcessingResult(
    worksheet_id=1,
    page_no=1,
    first_question_index=1,
    template_name="regular",
    roll_number="0000",
    roll_number_confidence=None,
    question_paper_code="",
    question_marks=[],
)


def _valid_image_payload(from_number: str = "+911234567890", image_url: str = "http://fake-cdn/img.jpg") -> dict:
    return {
        "whatsapp": {
            "messages": [
                {
                    "from": from_number,
                    "callback_type": "incoming_message",
                    "content": {"type": "image", "image": {"url": image_url}},
                }
            ]
        }
    }


@pytest.fixture
def session():
    with Session(engine) as s:
        yield s


@pytest.fixture
def comm_client():
    return FakeCommunicationClient()


@pytest.fixture
def client(session: Session, comm_client: FakeCommunicationClient):
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_vision_client] = lambda: FakeVisionClient(UNREGISTERED_STUDENT_RESULT)
    app.dependency_overrides[get_comm_client] = lambda: comm_client
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class _FakeImageResponse:
    status_code = 200
    headers = {"Content-Type": "image/jpeg"}
    content = b"fake-image-bytes"

    def raise_for_status(self):
        pass


def _cleanup_scan_reviews(session: Session):
    session.exec(delete(ScanReview).where(ScanReview.detected_roll_number == "0000"))
    session.commit()


def test_webhook_processes_valid_image_message_end_to_end(client: TestClient, session: Session, comm_client, monkeypatch):
    monkeypatch.setattr("app.routes.webhook.httpx.get", lambda *a, **k: _FakeImageResponse())

    response = client.post("/webhook", json=_valid_image_payload())

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert len(comm_client.sent_messages) == 1
    assert "Roll number not recognized" in comm_client.sent_messages[0][1]

    _cleanup_scan_reviews(session)


def test_webhook_ignores_non_image_messages(client: TestClient, session: Session, comm_client, monkeypatch):
    monkeypatch.setattr("app.routes.webhook.httpx.get", lambda *a, **k: _FakeImageResponse())

    payload = {
        "whatsapp": {
            "messages": [
                {"from": "+911234567890", "callback_type": "incoming_message", "content": {"type": "text", "text": {"body": "hi"}}}
            ]
        }
    }
    response = client.post("/webhook", json=payload)

    assert response.status_code == 200
    assert comm_client.sent_messages == []


def test_webhook_ignores_delivery_receipt_callbacks(client: TestClient, session: Session, comm_client, monkeypatch):
    monkeypatch.setattr("app.routes.webhook.httpx.get", lambda *a, **k: _FakeImageResponse())

    payload = _valid_image_payload()
    payload["whatsapp"]["messages"][0]["callback_type"] = "message_status"
    response = client.post("/webhook", json=payload)

    assert response.status_code == 200
    assert comm_client.sent_messages == []


def test_webhook_skips_message_on_download_failure_but_returns_ok(client: TestClient, session: Session, comm_client, monkeypatch):
    import httpx

    def _raise(*a, **k):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("app.routes.webhook.httpx.get", _raise)

    response = client.post("/webhook", json=_valid_image_payload())

    assert response.status_code == 200
    assert comm_client.sent_messages == []


def test_webhook_continues_batch_after_one_message_raises_unhandled_error(session: Session, monkeypatch):
    """The Phase 5 try/except around handle_incoming_image must not let one message's crash
    stop the rest of the batch from being processed.
    """
    monkeypatch.setattr("app.routes.webhook.httpx.get", lambda *a, **k: _FakeImageResponse())

    comm_client = FakeCommunicationClient()
    app.dependency_overrides[get_session] = lambda: session
    # FailingFakeVisionClient raises VisionClientError -- a *handled* exception (caught inside
    # handle_incoming_image itself), so use it for message 1 to prove handling still occurs, and
    # a genuinely unhandled RuntimeError for message 2's underlying client to prove the batch
    # survives even a bug handle_incoming_image doesn't itself catch.
    call_count = {"n": 0}

    class SequencedVisionClient:
        def process(self, image_path, correlation_id, template_hint=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("unexpected bug, not a VisionClientError")
            return UNREGISTERED_STUDENT_RESULT

    app.dependency_overrides[get_vision_client] = lambda: SequencedVisionClient()
    app.dependency_overrides[get_comm_client] = lambda: comm_client

    payload = {
        "whatsapp": {
            "messages": [
                {"from": "+911111111111", "callback_type": "incoming_message", "content": {"type": "image", "image": {"url": "http://fake-cdn/1.jpg"}}},
                {"from": "+912222222222", "callback_type": "incoming_message", "content": {"type": "image", "image": {"url": "http://fake-cdn/2.jpg"}}},
            ]
        }
    }

    try:
        with TestClient(app) as client:
            response = client.post("/webhook", json=payload)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert call_count["n"] == 2  # both messages were attempted despite message 1 crashing
    # Only message 2 reached a reply -- message 1's RuntimeError was swallowed by the batch guard.
    assert len(comm_client.sent_messages) == 1
    assert comm_client.sent_messages[0][0] == "+912222222222"

    _cleanup_scan_reviews(session)
