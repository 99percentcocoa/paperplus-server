import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

FIXTURES_DIR = Path(__file__).parent / "fixtures"
HEADERS = {"x-service-secret": settings.shared_secret}


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _post(client: TestClient, image_b64: str, headers: dict | None = HEADERS):
    return client.post("/process", json={"correlation_id": "abc", "image_b64": image_b64}, headers=headers)


def test_health_does_not_require_auth():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_process_requires_shared_secret():
    with TestClient(app) as client:
        assert _post(client, _b64(b"whatever"), headers=None).status_code == 401
        assert _post(client, _b64(b"whatever"), headers={"x-service-secret": "wrong"}).status_code == 401


def test_process_rejects_input_that_is_not_an_image():
    with TestClient(app) as client:
        assert _post(client, "!!! not base64 !!!").status_code == 400  # invalid base64
        assert _post(client, "").status_code == 400  # empty
        assert _post(client, _b64(b"valid base64, but plain text")).status_code == 400  # not an image


def test_process_rejects_the_old_path_based_request():
    """image_path is gone from the contract: a caller still sending it must fail loudly (422), not
    have vision-service quietly open files on its own disk."""
    with TestClient(app) as client:
        response = client.post(
            "/process", json={"correlation_id": "abc", "image_path": "/nonexistent.jpg"}, headers=HEADERS
        )
        assert response.status_code == 422


def test_process_is_stateless_and_returns_the_dewarped_page(monkeypatch):
    """A real scan goes in as bytes; the dewarped JPEG comes back in the response (the ROI boxes are
    in its coordinates), and nothing is written to disk."""

    def no_disk_writes(*args, **kwargs):
        raise AssertionError("vision-service must not write images to disk")

    monkeypatch.setattr(cv2, "imwrite", no_disk_writes)
    old_artifact_dir = Path("files")  # where earlier versions wrote dewarped/debug images
    files_before = sorted(old_artifact_dir.rglob("*")) if old_artifact_dir.exists() else []
    photo = (FIXTURES_DIR / "0151_1.jpeg").read_bytes()

    with TestClient(app) as client:
        response = _post(client, _b64(photo))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["worksheet_id"] == 3491 and len(body["question_marks"]) == 20
    assert body.get("dewarped_image_path") is None
    assert "debug_image_path" not in body

    dewarped = cv2.imdecode(np.frombuffer(base64.b64decode(body["dewarped_image_b64"]), np.uint8), cv2.IMREAD_COLOR)
    assert dewarped is not None
    assert dewarped.shape[:2] == (settings.target_height, settings.target_width)
    last = body["question_marks"][-1]
    assert last["roi_x2"] <= dewarped.shape[1] and last["roi_y2"] <= dewarped.shape[0]  # ROIs fit the returned image

    files_after = sorted(old_artifact_dir.rglob("*")) if old_artifact_dir.exists() else []
    assert files_after == files_before
