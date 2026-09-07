from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


def test_health_does_not_require_auth():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_process_requires_shared_secret():
    with TestClient(app) as client:
        response = client.post(
            "/process",
            json={"correlation_id": "abc", "image_path": "/nonexistent.jpg"},
        )
        assert response.status_code == 401


def test_process_rejects_unreadable_image():
    with TestClient(app) as client:
        response = client.post(
            "/process",
            json={"correlation_id": "abc", "image_path": "/nonexistent.jpg"},
            headers={"x-service-secret": settings.shared_secret},
        )
        assert response.status_code == 400
