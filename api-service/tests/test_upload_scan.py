"""POST /api/admin/projects/{code}/scans: a photo sent straight to the API (curl) instead of over
WhatsApp, optionally with the handwritten fields (roll number, question-paper code) supplied."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import select

from app.db.session import get_session
from app.main import app
from app.models import Scan, ScanReview, Student
from app.routes.admin import get_vision_client
from shared.contracts import ProcessingResult, QuestionMark
from tests.conftest import ENV_FROM_NUMBER, ENV_STUDENT_ID
from tests.fakes import FailingFakeVisionClient

NAV_STUDENT_ID = "9993"
PP = "/api/admin/projects/paperplus/scans"
NAV = "/api/admin/projects/navodaya/scans"


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), color="white").save(buffer, format="JPEG")
    return buffer.getvalue()


class RecordingVisionClient:
    """Returns a graded-looking result for whatever roll number it's told (or `detected_roll`),
    and remembers the overrides it was called with."""

    def __init__(self, env, detected_roll=None):
        self.env = env
        self.detected_roll = detected_roll
        self.calls = []

    def process(self, image_path, correlation_id, template_hint=None, skip_corner_tags=False,
                roll_number=None, question_paper_code=None):
        self.calls.append({"roll_number": roll_number, "question_paper_code": question_paper_code})
        dewarped = self.env.storage / "dewarped" / f"{correlation_id}_dewarped.jpg"
        dewarped.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (700, 500), color="white").save(dewarped)
        return ProcessingResult(
            worksheet_id=self.env.worksheet.worksheet_id, page_no=1, first_question_index=1,
            template_name="regular", roll_number=roll_number or self.detected_roll,
            roll_number_confidence=None, question_paper_code=question_paper_code or "",
            question_marks=[
                QuestionMark(question_index=i, marked_option=o, confidence=0.9,
                             roi_x1=10, roi_y1=10 + 60 * (i - 1), roi_x2=300, roi_y2=60 + 60 * (i - 1))
                for i, o in ((1, "A"), (2, "B"), (3, "A"), (4, None))
            ],
            dewarped_image_path=str(dewarped),
        )


@pytest.fixture
def client(scan_env):
    app.dependency_overrides[get_session] = lambda: scan_env.session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def nav_student(scan_env):
    scan_env.session.add(Student(student_id=NAV_STUDENT_ID, student_name="Navodaya Upload", project_code="navodaya"))
    scan_env.session.commit()
    scan_env.extra_student_ids.append(NAV_STUDENT_ID)


def _use(vision):
    app.dependency_overrides[get_vision_client] = lambda: vision
    return vision


def _post(client, url, body=None, **params):
    params.setdefault("from_number", ENV_FROM_NUMBER)  # so scan_env's teardown removes the scan
    return client.post(url, content=_jpeg() if body is None else body, params=params)


def test_upload_with_supplied_fields_grades_and_returns_replies(scan_env, client, nav_student):
    vision = _use(RecordingVisionClient(scan_env))
    response = _post(client, NAV, roll_number=f" {NAV_STUDENT_ID} ", question_paper_code="d")
    assert response.status_code == 200, response.text
    data = response.json()

    assert vision.calls == [{"roll_number": NAV_STUDENT_ID, "question_paper_code": "D"}]
    assert data["outcome"] == "graded" and data["score"] == 2 and data["total_questions"] == 4
    assert data["roll_number"] == NAV_STUDENT_ID and data["review_id"] is None
    assert "Your marks: 2/4" in data["replies"]
    assert data["checked_image_url"].startswith("/files/checked/")
    assert scan_env.comm.sent_messages == []  # nothing went out over WhatsApp
    scan = scan_env.session.get(Scan, data["scan_id"])
    assert scan.project_code == "navodaya"


def test_upload_without_fields_reads_them_from_the_photo(scan_env, client):
    vision = _use(RecordingVisionClient(scan_env, detected_roll=ENV_STUDENT_ID))
    data = _post(client, PP).json()
    assert vision.calls == [{"roll_number": None, "question_paper_code": None}]
    assert data["outcome"] == "graded" and data["roll_number"] == ENV_STUDENT_ID


def test_supplied_student_from_another_project_is_rejected_up_front(scan_env, client, nav_student):
    vision = _use(RecordingVisionClient(scan_env))
    response = _post(client, PP, roll_number=NAV_STUDENT_ID)
    assert response.status_code == 400 and "paperplus" in response.json()["detail"]
    assert vision.calls == []
    assert _post(client, NAV, roll_number="does-not-exist").status_code == 400


def test_photo_read_as_a_student_from_another_project_becomes_a_failed_scan_here(scan_env, client, nav_student):
    _use(RecordingVisionClient(scan_env, detected_roll=NAV_STUDENT_ID))
    data = _post(client, PP).json()
    assert data["outcome"] == "failed" and "navodaya" in data["error_reason"]
    review = scan_env.session.get(ScanReview, data["review_id"])
    assert scan_env.session.get(Scan, review.scan_id).project_code == "paperplus"


def test_vision_failure_lands_on_this_projects_failed_scans(scan_env, client):
    _use(FailingFakeVisionClient())
    data = _post(client, NAV).json()
    assert data["outcome"] == "failed" and data["review_id"] is not None
    scan = scan_env.session.exec(select(Scan).where(Scan.correlation_id == data["correlation_id"])).one()
    assert scan.project_code == "navodaya"  # not "unassigned": it was uploaded through Navodaya's API
    reviews = client.get("/api/admin/projects/navodaya/reviews").json()["items"]
    assert any(r["review_id"] == data["review_id"] and not r["unassigned"] for r in reviews)


def test_rejects_empty_and_non_image_bodies(client):
    assert _post(client, NAV, body=b"").status_code == 400
    assert _post(client, NAV, body=b"roll_number=0151").status_code == 415
