"""Admin dashboard JSON API + generalized /files route, driven through TestClient with the get_session
dependency pointed at the test's own session (same pattern as test_webhook.py)."""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from app.db.session import get_session
from app.main import app
from app.models import ScanReview, Submission
from tests.conftest import ENV_STUDENT_ID

MARKS = {1: "A", 2: "B", 3: "A", 4: None}  # correct answer is A everywhere -> score 2


@pytest.fixture
def client(scan_env):
    app.dependency_overrides[get_session] = lambda: scan_env.session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _submission_id(env) -> int:
    return env.session.exec(select(Submission).where(Submission.worksheet_id == env.worksheet.worksheet_id)).one().submission_id


def test_summary_and_submission_list_and_filters(scan_env, client):
    scan_env.run_scan(MARKS)

    summary = client.get("/api/admin/summary").json()
    assert set(summary) == {"schools", "students", "submissions", "submissions_24h", "open_reviews"}

    listing = client.get("/api/admin/submissions", params={"student_id": ENV_STUDENT_ID}).json()
    assert listing["total"] == 1
    item = listing["items"][0]
    assert (item["student_name"], item["score"], item["total_questions"], item["has_image"]) == ("Corrections Test", 2, 4, True)

    assert client.get("/api/admin/submissions", params={"student_id": "0000"}).json()["total"] == 0
    assert client.get("/api/admin/submissions", params={"limit": 0}).status_code == 422


def test_submission_detail_has_answer_key_and_servable_images(scan_env, client):
    scan_env.run_scan(MARKS)
    detail = client.get(f"/api/admin/submissions/{_submission_id(scan_env)}").json()

    assert detail["score"] == 2 and detail["total_questions"] == 4
    q2 = next(q for q in detail["questions"] if q["question_index"] == 2)
    assert (q2["selected_option"], q2["correct_option"], q2["is_correct"], q2["question_text"]) == ("B", "A", False, "Q2")
    assert q2["labels"] == ["A", "B", "C", "D"]

    images = detail["scans"][0]["images"]
    assert all(images[k] for k in ("upload", "dewarped", "checked"))
    served = client.get(images["checked"])
    assert served.status_code == 200 and served.headers["content-type"] == "image/jpeg"
    assert served.headers["x-robots-tag"] == "noindex, nofollow"

    assert client.get("/api/admin/submissions/999999999").status_code == 404


def test_patch_answers_saves_and_reports_errors(scan_env, client):
    scan_env.run_scan(MARKS)
    submission_id = _submission_id(scan_env)

    ok = client.patch(
        f"/api/admin/submissions/{submission_id}/answers",
        json={"corrected_by": "Asha", "corrections": [{"question_index": 2, "selected_option": "A"}]},
    )
    assert ok.status_code == 200 and ok.json() == {"submission_id": submission_id, "score": 3, "total_questions": 4}

    bad_option = client.patch(
        f"/api/admin/submissions/{submission_id}/answers",
        json={"corrected_by": "Asha", "corrections": [{"question_index": 2, "selected_option": "Z"}]},
    )
    assert bad_option.status_code == 400 and "not a valid option" in bad_option.json()["detail"]

    missing = client.patch(
        "/api/admin/submissions/999999999/answers",
        json={"corrected_by": "Asha", "corrections": [{"question_index": 1, "selected_option": "A"}]},
    )
    assert missing.status_code == 404
    assert client.patch(f"/api/admin/submissions/{submission_id}/answers", json={"corrections": []}).status_code == 422


def test_failed_scan_appears_in_reviews_and_can_be_resolved(scan_env, client):
    scan_env.run_scan(MARKS, roll_number="0000")

    reviews = client.get("/api/admin/reviews").json()
    item = next(r for r in reviews["items"] if r["detected_roll_number"] == "0000")
    assert (item["status"], item["resolvable"], item["has_image"]) == ("failed", True, True)
    assert "No registered student" in item["error_reason"]

    detail = client.get(f"/api/admin/reviews/{item['review_id']}").json()
    assert detail["resolvable"] is True and detail["worksheet_missing"] is False
    assert [q["detected_option"] for q in detail["questions"]] == ["A", "B", "A", ""]
    assert all(q["correct_option"] == "A" for q in detail["questions"])
    assert detail["scan"]["images"]["dewarped"]

    resolved = client.post(
        f"/api/admin/reviews/{item['review_id']}/resolve",
        json={"student_id": ENV_STUDENT_ID, "corrected_by": "Asha", "corrections": [{"question_index": 4, "selected_option": "A"}]},
    )
    assert resolved.status_code == 200 and resolved.json()["score"] == 3

    after = client.get(f"/api/admin/reviews/{item['review_id']}").json()
    assert after["status"] == "corrected" and after["resolvable"] is False and after["submission_id"]
    assert all(r["review_id"] != item["review_id"] for r in client.get("/api/admin/reviews").json()["items"])

    again = client.post(
        f"/api/admin/reviews/{item['review_id']}/resolve", json={"student_id": ENV_STUDENT_ID, "corrected_by": "Asha"}
    )
    assert again.status_code == 400


def test_review_status_and_unknown_review(scan_env, client):
    scan_env.run_scan(MARKS, roll_number="0000")
    review = scan_env.session.exec(select(ScanReview).where(ScanReview.detected_roll_number == "0000")).one()

    assert client.post(f"/api/admin/reviews/{review.review_id}/status", json={"status": "approved", "corrected_by": "Asha"}).json() == {
        "review_id": review.review_id, "status": "approved",
    }
    assert client.post(f"/api/admin/reviews/{review.review_id}/status", json={"status": "bogus"}).status_code == 400
    assert client.post("/api/admin/reviews/999999999/status", json={"status": "approved"}).status_code == 404
    assert client.get("/api/admin/reviews/999999999").status_code == 404
    assert client.get("/api/admin/reviews", params={"status": "approved"}).json()["total"] >= 1


def test_student_search_and_schools(scan_env, client):
    by_id = client.get("/api/admin/students", params={"q": ENV_STUDENT_ID[:3]}).json()
    assert any(s["student_id"] == ENV_STUDENT_ID for s in by_id)
    by_name = client.get("/api/admin/students", params={"q": "corrections tes"}).json()
    assert any(s["student_id"] == ENV_STUDENT_ID for s in by_name)
    schools = client.get("/api/admin/schools").json()
    assert any(s["school_code"] == "CRT" and s["student_count"] == 1 for s in schools)


def test_files_route_kinds_and_guards(scan_env, client):
    scan_env.run_scan(MARKS)
    name = f"crt-{scan_env.worksheet.worksheet_id}-1"

    assert client.get(f"/files/uploads/{name}.jpg").status_code == 200
    assert client.get(f"/files/dewarped/{name}_dewarped.jpg").status_code == 200
    assert client.get(f"/files/checked/{name}_checked.jpg").headers["content-type"] == "image/jpeg"
    assert client.get("/files/secrets/anything.jpg").status_code == 404  # unknown kind
    assert client.get("/files/checked/missing.jpg").status_code == 404
    assert client.get("/files/checked/..").status_code in (400, 404)


def test_admin_ui_is_served_with_noindex(client):
    page = client.get("/admin/")
    assert page.status_code == 200 and "PaperPlus" in page.text
    assert page.headers["x-robots-tag"] == "noindex, nofollow"
    assert client.get("/admin/app.js").status_code == 200
    assert client.get("/admin", follow_redirects=False).status_code in (301, 307)
