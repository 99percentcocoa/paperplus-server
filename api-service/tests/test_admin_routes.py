"""Admin dashboard JSON API + generalized /files route, driven through TestClient with the get_session
dependency pointed at the test's own session (same pattern as test_webhook.py)."""

from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlmodel import delete, select

from app.db.session import get_session
from app.main import app
from app.models import Scan, ScanReview, Submission, Worksheet
from app.models.core import utcnow
from app.routes.admin import get_vision_client
from shared.contracts import ProcessingResult, QuestionMark
from tests.conftest import ENV_SCHOOL_CODE, ENV_STUDENT_ID
from tests.fakes import FakeVisionClient

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

    summary = client.get("/api/admin/projects/paperplus/summary").json()
    assert set(summary) == {"schools", "students", "submissions", "submissions_24h", "open_reviews"}

    listing = client.get("/api/admin/projects/paperplus/submissions", params={"student_id": ENV_STUDENT_ID}).json()
    assert listing["total"] == 1
    item = listing["items"][0]
    assert (item["student_name"], item["score"], item["total_questions"], item["has_image"]) == ("Corrections Test", 2, 4, True)

    assert client.get("/api/admin/projects/paperplus/submissions", params={"student_id": "0000"}).json()["total"] == 0
    assert client.get("/api/admin/projects/paperplus/submissions", params={"limit": 0}).status_code == 422


def test_submission_detail_has_answer_key_and_servable_images(scan_env, client):
    scan_env.run_scan(MARKS)
    detail = client.get(f"/api/admin/projects/paperplus/submissions/{_submission_id(scan_env)}").json()

    assert detail["score"] == 2 and detail["total_questions"] == 4
    q2 = next(q for q in detail["questions"] if q["question_index"] == 2)
    assert (q2["selected_option"], q2["correct_option"], q2["is_correct"], q2["question_text"]) == ("B", "A", False, "Q2")
    assert q2["labels"] == ["A", "B", "C", "D"]

    images = detail["scans"][0]["images"]
    assert all(images[k] for k in ("upload", "checked"))
    assert "dewarped" not in images  # stored for redrawing the checked image, but never exposed
    served = client.get(images["checked"])
    assert served.status_code == 200 and served.headers["content-type"] == "image/jpeg"
    assert served.headers["x-robots-tag"] == "noindex, nofollow"

    assert client.get("/api/admin/projects/paperplus/submissions/999999999").status_code == 404


def test_patch_answers_saves_and_reports_errors(scan_env, client):
    scan_env.run_scan(MARKS)
    submission_id = _submission_id(scan_env)

    ok = client.patch(
        f"/api/admin/projects/paperplus/submissions/{submission_id}/answers",
        json={"corrected_by": "Asha", "corrections": [{"question_index": 2, "selected_option": "A"}]},
    )
    assert ok.status_code == 200 and ok.json() == {"submission_id": submission_id, "score": 3, "total_questions": 4}

    bad_option = client.patch(
        f"/api/admin/projects/paperplus/submissions/{submission_id}/answers",
        json={"corrected_by": "Asha", "corrections": [{"question_index": 2, "selected_option": "Z"}]},
    )
    assert bad_option.status_code == 400 and "not a valid option" in bad_option.json()["detail"]

    missing = client.patch(
        "/api/admin/projects/paperplus/submissions/999999999/answers",
        json={"corrected_by": "Asha", "corrections": [{"question_index": 1, "selected_option": "A"}]},
    )
    assert missing.status_code == 404
    assert client.patch(f"/api/admin/projects/paperplus/submissions/{submission_id}/answers", json={"corrections": []}).status_code == 422


def test_failed_scan_appears_in_reviews_and_can_be_resolved(scan_env, client):
    scan_env.run_scan(MARKS, roll_number="0000")

    reviews = client.get("/api/admin/projects/paperplus/reviews").json()
    item = next(r for r in reviews["items"] if r["detected_roll_number"] == "0000")
    assert (item["status"], item["resolvable"], item["has_image"]) == ("failed", True, True)
    assert "No registered student" in item["error_reason"]

    detail = client.get(f"/api/admin/projects/paperplus/reviews/{item['review_id']}").json()
    assert detail["resolvable"] is True and detail["worksheet_missing"] is False
    assert [q["detected_option"] for q in detail["questions"]] == ["A", "B", "A", ""]
    assert all(q["correct_option"] == "A" for q in detail["questions"])
    assert detail["scan"]["images"]["upload"]

    resolved = client.post(
        f"/api/admin/projects/paperplus/reviews/{item['review_id']}/resolve",
        json={"student_id": ENV_STUDENT_ID, "corrected_by": "Asha", "corrections": [{"question_index": 4, "selected_option": "A"}]},
    )
    assert resolved.status_code == 200 and resolved.json()["score"] == 3

    after = client.get(f"/api/admin/projects/paperplus/reviews/{item['review_id']}").json()
    assert after["status"] == "corrected" and after["resolvable"] is False and after["submission_id"]
    assert all(r["review_id"] != item["review_id"] for r in client.get("/api/admin/projects/paperplus/reviews").json()["items"])

    again = client.post(
        f"/api/admin/projects/paperplus/reviews/{item['review_id']}/resolve", json={"student_id": ENV_STUDENT_ID, "corrected_by": "Asha"}
    )
    assert again.status_code == 400


def test_retry_recovers_marks_for_a_scan_with_no_vision_result(scan_env, client):
    """Simulates the "tags not detected"/"roll number invalid" failure: vision-service raised
    entirely, so the review starts with no question_marks to grade from at all. Retrying with an
    admin-supplied worksheet_id should recover marks and make the review resolvable."""
    correlation_id = scan_env.run_failed_scan()
    scan = scan_env.session.exec(select(Scan).where(Scan.correlation_id == correlation_id)).one()
    review = scan_env.session.exec(select(ScanReview).where(ScanReview.scan_id == scan.id)).one()

    listed = next(r for r in client.get("/api/admin/projects/paperplus/reviews").json()["items"] if r["review_id"] == review.review_id)
    assert listed["resolvable"] is False and listed["worksheet_id"] is None

    retry_result = ProcessingResult(
        worksheet_id=scan_env.worksheet.worksheet_id, page_no=1, first_question_index=1,
        template_name="regular", roll_number=None, roll_number_confidence=None, question_paper_code="",
        question_marks=[
            QuestionMark(question_index=i, marked_option="A", confidence=0.9, roi_x1=0, roi_y1=0, roi_x2=1, roi_y2=1)
            for i in (1, 2, 3, 4)
        ],
    )
    app.dependency_overrides[get_vision_client] = lambda: FakeVisionClient(retry_result)
    try:
        retried = client.post(
            f"/api/admin/projects/paperplus/reviews/{review.review_id}/retry", json={"worksheet_id": scan_env.worksheet.worksheet_id}
        )
    finally:
        del app.dependency_overrides[get_vision_client]

    assert retried.status_code == 200
    body = retried.json()
    assert body["resolvable"] is True
    assert body["worksheet"]["worksheet_id"] == scan_env.worksheet.worksheet_id
    assert [q["detected_option"] for q in body["questions"]] == ["A", "A", "A", "A"]

    resolved = client.post(
        f"/api/admin/projects/paperplus/reviews/{review.review_id}/resolve", json={"student_id": ENV_STUDENT_ID, "corrected_by": "Asha"}
    )
    assert resolved.status_code == 200 and resolved.json()["score"] == 4

    assert client.post(
        "/api/admin/projects/paperplus/reviews/999999999/retry", json={"worksheet_id": scan_env.worksheet.worksheet_id}
    ).status_code == 404
    assert client.post(
        f"/api/admin/projects/paperplus/reviews/{review.review_id}/retry", json={"worksheet_id": scan_env.worksheet.worksheet_id}
    ).status_code == 400  # already resolved


def test_manual_grading_when_no_vision_result_and_no_retry(scan_env, client):
    """Covers grading straight from the photo with no vision-service help at all -- neither the
    corner/row tags nor a retry are required once the admin knows the worksheet and can see the
    image: every question starts blank and the admin fills each one in by hand."""
    correlation_id = scan_env.run_failed_scan()
    scan = scan_env.session.exec(select(Scan).where(Scan.correlation_id == correlation_id)).one()
    review = scan_env.session.exec(select(ScanReview).where(ScanReview.scan_id == scan.id)).one()

    bad_worksheet = client.get(f"/api/admin/projects/paperplus/reviews/{review.review_id}", params={"worksheet_id": 999999999})
    assert bad_worksheet.json()["worksheet_missing"] is True and bad_worksheet.json()["resolvable"] is False

    detail = client.get(
        f"/api/admin/projects/paperplus/reviews/{review.review_id}", params={"worksheet_id": scan_env.worksheet.worksheet_id}
    ).json()
    assert detail["resolvable"] is True and detail["manual_entry"] is True
    assert [q["detected_option"] for q in detail["questions"]] == ["", "", "", ""]
    assert [q["question_index"] for q in detail["questions"]] == [1, 2, 3, 4]

    resolved = client.post(
        f"/api/admin/projects/paperplus/reviews/{review.review_id}/resolve",
        json={
            "student_id": ENV_STUDENT_ID,
            "corrected_by": "Asha",
            "worksheet_id": scan_env.worksheet.worksheet_id,
            "corrections": [{"question_index": i, "selected_option": "A"} for i in (1, 2, 3, 4)],
        },
    )
    assert resolved.status_code == 200 and resolved.json()["score"] == 4


def test_review_status_and_unknown_review(scan_env, client):
    scan_env.run_scan(MARKS, roll_number="0000")
    review = scan_env.session.exec(select(ScanReview).where(ScanReview.detected_roll_number == "0000")).one()

    assert client.post(f"/api/admin/projects/paperplus/reviews/{review.review_id}/status", json={"status": "approved", "corrected_by": "Asha"}).json() == {
        "review_id": review.review_id, "status": "approved",
    }
    assert client.post(f"/api/admin/projects/paperplus/reviews/{review.review_id}/status", json={"status": "bogus"}).status_code == 400
    assert client.post("/api/admin/projects/paperplus/reviews/999999999/status", json={"status": "approved"}).status_code == 404
    assert client.get("/api/admin/projects/paperplus/reviews/999999999").status_code == 404
    assert client.get("/api/admin/projects/paperplus/reviews", params={"status": "approved"}).json()["total"] >= 1


def test_student_search_and_schools(scan_env, client):
    by_id = client.get("/api/admin/projects/paperplus/students", params={"q": ENV_STUDENT_ID[:3]}).json()
    assert any(s["student_id"] == ENV_STUDENT_ID for s in by_id)
    by_name = client.get("/api/admin/projects/paperplus/students", params={"q": "corrections tes"}).json()
    assert any(s["student_id"] == ENV_STUDENT_ID for s in by_name)
    schools = client.get("/api/admin/projects/paperplus/schools").json()
    assert any(s["school_code"] == "CRT" and s["student_count"] == 1 for s in schools)


def test_files_route_kinds_and_guards(scan_env, client):
    scan_env.run_scan(MARKS)
    name = f"crt-{scan_env.worksheet.worksheet_id}-1"

    assert client.get(f"/files/uploads/{name}.jpg").status_code == 200
    assert Path(scan_env.storage / "dewarped" / f"{name}_dewarped.jpg").is_file()  # kept on disk...
    assert client.get(f"/files/dewarped/{name}_dewarped.jpg").status_code == 404  # ...but not served
    assert client.get(f"/files/debug/{name}_debug.jpg").status_code == 404
    assert client.get(f"/files/checked/{name}_checked.jpg").headers["content-type"] == "image/jpeg"
    assert client.get("/files/secrets/anything.jpg").status_code == 404  # unknown kind
    assert client.get("/files/checked/missing.jpg").status_code == 404
    assert client.get("/files/checked/..").status_code in (400, 404)


def test_school_detail_sorts_never_submitted_first(scan_env, client):
    never_submitted = scan_env.add_student("9996")
    scan_env.run_scan(MARKS)  # ENV_STUDENT_ID submits, "9996" doesn't

    resp = client.get(f"/api/admin/projects/paperplus/schools/{ENV_SCHOOL_CODE}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["school"] == {"school_code": ENV_SCHOOL_CODE, "school_name": "Corrections Test School"}

    students = body["students"]
    assert [s["student_id"] for s in students] == [never_submitted.student_id, ENV_STUDENT_ID]

    never = students[0]
    assert never["last_submitted_at"] is None
    assert never["recent_worksheets"] == []

    submitted = students[1]
    assert submitted["last_submitted_at"] is not None
    assert len(submitted["recent_worksheets"]) == 1
    entry = submitted["recent_worksheets"][0]
    assert entry["worksheet_id"] == scan_env.worksheet.worksheet_id
    assert entry["level"] == "A"
    assert entry["score"] == 2
    assert entry["total_questions"] == 4


def test_school_detail_returns_only_last_three_worksheets(scan_env, client):
    extra_worksheets = [
        Worksheet(worksheet_level=str(i), worksheet_category="practice", lang="en", title=f"Extra {i}")
        for i in range(4)
    ]
    scan_env.session.add_all(extra_worksheets)
    scan_env.session.commit()
    for w in extra_worksheets:
        scan_env.session.refresh(w)

    now = utcnow()
    extra_submissions = [
        Submission(
            student_id=ENV_STUDENT_ID,
            worksheet_id=w.worksheet_id,
            worksheet_category="practice",
            score=idx,
            answers_json=[],
            submitted_at=now - timedelta(minutes=idx),
        )
        for idx, w in enumerate(extra_worksheets)
    ]
    scan_env.session.add_all(extra_submissions)
    scan_env.session.commit()

    try:
        resp = client.get(f"/api/admin/projects/paperplus/schools/{ENV_SCHOOL_CODE}")
        assert resp.status_code == 200
        [student] = [s for s in resp.json()["students"] if s["student_id"] == ENV_STUDENT_ID]
        assert len(student["recent_worksheets"]) == 3
        # Most recent 3 by submitted_at (minutes=0,1,2), i.e. the first three extra worksheets.
        expected_ids = [w.worksheet_id for w in extra_worksheets[:3]]
        assert [entry["worksheet_id"] for entry in student["recent_worksheets"]] == expected_ids
    finally:
        submission_ids = [s.submission_id for s in extra_submissions]
        scan_env.session.exec(delete(Submission).where(Submission.submission_id.in_(submission_ids)))
        worksheet_ids = [w.worksheet_id for w in extra_worksheets]
        scan_env.session.exec(delete(Worksheet).where(Worksheet.worksheet_id.in_(worksheet_ids)))
        scan_env.session.commit()


def test_school_detail_404_for_unknown_school(client):
    assert client.get("/api/admin/projects/paperplus/schools/DOESNOTEXIST").status_code == 404


def test_admin_ui_is_served_with_noindex(client):
    assert client.get("/admin/", follow_redirects=False).headers["location"] == "/admin/paperplus/"
    page = client.get("/admin/paperplus/")
    assert page.status_code == 200 and "/admin/assets/app.js" in page.text
    assert page.headers["x-robots-tag"] == "noindex, nofollow"
    assert client.get("/admin/navodaya/").status_code == 200
    assert client.get("/admin/nosuchproject/").status_code == 404
    assert client.get("/admin/assets/app.js").status_code == 200
    assert client.get("/admin", follow_redirects=False).headers["location"] == "/admin/paperplus/"
    assert client.get("/admin/paperplus", follow_redirects=False).headers["location"] == "/admin/paperplus/"


def test_retry_passes_admin_entered_fields_to_vision_and_skips_corner_tags(scan_env, client):
    correlation_id = scan_env.run_failed_scan()
    scan = scan_env.session.exec(select(Scan).where(Scan.correlation_id == correlation_id)).one()
    review = scan_env.session.exec(select(ScanReview).where(ScanReview.scan_id == scan.id)).one()

    seen = {}
    result = ProcessingResult(
        worksheet_id=None, page_no=None, first_question_index=None, template_name="regular",
        roll_number=ENV_STUDENT_ID, roll_number_confidence=None, question_paper_code="D", question_marks=[],
    )

    class RecordingVisionClient:
        def process(self, image_path, correlation_id, template_hint=None, skip_corner_tags=False,
                    roll_number=None, question_paper_code=None):
            seen.update(skip_corner_tags=skip_corner_tags, roll_number=roll_number, question_paper_code=question_paper_code)
            return result

    app.dependency_overrides[get_vision_client] = lambda: RecordingVisionClient()
    try:
        response = client.post(
            f"/api/admin/projects/paperplus/reviews/{review.review_id}/retry",
            json={"worksheet_id": scan_env.worksheet.worksheet_id, "roll_number": f" {ENV_STUDENT_ID} ", "question_paper_code": "d"},
        )
    finally:
        del app.dependency_overrides[get_vision_client]

    assert response.status_code == 200
    assert seen == {"skip_corner_tags": True, "roll_number": ENV_STUDENT_ID, "question_paper_code": "D"}
    assert response.json()["detected_roll_number"] == ENV_STUDENT_ID
