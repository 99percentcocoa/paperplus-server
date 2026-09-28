"""Project separation (PaperPlus vs Navodaya): scans are routed to a project by their student,
every /api/admin/projects/{code}/... endpoint only sees its own project's data, and failed scans
with no identifiable student are "unassigned" -- visible on every project until resolved."""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import delete, select

from app.db.session import get_session
from app.domain.errors import ProjectMismatchError
from app.domain.worksheet_import import upsert_school, upsert_student
from app.main import app
from app.models import Scan, ScanReview, School, Student, Submission
from tests.conftest import ENV_SCHOOL_CODE, ENV_STUDENT_ID

MARKS = {1: "A", 2: "B", 3: "A", 4: None}
NAV_STUDENT_ID = "9993"
NAV_SCHOOL_CODE = "NAVT"
PP = "/api/admin/projects/paperplus"
NAV = "/api/admin/projects/navodaya"


@pytest.fixture
def nav(scan_env):
    """A Navodaya school (no students) and a school-less Navodaya student; the student is torn
    down by scan_env (with its submissions/mastery), the school here."""
    session = scan_env.session
    session.add(School(school_code=NAV_SCHOOL_CODE, school_name="Navodaya Test School", project_code="navodaya"))
    session.add(Student(student_id=NAV_STUDENT_ID, student_name="Navodaya Test", project_code="navodaya"))
    session.commit()
    scan_env.extra_student_ids.append(NAV_STUDENT_ID)
    yield scan_env
    session.rollback()
    session.exec(delete(School).where(School.school_code == NAV_SCHOOL_CODE))
    session.commit()


@pytest.fixture
def client(scan_env):
    app.dependency_overrides[get_session] = lambda: scan_env.session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _scan(env, correlation_id) -> Scan:
    return env.session.exec(select(Scan).where(Scan.correlation_id == correlation_id)).one()


def _review_ids(client, base) -> set[int]:
    return {r["review_id"] for r in client.get(f"{base}/reviews?status=all&limit=200").json()["items"]}


def test_project_info_and_unknown_project(client):
    assert client.get(NAV).json()["project_name"] == "Navodaya"
    assert client.get("/api/admin/projects").status_code in (404, 405)  # no endpoint enumerating projects
    assert client.get("/api/admin/projects/nosuchproject/summary").status_code == 404


def test_scan_is_routed_to_the_students_project(nav, client):
    correlation_id = nav.run_scan(MARKS, roll_number=NAV_STUDENT_ID)
    assert _scan(nav, correlation_id).project_code == "navodaya"
    submission_id = nav.session.exec(
        select(Submission.submission_id).where(Submission.student_id == NAV_STUDENT_ID)
    ).one()

    nav_ids = {s["submission_id"] for s in client.get(f"{NAV}/submissions?limit=200").json()["items"]}
    pp_ids = {s["submission_id"] for s in client.get(f"{PP}/submissions?limit=200").json()["items"]}
    assert submission_id in nav_ids and submission_id not in pp_ids

    assert client.get(f"{NAV}/submissions/{submission_id}").status_code == 200
    assert client.get(f"{PP}/submissions/{submission_id}").status_code == 404
    patch = {"corrected_by": "Asha", "corrections": [{"question_index": 2, "selected_option": "A"}]}
    assert client.patch(f"{PP}/submissions/{submission_id}/answers", json=patch).status_code == 404

    weeks = client.get(f"{NAV}/metrics/weekly?weeks=1").json()
    assert weeks["weeks"][-1]["total_worksheets"] >= 1


def test_schools_and_students_are_scoped(nav, client):
    nav_schools = {s["school_code"] for s in client.get(f"{NAV}/schools").json()}
    pp_schools = {s["school_code"] for s in client.get(f"{PP}/schools").json()}
    assert NAV_SCHOOL_CODE in nav_schools and NAV_SCHOOL_CODE not in pp_schools
    assert ENV_SCHOOL_CODE in pp_schools and ENV_SCHOOL_CODE not in nav_schools
    assert client.get(f"{PP}/schools/{NAV_SCHOOL_CODE}").status_code == 404
    assert client.get(f"{NAV}/schools/{NAV_SCHOOL_CODE}").status_code == 200

    assert [s["student_id"] for s in client.get(f"{NAV}/students?q={NAV_STUDENT_ID}").json()] == [NAV_STUDENT_ID]
    assert client.get(f"{PP}/students?q={NAV_STUDENT_ID}").json() == []
    assert client.get(f"{NAV}/students?q={ENV_STUDENT_ID}").json() == []


def test_unassigned_review_shows_everywhere_and_resolves_into_one_project(nav, client):
    correlation_id = nav.run_scan(MARKS, roll_number="0000")  # unknown student, no sender history
    scan = _scan(nav, correlation_id)
    assert scan.project_code is None
    review = nav.session.exec(select(ScanReview).where(ScanReview.scan_id == scan.id)).one()

    for base in (PP, NAV):
        item = next(r for r in client.get(f"{base}/reviews").json()["items"] if r["review_id"] == review.review_id)
        assert item["unassigned"] is True
        assert client.get(f"{base}/reviews/{review.review_id}").json()["unassigned"] is True

    body = {"student_id": NAV_STUDENT_ID, "corrected_by": "Asha", "corrections": []}
    wrong = client.post(f"{PP}/reviews/{review.review_id}/resolve", json=body)
    assert wrong.status_code == 400 and "navodaya" in wrong.json()["detail"]

    assert client.post(f"{NAV}/reviews/{review.review_id}/resolve", json=body).status_code == 200
    nav.session.refresh(scan)
    assert scan.project_code == "navodaya"
    assert review.review_id in _review_ids(client, NAV)
    assert review.review_id not in _review_ids(client, PP)
    assert client.get(f"{PP}/reviews/{review.review_id}").status_code == 404


def test_unidentified_scan_inherits_project_from_sender_history(nav, client):
    nav.run_scan(MARKS, roll_number=NAV_STUDENT_ID)  # this phone has now sent a Navodaya scan
    correlation_id = nav.run_scan(MARKS, roll_number="0000")
    scan = _scan(nav, correlation_id)
    assert scan.project_code == "navodaya"
    review = nav.session.exec(select(ScanReview).where(ScanReview.scan_id == scan.id)).one()

    assert review.review_id in _review_ids(client, NAV)
    assert review.review_id not in _review_ids(client, PP)
    assert client.post(f"{PP}/reviews/{review.review_id}/status", json={"status": "approved"}).status_code == 404


def test_import_refuses_ids_and_schools_from_another_project(scan_env):
    session = scan_env.session
    with pytest.raises(ProjectMismatchError, match="paperplus"):
        upsert_student(session, ENV_STUDENT_ID, "Someone Else", NAV_SCHOOL_CODE, project_code="navodaya")
    with pytest.raises(ProjectMismatchError, match="paperplus"):
        upsert_school(session, ENV_SCHOOL_CODE, "Renamed", project_code="navodaya")
    with pytest.raises(ProjectMismatchError, match="Unknown project"):
        upsert_school(session, "ZZZ", "Nowhere", project_code="nosuchproject")
    # Same project is still an idempotent no-op.
    assert upsert_student(session, ENV_STUDENT_ID, "Ignored", ENV_SCHOOL_CODE).student_id == ENV_STUDENT_ID
    session.rollback()
