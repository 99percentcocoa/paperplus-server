"""JSON API behind the admin/facilitator dashboard (static UI in app/static/admin, served at
/admin/{project_code}/). Intentionally unauthenticated for now, per the deployment decision -- the
URL isn't shared publicly. Write endpoints (corrections, review resolution) go through
app.services.corrections so grading rules live in one place.

Everything except the project list and system monitoring is scoped to one project under
/api/admin/projects/{project_code}/...: a submission/school/student outside it is a 404. Failed
scans whose student (and so project) couldn't be determined are "unassigned" and show on every
project's list until someone resolves them to a student.
"""

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, or_
from sqlalchemy.sql import ColumnElement
from sqlmodel import Session, select
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.db.session import get_session
from app.domain.grading import resolve_answer_key
from app.models import Project, Question, QuestionOption, School, Scan, ScanReview, Student, Submission, Worksheet
from app.routes.files import relative_artifact_url
from app.services import corrections as corrections_service
from app.services import metrics as metrics_service
from app.services import monitoring as monitoring_service
from app.services.corrections import CorrectionError, NotFoundError
from app.services.monitoring import OPEN_REVIEW_STATUSES
from app.services.communication import CapturingCommunicationClient
from app.services.submission_service import handle_incoming_image
from app.services.vision_client import HTTPVisionClient, VisionClient

router = APIRouter(prefix="/api/admin", tags=["admin"])
project_router = APIRouter(prefix="/api/admin/projects/{project_code}", tags=["admin"])


def get_vision_client() -> VisionClient:
    return HTTPVisionClient()


def get_project(project_code: str, session: Session = Depends(get_session)) -> Project:
    project = session.get(Project, project_code)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return project


def _review_project() -> ColumnElement:
    """A review's project: its scan's (set once the student is known, or guessed from the
    sender), else its student's. NULL = unassigned. Needs Scan and Student outer-joined."""
    return func.coalesce(Scan.project_code, Student.project_code)


def _reviews_in_project(query, project_code: str):
    project = _review_project()
    return (
        query.outerjoin(Scan, Scan.id == ScanReview.scan_id)
        .outerjoin(Student, Student.student_id == ScanReview.student_id)
        .where(or_(project == project_code, project.is_(None)))
    )


def _get_submission(session: Session, submission_id: int, project_code: str) -> Submission:
    row = session.exec(
        select(Submission)
        .join(Student, Student.student_id == Submission.student_id)
        .where(Submission.submission_id == submission_id, Student.project_code == project_code)
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="submission not found")
    return row


def _get_review(session: Session, review_id: int, project_code: str) -> ScanReview:
    row = session.exec(
        _reviews_in_project(select(ScanReview), project_code).where(ScanReview.review_id == review_id)
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="review not found")
    return row


@project_router.get("")
def project_info(project: Project = Depends(get_project)) -> dict:
    return {"project_code": project.project_code, "project_name": project.project_name}


class AnswerCorrection(BaseModel):
    question_index: int
    selected_option: str | None = None  # None / "" = unanswered


class CorrectSubmissionRequest(BaseModel):
    corrections: list[AnswerCorrection]
    corrected_by: str = Field(min_length=1, max_length=100)
    question_paper_code: str | None = None


class ResolveReviewRequest(BaseModel):
    student_id: str = Field(min_length=1)
    corrections: list[AnswerCorrection] = []
    corrected_by: str = Field(min_length=1, max_length=100)
    worksheet_id: int | None = None
    question_paper_code: str | None = None


class ReviewStatusRequest(BaseModel):
    status: str
    corrected_by: str | None = None


class RetryScanRequest(BaseModel):
    worksheet_id: int
    roll_number: str | None = None
    question_paper_code: str | None = None


MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _image_extension(data: bytes) -> str | None:
    """From the file's magic bytes rather than Content-Type, which a plain `curl --data-binary`
    sets to application/x-www-form-urlencoded."""
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _to_dict(corrections: list[AnswerCorrection]) -> dict[int, str | None]:
    return {c.question_index: c.selected_option for c in corrections}


def _artifact_url(kind: str, stored_path: str | None) -> str | None:
    """Same-origin URL plus a mtime version so a regenerated checked image isn't served stale."""
    url = relative_artifact_url(kind, stored_path)
    if url is None:
        return None
    try:
        version = int((Path(settings.storage_root) / kind / Path(stored_path).name).stat().st_mtime)
    except OSError:
        return None  # recorded but the file is gone (pruned, or another host's volume)
    return f"{url}?v={version}"


def _scan_images(scan: Scan) -> dict:
    return {
        "upload": _artifact_url("uploads", scan.upload_path),
        "checked": _artifact_url("checked", scan.checked_image_path),
    }


def _iso(value: datetime | None) -> str | None:
    """DB timestamps round-trip as naive datetimes (the `timestamp` columns have no zone, and
    utcnow() always writes real UTC) -- tag them as UTC explicitly before serializing, otherwise
    the browser's Date parser treats the zone-less string as already being in the viewer's own
    local time and renders it hours off (e.g. IST is UTC+5:30)."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _question_details(session: Session, worksheet_id: int) -> dict[int, dict]:
    questions = session.exec(select(Question).where(Question.worksheet_id == worksheet_id)).all()
    options: dict[int, list[str]] = {q.question_id: [] for q in questions}
    if options:
        rows = session.exec(
            select(QuestionOption).where(QuestionOption.question_id.in_(list(options))).order_by(QuestionOption.id)
        ).all()
        for option in rows:
            options[option.question_id].append(option.option_label)
    details = {}
    for q in questions:
        if q.index is None:
            continue
        payload = q.question_json or {}
        details[q.index] = {
            "question_text": payload.get("question_text"),
            "options": payload.get("options"),
            "labels": options[q.question_id] or ["A", "B", "C", "D"],
            "skill_code": q.skill_code,
        }
    return details


@project_router.get("/summary")
def summary(project: Project = Depends(get_project), session: Session = Depends(get_session)) -> dict:
    code = project.project_code
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    count = lambda stmt: session.exec(stmt).one()  # noqa: E731
    submissions = (
        select(func.count()).select_from(Submission)
        .join(Student, Student.student_id == Submission.student_id).where(Student.project_code == code)
    )
    return {
        "schools": count(select(func.count()).select_from(School).where(School.project_code == code)),
        "students": count(
            select(func.count()).select_from(Student).where(Student.is_active.is_(True), Student.project_code == code)
        ),
        "submissions": count(submissions),
        "submissions_24h": count(submissions.where(Submission.submitted_at >= since)),
        "open_reviews": count(
            _reviews_in_project(select(func.count()).select_from(ScanReview), code)
            .where(ScanReview.status.in_(OPEN_REVIEW_STATUSES))
        ),
    }


@router.get("/monitoring")
def monitoring(session: Session = Depends(get_session)) -> dict:
    """Live health/throughput/backlog/disk snapshot plus any active alerts (see app.services.monitoring)."""
    return monitoring_service.collect(session)


@project_router.get("/metrics/weekly")
def weekly_metrics(
    weeks: int = Query(metrics_service.DEFAULT_WEEKS, ge=1, le=metrics_service.MAX_WEEKS),
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict:
    return metrics_service.collect(session, weeks, project.project_code)


@project_router.get("/submissions")
def list_submissions(
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
    student_id: str | None = None,
    worksheet_id: int | None = None,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict:
    in_project = Student.project_code == project.project_code
    query = select(Submission, Student).join(Student, Student.student_id == Submission.student_id).where(in_project)
    total_query = (
        select(func.count()).select_from(Submission)
        .join(Student, Student.student_id == Submission.student_id).where(in_project)
    )
    if student_id:
        query = query.where(Submission.student_id == student_id)
        total_query = total_query.where(Submission.student_id == student_id)
    if worksheet_id is not None:
        query = query.where(Submission.worksheet_id == worksheet_id)
        total_query = total_query.where(Submission.worksheet_id == worksheet_id)
    rows = session.exec(query.order_by(Submission.submitted_at.desc()).offset(offset).limit(limit)).all()
    return {
        "total": session.exec(total_query).one(),
        "items": [
            {
                "submission_id": s.submission_id,
                "student_id": s.student_id,
                "student_name": st.student_name,
                "school_code": st.student_school_code,
                "worksheet_id": s.worksheet_id,
                "score": s.score,
                "total_questions": len(s.answers_json or []),
                "state": s.state,
                "submitted_at": _iso(s.submitted_at),
                "has_image": bool(s.checked_image_path),
            }
            for s, st in rows
        ],
    }


@project_router.get("/submissions/{submission_id}")
def submission_detail(
    submission_id: int, project: Project = Depends(get_project), session: Session = Depends(get_session)
) -> dict:
    submission = _get_submission(session, submission_id, project.project_code)
    student = session.get(Student, submission.student_id)
    worksheet = session.get(Worksheet, submission.worksheet_id)
    scans = session.exec(
        select(Scan).where(Scan.submission_id == submission_id).order_by(Scan.created_at.desc())
    ).all()
    code = next((s.question_paper_code for s in scans if s.question_paper_code), None)
    answer_key = resolve_answer_key(session, submission.worksheet_id, code)
    details = _question_details(session, submission.worksheet_id)
    answers = {a["question_index"]: a for a in (submission.answers_json or [])}

    questions = []
    for index in sorted(set(details) | set(answers)):
        answer = answers.get(index)
        detail = details.get(index, {})
        questions.append(
            {
                "question_index": index,
                "selected_option": (answer or {}).get("selected_option") or (answer or {}).get("answer") or "",
                "is_correct": bool((answer or {}).get("is_correct")),
                "scanned": answer is not None,
                "correct_option": answer_key.get(index),
                "question_text": detail.get("question_text"),
                "options": detail.get("options"),
                "labels": detail.get("labels", ["A", "B", "C", "D"]),
            }
        )

    reviews = session.exec(
        select(ScanReview).where(ScanReview.submission_id == submission_id).order_by(ScanReview.created_at.desc())
    ).all()
    return {
        "submission_id": submission.submission_id,
        "student": {
            "student_id": submission.student_id,
            "student_name": student.student_name if student else None,
            "school_code": student.student_school_code if student else None,
        },
        "worksheet": {
            "worksheet_id": submission.worksheet_id,
            "title": worksheet.title if worksheet else None,
            "category": worksheet.worksheet_category if worksheet else None,
            "level": worksheet.worksheet_level if worksheet else None,
            "page_count": worksheet.page_count if worksheet else None,
        },
        "score": submission.score,
        "total_questions": len(questions),
        "state": submission.state,
        "submitted_at": _iso(submission.submitted_at),
        "from_number": submission.from_number,
        "question_paper_code": code,
        "questions": questions,
        "scans": [
            {
                "scan_id": s.id,
                "page_no": s.page_no,
                "created_at": _iso(s.created_at),
                "question_paper_code": s.question_paper_code,
                "images": _scan_images(s),
                "question_indices": sorted(
                    m["question_index"] for m in (s.vision_result or {}).get("question_marks", [])
                ),
            }
            for s in scans
        ],
        "history": [
            {
                "review_id": r.review_id,
                "status": r.status,
                "corrected_by": r.corrected_by,
                "corrected_at": _iso(r.corrected_at),
                "original_score": r.original_score,
                "corrected_score": r.corrected_score,
            }
            for r in reviews
        ],
    }


@project_router.patch("/submissions/{submission_id}/answers")
def correct_submission_answers(
    submission_id: int,
    body: CorrectSubmissionRequest,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict:
    _get_submission(session, submission_id, project.project_code)
    try:
        submission = corrections_service.correct_submission(
            session, submission_id, _to_dict(body.corrections), body.corrected_by.strip(), body.question_paper_code
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CorrectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "submission_id": submission.submission_id,
        "score": submission.score,
        "total_questions": len(submission.answers_json or []),
    }


@project_router.post("/scans")
async def upload_scan(
    request: Request,
    roll_number: str | None = Query(None, description="Student ID; replaces reading it from the photo"),
    question_paper_code: str | None = Query(None, description="OMR set code A-F; replaces reading it from the photo"),
    from_number: str | None = Query(None, description="Optional sender to record on the scan (no WhatsApp reply is sent)"),
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
    vision_client: VisionClient = Depends(get_vision_client),
) -> dict:
    """Grade a photo sent straight to the API instead of over WhatsApp, e.g.
    `curl --data-binary @scan.jpg '.../scans?roll_number=0151&question_paper_code=D'`.
    Same pipeline as the webhook (failures land on this project's failed-scan list), except the
    handwritten fields can be supplied, the student must be in this project, and the WhatsApp
    replies come back in the response instead of being sent."""
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body -- send the photo with curl --data-binary @photo.jpg.")
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Photo larger than 20 MB.")
    extension = _image_extension(body)
    if extension is None:
        raise HTTPException(
            status_code=415,
            detail="Body is not a JPEG/PNG/WebP image -- send the file itself with --data-binary @photo.jpg (not -d or -F).",
        )

    # Grading blocks (DB + a vision-service call of up to a minute) -- keep it off the event loop.
    return await run_in_threadpool(
        _grade_upload, session, vision_client, project, body, extension, roll_number, question_paper_code, from_number,
    )


def _grade_upload(
    session: Session,
    vision_client: VisionClient,
    project: Project,
    body: bytes,
    extension: str,
    roll_number: str | None,
    question_paper_code: str | None,
    from_number: str | None,
) -> dict:
    roll_number = (roll_number or "").strip() or None
    question_paper_code = (question_paper_code or "").strip().upper() or None
    if roll_number:
        student = session.get(Student, roll_number)
        if student is None or student.project_code != project.project_code:
            raise HTTPException(status_code=400, detail=f"No student '{roll_number}' in project '{project.project_code}'.")

    correlation_id = str(uuid.uuid4())
    upload_dir = Path(settings.storage_root) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    image_path = upload_dir / f"{correlation_id}.{extension}"
    image_path.write_bytes(body)

    replies = CapturingCommunicationClient()
    handle_incoming_image(
        session, vision_client, replies, (from_number or "").strip() or None, str(image_path), correlation_id,
        roll_number=roll_number, question_paper_code=question_paper_code, project_code=project.project_code,
    )

    scan = session.exec(select(Scan).where(Scan.correlation_id == correlation_id)).first()
    review = session.exec(select(ScanReview).where(ScanReview.scan_id == scan.id)).first() if scan else None
    submission = session.get(Submission, scan.submission_id) if scan and scan.submission_id else None
    return {
        "correlation_id": correlation_id,
        "outcome": scan.outcome if scan else "failed",
        "scan_id": scan.id if scan else None,
        "roll_number": scan.roll_number if scan else None,
        "question_paper_code": scan.question_paper_code if scan else None,
        "worksheet_id": scan.worksheet_id if scan else None,
        "submission_id": submission.submission_id if submission else None,
        "score": submission.score if submission else None,
        "total_questions": len(submission.answers_json or []) if submission else None,
        "review_id": review.review_id if review else None,
        "error_reason": review.error_reason if review else None,
        "replies": replies.messages,
        "checked_image_url": _artifact_url("checked", scan.checked_image_path) if scan else None,
    }


@project_router.get("/reviews")
def list_reviews(
    status: str = Query("open", description="open (failed+needs_review), all, or a specific status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict:
    query = _reviews_in_project(select(ScanReview, _review_project()), project.project_code)
    total_query = _reviews_in_project(select(func.count()).select_from(ScanReview), project.project_code)
    if status == "open":
        query = query.where(ScanReview.status.in_(OPEN_REVIEW_STATUSES))
        total_query = total_query.where(ScanReview.status.in_(OPEN_REVIEW_STATUSES))
    elif status != "all":
        query = query.where(ScanReview.status == status)
        total_query = total_query.where(ScanReview.status == status)
    rows = session.exec(query.order_by(ScanReview.created_at.desc()).offset(offset).limit(limit)).all()
    reviews = [r for r, _ in rows]
    unassigned = {r.review_id for r, review_project in rows if review_project is None}
    scans = {
        s.id: s
        for s in session.exec(select(Scan).where(Scan.id.in_([r.scan_id for r in reviews if r.scan_id]))).all()
    } if reviews else {}
    return {
        "total": session.exec(total_query).one(),
        "items": [
            {
                "review_id": r.review_id,
                "status": r.status,
                "detected_roll_number": r.detected_roll_number,
                "student_id": r.student_id,
                "worksheet_id": (worksheet_id := r.worksheet_id or (scans[r.scan_id].worksheet_id if r.scan_id in scans else None)),
                "error_reason": r.error_reason,
                "created_at": _iso(r.created_at),
                "submission_id": r.submission_id,
                "from_number": scans[r.scan_id].from_number if r.scan_id in scans else None,
                "has_image": r.scan_id in scans,
                "unassigned": r.review_id in unassigned,
                # Resolvable straight from this list either with real marks, or -- once a worksheet
                # is known (e.g. after a retry, or an earlier scan of the same worksheet decoded it
                # fine) -- by manually grading from the photo; see review_detail for the full logic.
                "resolvable": bool(
                    r.scan_id in scans and (
                        (scans[r.scan_id].vision_result or {}).get("question_marks") or worksheet_id is not None
                    )
                ),
            }
            for r in reviews
        ],
    }


@project_router.get("/reviews/{review_id}")
def review_detail(
    review_id: int,
    worksheet_id: int | None = None,
    question_paper_code: str | None = None,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict:
    """worksheet_id / question_paper_code optionally override what the scan decoded, so the UI can
    preview the answer key for a different worksheet/code before saving."""
    review = _get_review(session, review_id, project.project_code)
    scan = session.get(Scan, review.scan_id) if review.scan_id else None
    vision = (scan.vision_result if scan else None) or {}
    marks = vision.get("question_marks", [])

    target_worksheet_id = worksheet_id or (scan.worksheet_id if scan else None) or review.worksheet_id
    worksheet = session.get(Worksheet, target_worksheet_id) if target_worksheet_id is not None else None
    code = question_paper_code or (scan.question_paper_code if scan else None)
    answer_key = resolve_answer_key(session, worksheet.worksheet_id, code) if worksheet else {}
    details = _question_details(session, worksheet.worksheet_id) if worksheet else {}

    if marks:
        questions = [
            {
                "question_index": m["question_index"],
                "detected_option": m.get("marked_option") or "",
                "confidence": m.get("confidence"),
                "correct_option": answer_key.get(m["question_index"]),
                "question_text": details.get(m["question_index"], {}).get("question_text"),
                "options": details.get(m["question_index"], {}).get("options"),
                "labels": details.get(m["question_index"], {}).get("labels", ["A", "B", "C", "D"]),
            }
            for m in marks
        ]
    elif worksheet is not None:
        # No vision result (tags weren't detected) -- an admin who can see the photo can still
        # grade it by hand once they've told us which worksheet it is, so offer every question on
        # that worksheet blank rather than nothing at all.
        questions = [
            {
                "question_index": index,
                "detected_option": "",
                "confidence": None,
                "correct_option": answer_key.get(index),
                "question_text": detail.get("question_text"),
                "options": detail.get("options"),
                "labels": detail.get("labels", ["A", "B", "C", "D"]),
            }
            for index, detail in sorted(details.items())
        ]
    else:
        questions = []

    return {
        "review_id": review.review_id,
        "status": review.status,
        "error_reason": review.error_reason,
        "created_at": _iso(review.created_at),
        "corrected_by": review.corrected_by,
        "corrected_at": _iso(review.corrected_at),
        "submission_id": review.submission_id,
        "detected_roll_number": review.detected_roll_number,
        "student_id": review.student_id,
        "unassigned": session.exec(
            select(_review_project())
            .select_from(ScanReview)
            .outerjoin(Scan, Scan.id == ScanReview.scan_id)
            .outerjoin(Student, Student.student_id == ScanReview.student_id)
            .where(ScanReview.review_id == review.review_id)
        ).one() is None,
        "scan": None if scan is None else {
            "scan_id": scan.id,
            "from_number": scan.from_number,
            "page_no": scan.page_no,
            "template_name": scan.template_name,
            "worksheet_id": scan.worksheet_id,
            "question_paper_code": scan.question_paper_code,
            "project_code": scan.project_code,
            "images": _scan_images(scan),
        },
        "worksheet": None if worksheet is None else {
            "worksheet_id": worksheet.worksheet_id,
            "title": worksheet.title,
            "category": worksheet.worksheet_category,
        },
        "worksheet_missing": worksheet is None,
        "answer_key_missing": worksheet is not None and not answer_key,
        "manual_entry": not marks and worksheet is not None,
        "resolvable": scan is not None and bool(questions) and review.status in OPEN_REVIEW_STATUSES,
        "questions": questions,
    }


@project_router.post("/reviews/{review_id}/retry")
def retry_review_scan(
    review_id: int,
    body: RetryScanRequest,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
    vision_client: VisionClient = Depends(get_vision_client),
) -> dict:
    """For reviews with no usable scan result yet (tags not detected, or roll number invalid --
    both currently leave scan.vision_result empty): re-sends the original photo to vision-service
    with the admin-supplied worksheet_id as a hint, so question_marks can be recovered and the
    review becomes resolvable through the normal /resolve flow below."""
    _get_review(session, review_id, project.project_code)
    try:
        corrections_service.retry_scan(
            session, vision_client, review_id, body.worksheet_id,
            roll_number=(body.roll_number or "").strip() or None,
            question_paper_code=(body.question_paper_code or "").strip().upper() or None,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CorrectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return review_detail(review_id, project=project, session=session)


@project_router.post("/reviews/{review_id}/resolve")
def resolve_review(
    review_id: int,
    body: ResolveReviewRequest,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict:
    _get_review(session, review_id, project.project_code)
    try:
        submission = corrections_service.resolve_review(
            session,
            review_id,
            student_id=body.student_id.strip(),
            corrections=_to_dict(body.corrections),
            corrected_by=body.corrected_by.strip(),
            worksheet_id=body.worksheet_id,
            question_paper_code=body.question_paper_code,
            project_code=project.project_code,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CorrectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "submission_id": submission.submission_id,
        "score": submission.score,
        "total_questions": len(submission.answers_json or []),
    }


@project_router.post("/reviews/{review_id}/status")
def set_review_status(
    review_id: int,
    body: ReviewStatusRequest,
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> dict:
    _get_review(session, review_id, project.project_code)
    try:
        review = corrections_service.set_review_status(session, review_id, body.status, body.corrected_by)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CorrectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"review_id": review.review_id, "status": review.status}


@project_router.get("/students")
def search_students(
    q: str = Query("", description="matches student_id prefix or name substring"),
    limit: int = Query(20, ge=1, le=100),
    project: Project = Depends(get_project),
    session: Session = Depends(get_session),
) -> list[dict]:
    query = select(Student).where(Student.project_code == project.project_code)
    term = q.strip()
    if term:
        query = query.where(or_(Student.student_id.like(f"{term}%"), Student.student_name.ilike(f"%{term}%")))
    rows = session.exec(query.order_by(Student.student_id).limit(limit)).all()
    return [
        {"student_id": s.student_id, "student_name": s.student_name, "school_code": s.student_school_code}
        for s in rows
    ]


@project_router.get("/schools")
def list_schools(project: Project = Depends(get_project), session: Session = Depends(get_session)) -> list[dict]:
    counts = dict(
        session.exec(
            select(Student.student_school_code, func.count())
            .where(Student.is_active.is_(True), Student.project_code == project.project_code)
            .group_by(Student.student_school_code)
        ).all()
    )
    schools = session.exec(
        select(School).where(School.project_code == project.project_code).order_by(School.school_name)
    ).all()
    return [
        {"school_code": s.school_code, "school_name": s.school_name, "student_count": counts.get(s.school_code, 0)}
        for s in schools
    ]


@project_router.get("/schools/{school_code}")
def school_detail(
    school_code: str, project: Project = Depends(get_project), session: Session = Depends(get_session)
) -> dict:
    """Every active student at a school, with their last submission and last 3 worksheets
    (level + marks) -- so a facilitator visiting a school can see at a glance who has and hasn't
    submitted. Students who have never submitted sort first, then oldest-last-submission next,
    so whoever needs following up with is at the top rather than buried alphabetically.
    """
    school = session.get(School, school_code)
    if school is None or school.project_code != project.project_code:
        raise HTTPException(status_code=404, detail="school not found")

    students = session.exec(
        select(Student)
        .where(
            Student.student_school_code == school_code,
            Student.is_active.is_(True),
            Student.project_code == project.project_code,
        )
        .order_by(Student.student_name)
    ).all()
    student_ids = [s.student_id for s in students]

    submissions_by_student: dict[str, list] = {}
    if student_ids:
        rows = session.exec(
            select(Submission, Worksheet.worksheet_level)
            .join(Worksheet, Worksheet.worksheet_id == Submission.worksheet_id)
            .where(Submission.student_id.in_(student_ids))
            .order_by(Submission.submitted_at.desc())
        ).all()
        for submission, level in rows:
            submissions_by_student.setdefault(submission.student_id, []).append((submission, level))

    def student_row(student: Student) -> dict:
        recent = submissions_by_student.get(student.student_id, [])
        last_submitted_at = recent[0][0].submitted_at if recent else None
        return {
            "student_id": student.student_id,
            "student_name": student.student_name,
            "current_level": student.current_level,
            "last_submitted_at": _iso(last_submitted_at),
            # submitted_at is a naive DateTime column (no tzinfo -- a known pre-existing gap,
            # see PROGRESS.md) -- datetime.min here must stay naive too, or comparing it against
            # a real (naive) submitted_at raises "can't compare offset-naive and offset-aware".
            "_sort_key": (last_submitted_at is not None, last_submitted_at or datetime.min),
            "recent_worksheets": [
                {
                    "submission_id": submission.submission_id,
                    "worksheet_id": submission.worksheet_id,
                    "level": level,
                    "score": submission.score,
                    "total_questions": len(submission.answers_json or []),
                }
                for submission, level in recent[:3]
            ],
        }

    rows = sorted((student_row(s) for s in students), key=lambda r: (r["_sort_key"], r["student_name"]))
    for row in rows:
        del row["_sort_key"]

    return {
        "school": {"school_code": school.school_code, "school_name": school.school_name},
        "students": rows,
    }
