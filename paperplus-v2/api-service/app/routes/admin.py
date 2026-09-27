"""JSON API behind the admin/facilitator dashboard (static UI in app/static/admin, mounted at
/admin). Intentionally unauthenticated for now, per the deployment decision -- the URL isn't
shared publicly. Write endpoints (corrections, review resolution) go through
app.services.corrections so grading rules live in one place.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, or_
from sqlmodel import Session, select

from app.core.config import settings
from app.db.session import get_session
from app.domain.grading import resolve_answer_key
from app.models import Question, QuestionOption, School, Scan, ScanReview, Student, Submission, Worksheet
from app.routes.files import relative_artifact_url
from app.services import corrections as corrections_service
from app.services import monitoring as monitoring_service
from app.services.corrections import CorrectionError, NotFoundError
from app.services.monitoring import OPEN_REVIEW_STATUSES
from app.services.vision_client import HTTPVisionClient, VisionClient

router = APIRouter(prefix="/api/admin", tags=["admin"])


def get_vision_client() -> VisionClient:
    return HTTPVisionClient()


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
    return value.isoformat() if value else None


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


@router.get("/summary")
def summary(session: Session = Depends(get_session)) -> dict:
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    count = lambda stmt: session.exec(stmt).one()  # noqa: E731
    return {
        "schools": count(select(func.count()).select_from(School)),
        "students": count(select(func.count()).select_from(Student).where(Student.is_active.is_(True))),
        "submissions": count(select(func.count()).select_from(Submission)),
        "submissions_24h": count(select(func.count()).select_from(Submission).where(Submission.submitted_at >= since)),
        "open_reviews": count(
            select(func.count()).select_from(ScanReview).where(ScanReview.status.in_(OPEN_REVIEW_STATUSES))
        ),
    }


@router.get("/monitoring")
def monitoring(session: Session = Depends(get_session)) -> dict:
    """Live health/throughput/backlog/disk snapshot plus any active alerts (see app.services.monitoring)."""
    return monitoring_service.collect(session)


@router.get("/submissions")
def list_submissions(
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
    student_id: str | None = None,
    worksheet_id: int | None = None,
    session: Session = Depends(get_session),
) -> dict:
    query = select(Submission, Student).join(Student, Student.student_id == Submission.student_id)
    total_query = select(func.count()).select_from(Submission)
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


@router.get("/submissions/{submission_id}")
def submission_detail(submission_id: int, session: Session = Depends(get_session)) -> dict:
    submission = session.get(Submission, submission_id)
    if submission is None:
        raise HTTPException(status_code=404, detail="submission not found")
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


@router.patch("/submissions/{submission_id}/answers")
def correct_submission_answers(
    submission_id: int, body: CorrectSubmissionRequest, session: Session = Depends(get_session)
) -> dict:
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


@router.get("/reviews")
def list_reviews(
    status: str = Query("open", description="open (failed+needs_review), all, or a specific status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
) -> dict:
    query = select(ScanReview)
    total_query = select(func.count()).select_from(ScanReview)
    if status == "open":
        query = query.where(ScanReview.status.in_(OPEN_REVIEW_STATUSES))
        total_query = total_query.where(ScanReview.status.in_(OPEN_REVIEW_STATUSES))
    elif status != "all":
        query = query.where(ScanReview.status == status)
        total_query = total_query.where(ScanReview.status == status)
    reviews = session.exec(query.order_by(ScanReview.created_at.desc()).offset(offset).limit(limit)).all()
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


@router.get("/reviews/{review_id}")
def review_detail(
    review_id: int,
    worksheet_id: int | None = None,
    question_paper_code: str | None = None,
    session: Session = Depends(get_session),
) -> dict:
    """worksheet_id / question_paper_code optionally override what the scan decoded, so the UI can
    preview the answer key for a different worksheet/code before saving."""
    review = session.get(ScanReview, review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="review not found")
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
        "scan": None if scan is None else {
            "scan_id": scan.id,
            "from_number": scan.from_number,
            "page_no": scan.page_no,
            "template_name": scan.template_name,
            "worksheet_id": scan.worksheet_id,
            "question_paper_code": scan.question_paper_code,
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


@router.post("/reviews/{review_id}/retry")
def retry_review_scan(
    review_id: int,
    body: RetryScanRequest,
    session: Session = Depends(get_session),
    vision_client: VisionClient = Depends(get_vision_client),
) -> dict:
    """For reviews with no usable scan result yet (tags not detected, or roll number invalid --
    both currently leave scan.vision_result empty): re-sends the original photo to vision-service
    with the admin-supplied worksheet_id as a hint, so question_marks can be recovered and the
    review becomes resolvable through the normal /resolve flow below."""
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
    return review_detail(review_id, session=session)


@router.post("/reviews/{review_id}/resolve")
def resolve_review(review_id: int, body: ResolveReviewRequest, session: Session = Depends(get_session)) -> dict:
    try:
        submission = corrections_service.resolve_review(
            session,
            review_id,
            student_id=body.student_id.strip(),
            corrections=_to_dict(body.corrections),
            corrected_by=body.corrected_by.strip(),
            worksheet_id=body.worksheet_id,
            question_paper_code=body.question_paper_code,
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


@router.post("/reviews/{review_id}/status")
def set_review_status(review_id: int, body: ReviewStatusRequest, session: Session = Depends(get_session)) -> dict:
    try:
        review = corrections_service.set_review_status(session, review_id, body.status, body.corrected_by)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CorrectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"review_id": review.review_id, "status": review.status}


@router.get("/students")
def search_students(
    q: str = Query("", description="matches student_id prefix or name substring"),
    limit: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> list[dict]:
    query = select(Student)
    term = q.strip()
    if term:
        query = query.where(or_(Student.student_id.like(f"{term}%"), Student.student_name.ilike(f"%{term}%")))
    rows = session.exec(query.order_by(Student.student_id).limit(limit)).all()
    return [
        {"student_id": s.student_id, "student_name": s.student_name, "school_code": s.student_school_code}
        for s in rows
    ]


@router.get("/schools")
def list_schools(session: Session = Depends(get_session)) -> list[dict]:
    counts = dict(
        session.exec(
            select(Student.student_school_code, func.count()).where(Student.is_active.is_(True)).group_by(Student.student_school_code)
        ).all()
    )
    schools = session.exec(select(School).order_by(School.school_name)).all()
    return [
        {"school_code": s.school_code, "school_name": s.school_name, "student_count": counts.get(s.school_code, 0)}
        for s in schools
    ]


@router.get("/schools/{school_code}")
def school_detail(school_code: str, session: Session = Depends(get_session)) -> dict:
    """Every active student at a school, with their last submission and last 3 worksheets
    (level + marks) -- so a facilitator visiting a school can see at a glance who has and hasn't
    submitted. Students who have never submitted sort first, then oldest-last-submission next,
    so whoever needs following up with is at the top rather than buried alphabetically.
    """
    school = session.get(School, school_code)
    if school is None:
        raise HTTPException(status_code=404, detail="school not found")

    students = session.exec(
        select(Student)
        .where(Student.student_school_code == school_code, Student.is_active.is_(True))
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
