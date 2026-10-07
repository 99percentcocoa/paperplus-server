"""Weekly engagement metrics behind the admin dashboard's Metrics page: worksheets solved per
week, how many distinct students that represents, and how that compares against the currently
active student roster (avg worksheets/student, % of students sending at least one).

Weeks are ISO weeks (Monday 00:00 UTC to the following Monday), computed with Postgres's 3-arg
date_trunc(..., 'UTC') rather than assumed in Python -- see monitoring.py's _instant() for why a
naive `timestamp` column can't be trusted to already "be" UTC from Python's side (the DB
session's TimeZone setting decides what a bare timestamp means; this dev box is Asia/Calcutta,
docker's Postgres is UTC). The 3-arg form pins week boundaries to UTC regardless of that setting,
verified directly against Postgres before relying on it: the same instant comes back whether the
session TimeZone is UTC or Asia/Calcutta.

Caveat, not fixable without a submission-history table: a week's numbers are computed from
Submission.submitted_at as it stands *right now*. submitted_at is bumped forward every time a
worksheet is rescanned or corrected (so "recent submissions" reflects last activity, not first
attempt -- see PROGRESS.md), so a worksheet counted in an earlier week can move into a newer week
if it's rescanned later. Historical weeks here are not immutable snapshots.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import DateTime, cast, func
from sqlmodel import Session, select

from app.models import Student, Submission

MAX_WEEKS = 26
DEFAULT_WEEKS = 8


@dataclass(frozen=True)
class WeekRow:
    week_start: datetime  # Monday 00:00, UTC
    total_worksheets: int
    distinct_students: int


def week_start_utc(instant: datetime) -> datetime:
    """The Monday 00:00 UTC that instant falls in."""
    instant = instant.astimezone(timezone.utc)
    monday = instant - timedelta(days=instant.weekday())
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)


def build_weekly_series(rows: list[WeekRow], active_students: int, weeks: int, now: datetime) -> list[dict]:
    """rows: one entry per week that actually had submissions (sparse -- a quiet week is simply
    absent). Fills every week in the requested range oldest-first, including empty ones, and rates
    both headline metrics against active_students -- the *current* active roster size, since
    historical roster size isn't tracked.
    """
    by_week = {row.week_start: row for row in rows}
    current_week_start = week_start_utc(now)
    series = []
    for i in range(weeks - 1, -1, -1):
        start = current_week_start - timedelta(weeks=i)
        row = by_week.get(start)
        total = row.total_worksheets if row else 0
        distinct = row.distinct_students if row else 0
        series.append(
            {
                "week_start": start.date().isoformat(),
                "week_end": (start + timedelta(days=7)).date().isoformat(),
                "is_current": start == current_week_start,
                "total_worksheets": total,
                "distinct_students": distinct,
                "avg_worksheets_per_student": round(total / active_students, 2) if active_students else None,
                "pct_students_active": round(distinct / active_students * 100, 1) if active_students else None,
            }
        )
    return series


def _instant(column_expr):
    """See monitoring.py's _instant() -- same reasoning, duplicated rather than imported to keep
    the two services independent (metrics has no other dependency on monitoring)."""
    return cast(column_expr, DateTime(timezone=True))


def collect(session: Session, weeks: int = DEFAULT_WEEKS, project_code: str | None = None) -> dict:
    """project_code None = every project (used by tests of the bucketing logic itself)."""
    weeks = max(1, min(weeks, MAX_WEEKS))
    now = datetime.now(timezone.utc)
    since = week_start_utc(now) - timedelta(weeks=weeks - 1)

    in_project = [Student.project_code == project_code] if project_code is not None else []
    active_students = session.exec(
        select(func.count()).select_from(Student).where(Student.is_active.is_(True), *in_project)
    ).one()

    week_col = func.date_trunc("week", _instant(Submission.submitted_at), "UTC")
    query = (
        select(week_col, func.count(), func.count(Submission.student_id.distinct()))
        .join(Student, Student.student_id == Submission.student_id)
        .where(_instant(Submission.submitted_at) >= since, *in_project)
        .group_by(week_col)
    )
    rows = [
        WeekRow(week_start=week_start.astimezone(timezone.utc), total_worksheets=total, distinct_students=distinct)
        for week_start, total, distinct in session.exec(query).all()
    ]

    return {
        "active_students": active_students,
        "weeks": build_weekly_series(rows, active_students, weeks, now),
    }
