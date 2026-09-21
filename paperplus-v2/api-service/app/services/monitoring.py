"""Operational health snapshot + alert rules behind the admin dashboard's status panel.

collect() gathers the numbers (dependency health, scan throughput, review backlog, disk) and
evaluate_alerts() turns them into warnings/criticals against the thresholds in app.core.config.
The rules are pure functions of the snapshot so they can be unit-tested without a database.

Everything is derived on demand from the DB and live probes, not from in-process counters, so
it survives restarts and reflects real state. This is dashboard-level alerting (banners + a
status dot); nothing here pushes a notification -- point an external uptime monitor at
/api/admin/monitoring (or /health) if you need paging.

Error strings are deliberately generic (no URLs, DSNs or paths): the admin API is unauthenticated.
"""

import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from sqlalchemy import DateTime, cast, func, or_
from sqlmodel import Session, select

from app.core.config import settings
from app.models import Scan, ScanReview
from app.models.submission import ScanOutcome, ScanReviewStatus

OPEN_REVIEW_STATUSES = (ScanReviewStatus.FAILED.value, ScanReviewStatus.NEEDS_REVIEW.value)
VISION_HEALTH_TIMEOUT_SECONDS = 3.0

_SEVERITY_ORDER = {"critical": 0, "warning": 1}


def _describe(exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "timed out"
    if isinstance(exc, httpx.ConnectError):
        return "connection failed"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


def _ms_since(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


def _instant(column_expr):
    """Cast a naive `timestamp` column expression to an aware instant. The columns hold whatever the
    database session's TimeZone made of the (UTC) value when it was written -- UTC on the docker
    Postgres, local time on some dev machines -- so guessing "it's UTC" in Python is wrong on the
    latter. Letting Postgres do the cast applies the same interpretation it used on the way in."""
    return cast(column_expr, DateTime(timezone=True))


def _utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def check_database(session: Session) -> dict:
    start = time.perf_counter()
    try:
        session.exec(select(1)).one()
    except Exception as exc:  # noqa: BLE001 - a health probe must report, never raise
        return {"ok": False, "error": _describe(exc)}
    return {"ok": True, "latency_ms": _ms_since(start)}


def check_vision_service(timeout: float = VISION_HEALTH_TIMEOUT_SECONDS) -> dict:
    """vision-service's /health is unauthenticated and stays up while the models are loaded, so a
    failure here means scans will fail too."""
    url = settings.vision_service_url.rstrip("/") + "/health"
    start = time.perf_counter()
    try:
        httpx.get(url, timeout=timeout).raise_for_status()
    except httpx.HTTPError as exc:
        return {"ok": False, "error": _describe(exc)}
    return {"ok": True, "latency_ms": _ms_since(start)}


def storage_stats() -> dict:
    """Free space on the volume holding uploads/dewarped/checked images -- the thing that quietly
    fills up, then makes every scan fail at the first image write."""
    probe = Path(settings.storage_root)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        return {"ok": False, "error": type(exc).__name__}
    return {
        "ok": True,
        "total_gb": round(usage.total / 1e9, 1),
        "free_gb": round(usage.free / 1e9, 1),
        "used_pct": round(usage.used / usage.total * 100, 1),
    }


def scan_activity(session: Session, now: datetime) -> dict:
    since_1h = now - timedelta(hours=1)
    since_24h = now - timedelta(hours=24)

    def count(*conditions) -> int:
        return session.exec(select(func.count()).select_from(Scan).where(*conditions)).one()

    scans_24h = count(Scan.created_at >= since_24h)
    graded_24h = count(Scan.created_at >= since_24h, Scan.outcome == ScanOutcome.GRADED.value)
    failed_24h = count(Scan.created_at >= since_24h, Scan.outcome == ScanOutcome.FAILED.value)
    # A failed scan with no vision_result never got a reading back: vision-service errored or was
    # unreachable, as opposed to a readable scan that failed validation (bad roll number, etc.).
    # SQLAlchemy persists an explicit Python None in a JSONB column as JSON `null`, not SQL NULL,
    # so "no result" has to match both forms.
    no_vision_result = or_(Scan.vision_result.is_(None), func.jsonb_typeof(Scan.vision_result) == "null")
    vision_failures_1h = count(
        Scan.created_at >= since_1h, Scan.outcome == ScanOutcome.FAILED.value, no_vision_result
    )
    last_scan_at = session.exec(select(_instant(func.max(Scan.created_at)))).one()

    return {
        "scans_1h": count(Scan.created_at >= since_1h),
        "scans_24h": scans_24h,
        "graded_24h": graded_24h,
        "failed_24h": failed_24h,
        "failure_rate_24h": round(failed_24h / scans_24h, 4) if scans_24h else None,
        "vision_failures_1h": vision_failures_1h,
        "last_scan_at": _utc(last_scan_at).isoformat() if last_scan_at else None,
        "hours_since_last_scan": (
            round((now - last_scan_at).total_seconds() / 3600, 1) if last_scan_at else None
        ),
    }


def review_backlog(session: Session, now: datetime) -> dict:
    open_filter = ScanReview.status.in_(OPEN_REVIEW_STATUSES)
    open_reviews = session.exec(select(func.count()).select_from(ScanReview).where(open_filter)).one()
    oldest = session.exec(select(_instant(func.min(ScanReview.created_at))).where(open_filter)).one()
    return {
        "open_reviews": open_reviews,
        "oldest_open_review_at": _utc(oldest).isoformat() if oldest else None,
        "oldest_open_review_hours": round((now - oldest).total_seconds() / 3600, 1) if oldest else None,
    }


def evaluate_alerts(snapshot: dict, cfg=settings) -> list[dict]:
    """Snapshot -> [{severity, code, message}], criticals first. A missing section (e.g. the
    database was unreachable, so activity/backlog couldn't be read) simply skips its rules."""
    alerts: list[dict] = []

    def add(severity: str, code: str, message: str) -> None:
        alerts.append({"severity": severity, "code": code, "message": message})

    services = snapshot["services"]
    if not services["database"]["ok"]:
        add("critical", "database_down", f"Database is unreachable ({services['database']['error']}).")
    if not services["vision_service"]["ok"]:
        add(
            "critical", "vision_down",
            f"Vision service is unreachable ({services['vision_service']['error']}) -- "
            "incoming scans can't be read until it's back.",
        )

    activity = snapshot.get("activity")
    if activity:
        if activity["vision_failures_1h"] >= cfg.alert_vision_failures_1h:
            add(
                "warning", "vision_failures",
                f"{activity['vision_failures_1h']} scans in the last hour failed because the vision service errored.",
            )
        rate = activity["failure_rate_24h"]
        if rate is not None and activity["scans_24h"] >= cfg.alert_min_scans_for_failure_rate and rate >= cfg.alert_failure_rate:
            add(
                "warning", "high_failure_rate",
                f"{rate:.0%} of the last {activity['scans_24h']} scans (24h) failed "
                f"(alert threshold {cfg.alert_failure_rate:.0%}).",
            )
        silent_hours = activity["hours_since_last_scan"]
        if cfg.alert_silence_hours and silent_hours is not None and silent_hours >= cfg.alert_silence_hours:
            add(
                "warning", "no_recent_scans",
                f"No scans received for {silent_hours:g} hours -- if that's unexpected, check the Exotel webhook.",
            )

    backlog = snapshot.get("backlog")
    if backlog:
        if backlog["open_reviews"] >= cfg.alert_open_reviews:
            add("warning", "review_backlog", f"{backlog['open_reviews']} failed scans are waiting for review.")
        age = backlog["oldest_open_review_hours"]
        if age is not None and age >= cfg.alert_review_age_hours:
            add("warning", "stale_review", f"The oldest failed scan has been waiting {age:g} hours.")

    storage = snapshot["storage"]
    if storage["ok"]:
        used = storage["used_pct"]
        detail = f"Storage is {used:g}% full ({storage['free_gb']:g} GB free) -- new scans will fail once it's full."
        if used >= cfg.alert_disk_critical_pct:
            add("critical", "disk_critical", detail)
        elif used >= cfg.alert_disk_warn_pct:
            add("warning", "disk_warning", detail)

    return sorted(alerts, key=lambda a: _SEVERITY_ORDER[a["severity"]])


def collect(session: Session, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    snapshot: dict = {
        "generated_at": now.isoformat(),
        "services": {"database": check_database(session), "vision_service": check_vision_service()},
        "storage": storage_stats(),
    }
    if snapshot["services"]["database"]["ok"]:
        snapshot["activity"] = scan_activity(session, now)
        snapshot["backlog"] = review_backlog(session, now)

    snapshot["alerts"] = evaluate_alerts(snapshot)
    severities = {a["severity"] for a in snapshot["alerts"]}
    snapshot["status"] = "critical" if "critical" in severities else "warning" if severities else "ok"
    return snapshot
