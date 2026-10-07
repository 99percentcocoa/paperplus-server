"""Dashboard monitoring: alert rules (pure, no DB), the probe helpers, and the /api/admin/monitoring
endpoint driven through TestClient against the real dev DB (same fixtures/pattern as
test_admin_routes.py)."""

from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app.db.session import get_session
from app.main import app
from app.models import Scan, ScanReview
from app.models.submission import ScanOutcome, ScanReviewStatus
from app.services import monitoring
from tests.conftest import ENV_FROM_NUMBER

CFG = SimpleNamespace(
    alert_min_scans_for_failure_rate=5,
    alert_failure_rate=0.3,
    alert_vision_failures_1h=3,
    alert_open_reviews=10,
    alert_review_age_hours=24,
    alert_disk_warn_pct=85.0,
    alert_disk_critical_pct=95.0,
    alert_silence_hours=48,
)


def snapshot(**overrides) -> dict:
    """A fully healthy snapshot; tests override just the section they exercise."""
    base = {
        "services": {"database": {"ok": True, "latency_ms": 2}, "vision_service": {"ok": True, "latency_ms": 5}},
        "storage": {"ok": True, "total_gb": 100.0, "free_gb": 60.0, "used_pct": 40.0},
        "activity": {
            "scans_1h": 2, "scans_24h": 20, "graded_24h": 19, "failed_24h": 1, "failure_rate_24h": 0.05,
            "vision_failures_1h": 0, "last_scan_at": "2026-09-19T10:00:00+00:00", "hours_since_last_scan": 1.0,
        },
        "backlog": {"open_reviews": 1, "oldest_open_review_at": None, "oldest_open_review_hours": 2.0},
    }
    base.update(overrides)
    return base


def codes(alerts: list[dict]) -> list[str]:
    return [a["code"] for a in alerts]


# ---------- alert rules ----------

def test_healthy_snapshot_has_no_alerts():
    assert monitoring.evaluate_alerts(snapshot(), CFG) == []


def test_dependencies_down_are_critical():
    services = {"database": {"ok": False, "error": "OperationalError"}, "vision_service": {"ok": False, "error": "timed out"}}
    alerts = monitoring.evaluate_alerts(snapshot(services=services), CFG)
    assert codes(alerts) == ["database_down", "vision_down"]
    assert {a["severity"] for a in alerts} == {"critical"}
    assert "timed out" in alerts[1]["message"]


def test_failure_rate_needs_minimum_sample_and_threshold():
    def rate_alerts(scans_24h, rate):
        activity = {**snapshot()["activity"], "scans_24h": scans_24h, "failure_rate_24h": rate}
        return codes(monitoring.evaluate_alerts(snapshot(activity=activity), CFG))

    assert rate_alerts(2, 0.5) == []  # too few scans to mean anything
    assert rate_alerts(10, 0.29) == []  # under the threshold
    assert rate_alerts(10, 0.3) == ["high_failure_rate"]  # at the threshold
    assert rate_alerts(0, None) == []  # no scans at all: rate undefined, not an alert


def test_vision_failures_in_last_hour():
    activity = {**snapshot()["activity"], "vision_failures_1h": 3}
    assert codes(monitoring.evaluate_alerts(snapshot(activity=activity), CFG)) == ["vision_failures"]
    activity["vision_failures_1h"] = 2
    assert monitoring.evaluate_alerts(snapshot(activity=activity), CFG) == []


def test_silence_alert_fires_only_when_enabled_and_a_scan_was_ever_seen():
    quiet = {**snapshot()["activity"], "hours_since_last_scan": 60.0}
    assert codes(monitoring.evaluate_alerts(snapshot(activity=quiet), CFG)) == ["no_recent_scans"]

    disabled = SimpleNamespace(**{**vars(CFG), "alert_silence_hours": 0})
    assert monitoring.evaluate_alerts(snapshot(activity=quiet), disabled) == []

    never = {**snapshot()["activity"], "last_scan_at": None, "hours_since_last_scan": None, "scans_24h": 0, "failure_rate_24h": None}
    assert monitoring.evaluate_alerts(snapshot(activity=never), CFG) == []  # a brand-new deployment isn't "silent"


def test_review_backlog_and_staleness():
    backlog = {"open_reviews": 12, "oldest_open_review_at": None, "oldest_open_review_hours": 30.0}
    assert codes(monitoring.evaluate_alerts(snapshot(backlog=backlog), CFG)) == ["review_backlog", "stale_review"]


def test_disk_thresholds_pick_the_right_severity():
    def disk_alerts(used_pct):
        storage = {"ok": True, "total_gb": 100.0, "free_gb": 100.0 - used_pct, "used_pct": used_pct}
        return [(a["severity"], a["code"]) for a in monitoring.evaluate_alerts(snapshot(storage=storage), CFG)]

    assert disk_alerts(84.9) == []
    assert disk_alerts(85.0) == [("warning", "disk_warning")]
    assert disk_alerts(95.0) == [("critical", "disk_critical")]


def test_sections_missing_when_database_is_down_are_skipped_not_errors():
    partial = snapshot(services={"database": {"ok": False, "error": "OperationalError"}, "vision_service": {"ok": True}})
    del partial["activity"], partial["backlog"]
    assert codes(monitoring.evaluate_alerts(partial, CFG)) == ["database_down"]


def test_criticals_sort_before_warnings():
    storage = {"ok": True, "total_gb": 100.0, "free_gb": 3.0, "used_pct": 97.0}
    backlog = {"open_reviews": 50, "oldest_open_review_at": None, "oldest_open_review_hours": 1.0}
    alerts = monitoring.evaluate_alerts(snapshot(storage=storage, backlog=backlog), CFG)
    assert [a["severity"] for a in alerts] == ["critical", "warning"]


# ---------- probes ----------

def test_vision_probe_errors_do_not_leak_urls(monkeypatch):
    def refuse(*args, **kwargs):
        raise httpx.ConnectError("All connection attempts failed for http://vision-service:8100/health")

    monkeypatch.setattr(monitoring.httpx, "get", refuse)
    assert monitoring.check_vision_service() == {"ok": False, "error": "connection failed"}

    def slow(*args, **kwargs):
        raise httpx.ReadTimeout("read timed out")

    monkeypatch.setattr(monitoring.httpx, "get", slow)
    assert monitoring.check_vision_service() == {"ok": False, "error": "timed out"}


def test_storage_stats_falls_back_to_an_existing_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(monitoring.settings, "storage_root", str(tmp_path / "not" / "created" / "yet"))
    stats = monitoring.storage_stats()
    assert stats["ok"] and 0 <= stats["used_pct"] <= 100 and stats["total_gb"] > 0


# ---------- endpoint ----------

@pytest.fixture
def client(scan_env, monkeypatch):
    monkeypatch.setattr(monitoring, "check_vision_service", lambda: {"ok": True, "latency_ms": 3})
    app.dependency_overrides[get_session] = lambda: scan_env.session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_endpoint_reports_scan_activity_and_review_backlog(scan_env, client):
    before = client.get("/api/admin/monitoring").json()

    scan_env.run_scan({1: "A", 2: "B", 3: "A", 4: None})
    scan_env.session.add(Scan(correlation_id="crt-mon-vision", from_number=ENV_FROM_NUMBER, outcome=ScanOutcome.FAILED.value))
    scan_env.session.add(ScanReview(correlation_id="crt-mon-review", status=ScanReviewStatus.FAILED.value, error_reason="test"))
    scan_env.session.commit()

    after = client.get("/api/admin/monitoring").json()
    assert set(after) >= {"generated_at", "status", "alerts", "services", "storage", "activity", "backlog"}
    assert after["services"]["database"]["ok"] and after["services"]["vision_service"] == {"ok": True, "latency_ms": 3}

    delta = lambda section, key: after[section][key] - before[section][key]  # noqa: E731
    assert delta("activity", "scans_24h") == 2  # the graded scan + the vision failure
    assert delta("activity", "graded_24h") == 1
    assert delta("activity", "failed_24h") == 1
    assert delta("activity", "vision_failures_1h") == 1  # failed with no vision_result
    assert delta("backlog", "open_reviews") == 1
    # Just created, so "0 to <1 hour ago" -- a lower bound too, since a naive-timestamp/timezone mix-up
    # (DB session in a non-UTC zone) shows up as a large negative age, and `< 1` alone passes that.
    assert 0 <= after["activity"]["hours_since_last_scan"] < 1
    last_scan_at = datetime.fromisoformat(after["activity"]["last_scan_at"])
    assert last_scan_at.tzinfo is not None
    assert abs((datetime.now(timezone.utc) - last_scan_at).total_seconds()) < 300
    oldest_review = after["backlog"]["oldest_open_review_hours"]
    assert oldest_review is not None and 0 <= oldest_review < 1 or before["backlog"]["open_reviews"] > 0
    assert after["status"] in {"ok", "warning", "critical"}


def test_endpoint_goes_critical_when_vision_service_is_down(scan_env, client, monkeypatch):
    monkeypatch.setattr(monitoring, "check_vision_service", lambda: {"ok": False, "error": "connection failed"})
    body = client.get("/api/admin/monitoring").json()
    assert body["status"] == "critical"
    assert "vision_down" in codes(body["alerts"])
    assert body["alerts"][0]["severity"] == "critical"  # criticals are listed first


def test_collect_survives_an_unreachable_database(monkeypatch):
    class BrokenSession:
        def exec(self, *args, **kwargs):
            raise RuntimeError("server closed the connection: host=secret-db user=postgres")

    monkeypatch.setattr(monitoring, "check_vision_service", lambda: {"ok": True, "latency_ms": 3})
    result = monitoring.collect(BrokenSession(), now=datetime(2026, 9, 19, tzinfo=timezone.utc))

    assert result["status"] == "critical"
    assert result["services"]["database"] == {"ok": False, "error": "RuntimeError"}  # no DSN/host leaked
    assert "activity" not in result and "backlog" not in result  # can't be read, so not reported
    assert "database_down" in codes(result["alerts"])
