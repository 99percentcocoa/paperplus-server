"""Weekly metrics: the pure week-bucketing function (no DB), and collect()/the API route driven
against the real dev DB (same fixtures/pattern as test_admin_routes.py)."""

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.db.session import get_session
from app.main import app
from app.services.metrics import WeekRow, build_weekly_series, collect, week_start_utc

NOW = datetime(2026, 9, 24, 15, 0, tzinfo=timezone.utc)  # a Thursday
CURRENT_WEEK_START = datetime(2026, 9, 21, tzinfo=timezone.utc)  # the Monday of that week
PREVIOUS_WEEK_START = datetime(2026, 9, 14, tzinfo=timezone.utc)


def test_week_start_utc_finds_the_monday_regardless_of_weekday_or_input_zone():
    assert week_start_utc(NOW) == CURRENT_WEEK_START
    monday_midnight = datetime(2026, 9, 21, 0, 0, tzinfo=timezone.utc)
    sunday_end = datetime(2026, 9, 27, 23, 59, tzinfo=timezone.utc)
    assert week_start_utc(monday_midnight) == CURRENT_WEEK_START
    assert week_start_utc(sunday_end) == CURRENT_WEEK_START
    # An instant just after UTC midnight, expressed in a +5:30 zone as the previous local day --
    # must still land in the UTC week, since week boundaries are pinned to UTC, not the input zone.
    from datetime import timedelta

    ist = timezone(timedelta(hours=5, minutes=30))
    late_local_sunday = datetime(2026, 9, 21, 4, 0, tzinfo=ist)  # 2026-09-20 22:30 UTC -> previous week
    assert week_start_utc(late_local_sunday) == PREVIOUS_WEEK_START


def test_build_weekly_series_fills_gaps_and_computes_rates_against_active_students():
    rows = [WeekRow(week_start=CURRENT_WEEK_START, total_worksheets=9, distinct_students=3)]

    series = build_weekly_series(rows, active_students=6, weeks=3, now=NOW)

    assert [w["week_start"] for w in series] == ["2026-09-07", "2026-09-14", "2026-09-21"]
    empty_week = series[0]
    assert (empty_week["total_worksheets"], empty_week["distinct_students"]) == (0, 0)
    assert (empty_week["avg_worksheets_per_student"], empty_week["pct_students_active"]) == (0.0, 0.0)

    current = series[-1]
    assert current["is_current"] is True and current["week_end"] == "2026-09-28"
    assert (current["total_worksheets"], current["distinct_students"]) == (9, 3)
    assert (current["avg_worksheets_per_student"], current["pct_students_active"]) == (1.5, 50.0)
    assert all(not w["is_current"] for w in series[:-1])


def test_build_weekly_series_handles_zero_active_students_without_dividing_by_zero():
    series = build_weekly_series([], active_students=0, weeks=1, now=NOW)
    assert (series[0]["avg_worksheets_per_student"], series[0]["pct_students_active"]) == (None, None)


@pytest.fixture
def client(scan_env):
    app.dependency_overrides[get_session] = lambda: scan_env.session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_collect_counts_a_real_scan_in_its_own_week(scan_env):
    scan_env.run_scan({1: "A", 2: "B", 3: "A", 4: None})  # score irrelevant here; one worksheet solved

    data = collect(scan_env.session, weeks=2)

    assert data["active_students"] >= 1
    this_week = next(w for w in data["weeks"] if w["is_current"])
    assert this_week["total_worksheets"] >= 1 and this_week["distinct_students"] >= 1


def test_weekly_metrics_endpoint(scan_env, client):
    scan_env.run_scan({1: "A", 2: "B", 3: "A", 4: None})

    response = client.get("/api/admin/metrics/weekly", params={"weeks": 3})

    assert response.status_code == 200
    body = response.json()
    assert len(body["weeks"]) == 3
    assert body["weeks"][-1]["is_current"] is True
    assert body["weeks"][-1]["total_worksheets"] >= 1

    assert client.get("/api/admin/metrics/weekly", params={"weeks": 0}).status_code == 422
    assert client.get("/api/admin/metrics/weekly", params={"weeks": 999}).status_code == 422
