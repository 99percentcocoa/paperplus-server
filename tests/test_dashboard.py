import builtins
import importlib
import sys
from pathlib import Path

from app import app


def test_webhook_routes_import_is_lazy_for_heavy_image_dependencies():
    project_root = Path(__file__).resolve().parents[1]
    original_sys_path = list(sys.path)
    original_meta_path = list(sys.meta_path)
    original_import = builtins.__import__

    class BlockImageServiceImport:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "services.image_service":
                raise ModuleNotFoundError("simulated missing image dependency")
            return None

    sys.path.insert(0, str(project_root))
    sys.meta_path.insert(0, BlockImageServiceImport())

    try:
        for name in [
            "routes.webhook_routes",
            "services.message_service",
            "services.image_service",
        ]:
            sys.modules.pop(name, None)

        module = importlib.import_module("routes.webhook_routes")
        assert hasattr(module, "webhook_bp")
    finally:
        sys.path[:] = original_sys_path
        sys.meta_path[:] = original_meta_path
        builtins.__import__ = original_import


def test_dashboard_summary_api_returns_data():
    client = app.test_client()
    response = client.get('/api/dashboard/summary')
    assert response.status_code == 200
    payload = response.get_json()
    assert 'schools' in payload
    assert 'students' in payload
    assert 'submissions' in payload
    assert 'recent_submissions' in payload


def test_dashboard_page_loads():
    client = app.test_client()
    response = client.get('/dashboard')
    assert response.status_code == 200
    assert b'PaperPlus Dashboard' in response.data


def test_dashboard_review_api_returns_payload_shape():
    client = app.test_client()
    response = client.get('/api/dashboard/review')
    assert response.status_code == 200
    payload = response.get_json()
    assert 'all_scans' in payload
    assert 'failed_scans' in payload
    assert isinstance(payload['all_scans'], list)
    assert isinstance(payload['failed_scans'], list)


def test_dashboard_review_correct_route_exists():
    client = app.test_client()
    response = client.post('/api/dashboard/review/999/correct', json={})
    assert response.status_code in (200, 400, 404)


def test_dashboard_health_api_returns_status_payload():
    client = app.test_client()
    response = client.get('/api/dashboard/health')
    assert response.status_code == 200
    payload = response.get_json()
    assert 'status' in payload
    assert 'checked_at' in payload


def test_message_latency_helper_works():
    from datetime import datetime, timedelta
    from services.message_service import update_message_latency, get_recent_message_latency

    received_at = datetime.utcnow()
    response_sent_at = received_at + timedelta(seconds=2.5)
    update_message_latency(
        from_number='99999',
        received_at=received_at,
        response_sent_at=response_sent_at,
        status='ok',
    )

    recent = get_recent_message_latency(limit=10)
    assert recent
    assert recent[0]['duration_ms'] >= 2000


def test_timing_context_records_timestamps():
    import time
    from services.logging_service import timing_context

    with timing_context('unit_test_op', test_case='timing') as ctx:
        time.sleep(0.02)

    assert 'started_at' in ctx
    assert 'ended_at' in ctx
    assert 'duration_ms' in ctx
    assert ctx['duration_ms'] >= 10
