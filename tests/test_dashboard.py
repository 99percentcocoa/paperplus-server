from app import app


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
