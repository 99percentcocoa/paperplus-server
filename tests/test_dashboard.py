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
