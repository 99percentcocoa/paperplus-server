"""Dashboard Routes - read-only operational overview for schools and submissions."""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template_string

from db.connection import get_connection


dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/dashboard", methods=["GET"])
def dashboard_page():
    """Render a lightweight public dashboard summary page."""
    html = """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>PaperPlus Dashboard</title>
        <style>
            :root {
                --bg: #f4f7fb;
                --card: white;
                --text: #1f2937;
                --muted: #6b7280;
                --primary: #1d4ed8;
                --accent: #10b981;
                --border: #e5e7eb;
            }
            body {
                margin: 0;
                font-family: Arial, sans-serif;
                background: var(--bg);
                color: var(--text);
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 32px 20px 48px;
            }
            h1 {
                margin-bottom: 8px;
            }
            .subtitle {
                color: var(--muted);
                margin-bottom: 24px;
            }
            .cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 16px;
                margin-bottom: 24px;
            }
            .card {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 18px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            }
            .label {
                color: var(--muted);
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }
            .value {
                font-size: 2rem;
                font-weight: bold;
                margin-top: 8px;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                gap: 18px;
            }
            .panel {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 16px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 12px;
                font-size: 14px;
            }
            th, td {
                padding: 10px 8px;
                border-bottom: 1px solid var(--border);
                text-align: left;
            }
            th {
                color: var(--muted);
                font-weight: 600;
            }
            .status {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 999px;
                background: #ecfdf5;
                color: #065f46;
                font-size: 12px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PaperPlus Dashboard</h1>
            <div class="subtitle">Live summary of schools, students, and submissions.</div>

            <div class="cards">
                <div class="card">
                    <div class="label">Schools</div>
                    <div id="schools-count" class="value">--</div>
                </div>
                <div class="card">
                    <div class="label">Students</div>
                    <div id="students-count" class="value">--</div>
                </div>
                <div class="card">
                    <div class="label">Submissions</div>
                    <div id="submissions-count" class="value">--</div>
                </div>
                <div class="card">
                    <div class="label">Last 24h</div>
                    <div id="last-day-count" class="value">--</div>
                </div>
            </div>

            <div class="grid">
                <div class="panel">
                    <h3>Schools</h3>
                    <div id="schools-table"></div>
                </div>
                <div class="panel">
                    <h3>Recent submissions</h3>
                    <div id="submissions-table"></div>
                </div>
            </div>
        </div>

        <script>
            async function loadDashboard() {
                const summary = await fetch('/api/dashboard/summary');
                const data = await summary.json();

                document.getElementById('schools-count').textContent = data.schools.total;
                document.getElementById('students-count').textContent = data.students.total;
                document.getElementById('submissions-count').textContent = data.submissions.total;
                document.getElementById('last-day-count').textContent = data.submissions.last_24_hours;

                const schoolRows = data.schools.rows.map(row => `
                    <tr>
                        <td>${row.school_name}</td>
                        <td>${row.student_count}</td>
                    </tr>
                `).join('') || '<p>No schools found.</p>';

                document.getElementById('schools-table').innerHTML = `
                    <table>
                        <thead><tr><th>School</th><th>Students</th></tr></thead>
                        <tbody>${schoolRows}</tbody>
                    </table>
                `;

                const submissionRows = data.recent_submissions.map(row => `
                    <tr>
                        <td>${row.student_id}</td>
                        <td>${row.worksheet_id}</td>
                        <td>${row.score}</td>
                        <td>${new Date(row.submitted_at).toLocaleString()}</td>
                    </tr>
                `).join('') || '<p>No submissions yet.</p>';

                document.getElementById('submissions-table').innerHTML = `
                    <table>
                        <thead><tr><th>Student</th><th>Worksheet</th><th>Score</th><th>Submitted</th></tr></thead>
                        <tbody>${submissionRows}</tbody>
                    </table>
                `;
            }

            loadDashboard();
            setInterval(loadDashboard, 30000);
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


@dashboard_bp.route("/api/dashboard/summary", methods=["GET"])
def dashboard_summary():
    """Return a compact live summary for the dashboard UI."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS total FROM schools")
                schools_total = cur.fetchone()["total"]

                cur.execute("SELECT COUNT(*) AS total FROM students WHERE is_active = true")
                students_total = cur.fetchone()["total"]

                cur.execute("SELECT COUNT(*) AS total FROM submissions")
                submissions_total = cur.fetchone()["total"]

                cur.execute(
                    "SELECT COUNT(*) AS total FROM submissions WHERE submitted_at >= NOW() - INTERVAL '24 hours'"
                )
                last_24_hours = cur.fetchone()["total"]

                cur.execute(
                    """
                    SELECT s.school_code, s.school_name,
                           COUNT(st.student_id) AS student_count
                    FROM schools s
                    LEFT JOIN students st ON st.student_school_code = s.school_code
                    GROUP BY s.school_code, s.school_name
                    ORDER BY s.school_name
                    """
                )
                schools_rows = cur.fetchall()

                cur.execute(
                    """
                    SELECT submission_id, student_id, worksheet_id, score, from_number, submitted_at
                    FROM submissions
                    ORDER BY submitted_at DESC
                    LIMIT 10
                    """
                )
                recent_submissions = cur.fetchall()
    except Exception:
        return jsonify({
            "schools": {"total": 0, "rows": []},
            "students": {"total": 0},
            "submissions": {"total": 0, "last_24_hours": 0},
            "recent_submissions": [],
            "error": "Unable to load dashboard data",
        }), 500

    return jsonify({
        "schools": {"total": schools_total, "rows": schools_rows},
        "students": {"total": students_total},
        "submissions": {"total": submissions_total, "last_24_hours": last_24_hours},
        "recent_submissions": recent_submissions,
    })


@dashboard_bp.route("/api/dashboard/schools", methods=["GET"])
def dashboard_schools():
    """List schools with student counts."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.school_code, s.school_name,
                       COUNT(st.student_id) AS student_count
                FROM schools s
                LEFT JOIN students st ON st.student_school_code = s.school_code
                GROUP BY s.school_code, s.school_name
                ORDER BY s.school_name
                """
            )
            return jsonify(cur.fetchall())


@dashboard_bp.route("/api/dashboard/students", methods=["GET"])
def dashboard_students():
    """List students optionally filtered by school."""
    school_code = None
    if "school_code" in __import__("flask").request.args:
        school_code = __import__("flask").request.args.get("school_code")

    query = """
        SELECT student_id, student_name, student_school_code, current_level, is_active
        FROM students
    """
    params = []
    if school_code:
        query += " WHERE student_school_code = %s"
        params.append(school_code)
    query += " ORDER BY student_name"

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return jsonify(cur.fetchall())


@dashboard_bp.route("/api/dashboard/submissions", methods=["GET"])
def dashboard_submissions():
    """List the most recent submissions."""
    limit = __import__("flask").request.args.get("limit", default=25, type=int)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT submission_id, student_id, worksheet_id, score, from_number, submitted_at
                FROM submissions
                ORDER BY submitted_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            return jsonify(cur.fetchall())
