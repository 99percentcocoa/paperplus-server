"""Dashboard Routes - read-only operational overview for schools and submissions."""

from __future__ import annotations

import json

from flask import Blueprint, jsonify, render_template_string, request

from db.connection import get_connection


dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/dashboard", methods=["GET"])
def dashboard_page():
    """Render the dashboard summary page with review queue data."""
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
            <div class="subtitle">Live summary of schools, students, submissions, and review queue.</div>

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

            <div class="grid" style="margin-top: 20px;">
                <div class="panel">
                    <h3>All scans</h3>
                    <div id="all-scans-table"></div>
                </div>
                <div class="panel">
                    <h3>Failed scans</h3>
                    <div id="failed-scans-table"></div>
                </div>
            </div>
        </div>

        <script>
            function renderSimpleTable(targetId, rows, columns) {
                const table = rows.length ? `
                    <table>
                        <thead><tr>${columns.map(col => `<th>${col.label}</th>`).join('')}</tr></thead>
                        <tbody>${rows.map(row => `<tr>${columns.map(col => `<td>${row[col.key] ?? ''}</td>`).join('')}</tr>`).join('')}</tbody>
                    </table>
                ` : '<p>No rows found.</p>';
                document.getElementById(targetId).innerHTML = table;
            }

            async function loadDashboard() {
                const summary = await fetch('/api/dashboard/summary');
                const data = await summary.json();

                document.getElementById('schools-count').textContent = data.schools.total;
                document.getElementById('students-count').textContent = data.students.total;
                document.getElementById('submissions-count').textContent = data.submissions.total;
                document.getElementById('last-day-count').textContent = data.submissions.last_24_hours;

                const schoolRows = data.schools.rows.map(row => ({
                    school_name: row.school_name,
                    student_count: row.student_count,
                }));
                renderSimpleTable('schools-table', schoolRows, [
                    { key: 'school_name', label: 'School' },
                    { key: 'student_count', label: 'Students' },
                ]);

                const submissionRows = data.recent_submissions.map(row => ({
                    student_id: row.student_id,
                    worksheet_id: row.worksheet_id,
                    score: row.score,
                    submitted_at: new Date(row.submitted_at).toLocaleString(),
                }));
                renderSimpleTable('submissions-table', submissionRows, [
                    { key: 'student_id', label: 'Student' },
                    { key: 'worksheet_id', label: 'Worksheet' },
                    { key: 'score', label: 'Score' },
                    { key: 'submitted_at', label: 'Submitted' },
                ]);

                const reviewResponse = await fetch('/api/dashboard/review');
                const reviewData = await reviewResponse.json();
                const allScans = reviewData.all_scans || [];
                const failedScans = reviewData.failed_scans || [];

                renderSimpleTable('all-scans-table', allScans.map(row => ({
                    student_id: row.student_id,
                    worksheet_id: row.worksheet_id,
                    score: row.score,
                    submitted_at: row.submitted_at ? new Date(row.submitted_at).toLocaleString() : '',
                })), [
                    { key: 'student_id', label: 'Student' },
                    { key: 'worksheet_id', label: 'Worksheet' },
                    { key: 'score', label: 'Score' },
                    { key: 'submitted_at', label: 'Submitted' },
                ]);

                renderSimpleTable('failed-scans-table', failedScans.map(row => ({
                    status: row.status || 'failed',
                    student_id: row.student_id || row.detected_roll_number || 'Unknown',
                    worksheet_id: row.worksheet_id,
                    error_reason: row.error_reason || 'Unknown',
                })), [
                    { key: 'status', label: 'Status' },
                    { key: 'student_id', label: 'Student' },
                    { key: 'worksheet_id', label: 'Worksheet' },
                    { key: 'error_reason', label: 'Reason' },
                ]);
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


@dashboard_bp.route("/api/dashboard/review", methods=["GET"])
def dashboard_review():
    """Return all scans and failed review queue entries."""
    status = request.args.get("status", "all").lower()

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT submission_id, student_id, worksheet_id, score, from_number, answers_json, submitted_at
                    FROM submissions
                    ORDER BY submitted_at DESC
                    LIMIT 100
                    """
                )
                all_scans = cur.fetchall()

                try:
                    cur.execute(
                        """
                        SELECT review_id, submission_id, student_id, worksheet_id, detected_roll_number,
                               status, error_reason, corrected_answers, original_score, corrected_score,
                               created_at, updated_at
                        FROM scan_reviews
                        ORDER BY created_at DESC
                        LIMIT 100
                        """
                    )
                    failed_scans = cur.fetchall()
                except Exception:
                    failed_scans = []
    except Exception:
        return jsonify({
            "all_scans": [],
            "failed_scans": [],
            "total_all": 0,
            "total_failed": 0,
            "error": "Unable to load review data",
        }), 500

    if status != "all":
        failed_scans = [
            row for row in failed_scans
            if (row.get("status") or "failed").lower() == status
        ]

    return jsonify({
        "all_scans": all_scans,
        "failed_scans": failed_scans,
        "total_all": len(all_scans),
        "total_failed": len(failed_scans),
    })


@dashboard_bp.route("/api/dashboard/review/<int:review_id>/correct", methods=["POST"])
def dashboard_review_correct(review_id):
    """Update a review row and any matching submission row with corrected values."""
    payload = request.get_json(silent=True) or {}
    corrected_student_id = payload.get("student_id") or payload.get("roll_number")
    corrected_answers = payload.get("answers") or payload.get("marked_answers") or []
    corrected_score = payload.get("score")

    if not isinstance(corrected_answers, list):
        return jsonify({"error": "answers must be a list of question entries."}), 400

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM scan_reviews WHERE review_id = %s", (review_id,))
                review = cur.fetchone()
                if review is None:
                    return jsonify({"error": "Review row not found."}), 404

                if corrected_student_id is None:
                    corrected_student_id = review.get("student_id") or review.get("detected_roll_number")

                if corrected_score is None:
                    corrected_score = review.get("corrected_score") or review.get("original_score") or 0

                corrected_score = int(corrected_score)

                cur.execute(
                    """
                    UPDATE scan_reviews
                    SET student_id = %s,
                        detected_roll_number = %s,
                        corrected_answers = %s,
                        corrected_score = %s,
                        status = 'corrected',
                        corrected_at = NOW(),
                        corrected_by = %s,
                        updated_at = NOW()
                    WHERE review_id = %s
                    """,
                    (
                        corrected_student_id,
                        corrected_student_id,
                        json.dumps(corrected_answers),
                        corrected_score,
                        payload.get("corrected_by", "admin"),
                        review_id,
                    ),
                )

                if review.get("submission_id") is not None:
                    cur.execute(
                        """
                        UPDATE submissions
                        SET student_id = %s,
                            answers_json = %s,
                            score = %s,
                            submitted_at = NOW()
                        WHERE submission_id = %s
                        """,
                        (
                            corrected_student_id,
                            json.dumps(corrected_answers),
                            corrected_score,
                            review["submission_id"],
                        ),
                    )
    except Exception:
        return jsonify({"error": "Unable to update review record."}), 500

    return jsonify({
        "status": "corrected",
        "review_id": review_id,
        "student_id": corrected_student_id,
        "score": corrected_score,
        "answers": corrected_answers,
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
