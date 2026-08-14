CREATE TABLE IF NOT EXISTS scan_reviews (
    review_id serial PRIMARY KEY,
    submission_id integer REFERENCES submissions(submission_id) ON DELETE SET NULL,
    student_id text,
    worksheet_id integer REFERENCES worksheets(worksheet_id) ON DELETE SET NULL,
    detected_roll_number text,
    status text NOT NULL DEFAULT 'failed' CHECK (status IN ('failed', 'needs_review', 'corrected', 'approved')),
    error_reason text,
    original_answers jsonb,
    corrected_answers jsonb,
    original_score integer,
    corrected_score integer,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    corrected_by text,
    corrected_at timestamp with time zone
);

CREATE INDEX IF NOT EXISTS idx_scan_reviews_status
    ON scan_reviews (status);

CREATE INDEX IF NOT EXISTS idx_scan_reviews_created_at
    ON scan_reviews (created_at DESC);
