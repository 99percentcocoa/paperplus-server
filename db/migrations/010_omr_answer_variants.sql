-- Store answer-key variants keyed by OMR template and scanned question-paper code.
-- This keeps legacy worksheet JSON as the fallback, while allowing the new
-- basic_omr format to resolve the correct answer set from the OCR'd paper code.

CREATE TABLE IF NOT EXISTS omr_answer_sets (
    id serial PRIMARY KEY,
    template_name text NOT NULL,
    question_paper_code text NOT NULL,
    worksheet_id integer,
    answer_key_json jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (template_name, question_paper_code)
);

CREATE INDEX IF NOT EXISTS idx_omr_answer_sets_template_code
    ON omr_answer_sets (template_name, question_paper_code);

CREATE INDEX IF NOT EXISTS idx_omr_answer_sets_worksheet_id
    ON omr_answer_sets (worksheet_id);

ALTER TABLE omr_answer_sets
    ADD CONSTRAINT omr_answer_sets_question_paper_code_valid
    CHECK (question_paper_code ~ '^[A-F]$');
