-- Add OMR compatibility metadata while preserving legacy worksheets.
-- This keeps older rows valid while allowing new OMR sheets to store page-level
-- information and explicit sheet-version metadata without rewriting existing data.

ALTER TABLE worksheets
    ADD COLUMN IF NOT EXISTS sheet_version text NOT NULL DEFAULT 'legacy';

ALTER TABLE worksheets
    ADD COLUMN IF NOT EXISTS page_count integer NOT NULL DEFAULT 1;

ALTER TABLE worksheets
    ADD COLUMN IF NOT EXISTS total_question_count integer;

ALTER TABLE worksheets
    ADD COLUMN IF NOT EXISTS worksheet_metadata jsonb NOT NULL DEFAULT '{}'::jsonb;

UPDATE worksheets
SET sheet_version = 'legacy'
WHERE sheet_version IS NULL;

UPDATE worksheets
SET page_count = 1
WHERE page_count IS NULL;

CREATE TABLE IF NOT EXISTS worksheet_pages (
    worksheet_page_id serial PRIMARY KEY,
    worksheet_id integer NOT NULL REFERENCES worksheets(worksheet_id) ON DELETE CASCADE,
    page_no integer NOT NULL CHECK (page_no >= 1),
    first_question_index integer NOT NULL CHECK (first_question_index >= 1),
    last_question_index integer NOT NULL CHECK (last_question_index >= first_question_index),
    expected_row_tag_count integer NOT NULL DEFAULT 10,
    page_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (worksheet_id, page_no)
);

-- Backfill the first page for existing worksheets so legacy records remain queryable.
INSERT INTO worksheet_pages (worksheet_id, page_no, first_question_index, last_question_index, expected_row_tag_count, page_metadata)
SELECT
    w.worksheet_id,
    1,
    1,
    COALESCE(w.total_question_count, w.max_score, 0),
    10,
    jsonb_build_object('legacy', true)
FROM worksheets w
LEFT JOIN worksheet_pages wp
    ON wp.worksheet_id = w.worksheet_id AND wp.page_no = 1
WHERE wp.worksheet_id IS NULL
  AND COALESCE(w.total_question_count, w.max_score, 0) > 0;

UPDATE worksheets
SET total_question_count = COALESCE(total_question_count, max_score)
WHERE total_question_count IS NULL;
