-- Add category support so worksheets and submissions can be separated into
-- classroom practice vs homework tracking without changing the legacy is_test flag.

ALTER TABLE worksheets
    ADD COLUMN IF NOT EXISTS worksheet_category text NOT NULL DEFAULT 'practice';

ALTER TABLE submissions
    ADD COLUMN IF NOT EXISTS worksheet_category text NOT NULL DEFAULT 'practice';

UPDATE worksheets
SET worksheet_category = 'practice'
WHERE worksheet_category IS NULL;

UPDATE submissions
SET worksheet_category = 'practice'
WHERE worksheet_category IS NULL;
