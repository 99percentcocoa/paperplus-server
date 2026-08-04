-- Remove the assignments table and its links from other tables

ALTER TABLE submissions DROP COLUMN IF EXISTS assignment_id;

DROP TABLE IF EXISTS assignments;
