-- Ensure student IDs are stored as 4-digit numeric strings starting at 0002.
-- 0001 is reserved for the test student and is not auto-assigned by the importer.

ALTER TABLE students
    ALTER COLUMN student_id TYPE text;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'students_student_id_check'
    ) THEN
        ALTER TABLE students
            ADD CONSTRAINT students_student_id_check
            CHECK (student_id ~ '^[0-9]{4}$');
    END IF;
END $$;

UPDATE students
SET student_id = LPAD(student_id, 4, '0')
WHERE student_id IS NOT NULL
  AND student_id !~ '^[0-9]{4}$';

DO $$
DECLARE
    max_id integer;
BEGIN
    SELECT COALESCE(MAX(student_id::integer), 1) INTO max_id
    FROM students
    WHERE student_id ~ '^[0-9]{4}$';

    IF max_id < 2 THEN
        INSERT INTO students (student_id, student_name, student_school_code, current_level, is_active)
        VALUES ('0001', 'Test Student', NULL, 'A', true)
        ON CONFLICT (student_id) DO NOTHING;
    END IF;
END $$;
