--- Creates core tables for the PaperPlus schema

-- schools
CREATE TABLE IF NOT EXISTS schools (
    school_code text PRIMARY KEY,
    school_name text NOT NULL
);

-- students
CREATE TABLE IF NOT EXISTS students (
    student_id text PRIMARY KEY,
    student_name text NOT NULL,
    student_school_code text REFERENCES schools(school_code),
    current_level text,
    is_active boolean NOT NULL DEFAULT true
);

-- skill (competency)
CREATE TABLE IF NOT EXISTS skills (
    skill_code text PRIMARY KEY,
    skill_name text NOT NULL,
    skill_level text NOT NULL,
    skill_weight numeric DEFAULT 1.0
);

-- worksheet
CREATE TABLE IF NOT EXISTS worksheets (
    worksheet_id serial PRIMARY KEY,
    worksheet_level text,
    is_test boolean NOT NULL DEFAULT false,
    max_score integer,
    lang text,
    worksheet_json jsonb
);

-- question
CREATE TABLE IF NOT EXISTS questions (
    question_id serial PRIMARY KEY,
    worksheet_id integer REFERENCES worksheets(worksheet_id) ON DELETE CASCADE,
    skill_code text NOT NULL REFERENCES skills(skill_code),
    question_json jsonb
);

-- submissions
CREATE TABLE IF NOT EXISTS submissions (
    submission_id serial PRIMARY KEY,
    student_id text NOT NULL REFERENCES students(student_id),
    worksheet_id integer NOT NULL REFERENCES worksheets(worksheet_id),
    score integer,
    from_number text,
    answers_json jsonb,
    submitted_at timestamp with time zone DEFAULT now()
);

-- attempt (one per question)
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id serial PRIMARY KEY,
    student_id text NOT NULL REFERENCES students(student_id),
    submission_id integer NOT NULL REFERENCES submissions(submission_id) ON DELETE CASCADE,
    question_id integer NOT NULL REFERENCES questions(question_id) ON DELETE CASCADE,
    worksheet_id integer NOT NULL REFERENCES worksheets(worksheet_id) ON DELETE CASCADE,
    is_correct boolean,
    skill_code text NOT NULL REFERENCES skills(skill_code),
    attempted_at timestamp with time zone DEFAULT now()
);

-- student_skill_mastery
CREATE TABLE IF NOT EXISTS student_skill_mastery (
    student_id text NOT NULL REFERENCES students(student_id),
    skill_code text NOT NULL REFERENCES skills(skill_code),
    mastery_score numeric,
    last_updated timestamp with time zone DEFAULT now(),
    PRIMARY KEY (student_id, skill_code)
);

-- media
CREATE TABLE IF NOT EXISTS media (
    media_id serial PRIMARY KEY,
    owner_type text NOT NULL,
    owner_id integer NOT NULL,
    storage_path text NOT NULL,
    created_at timestamp with time zone DEFAULT now()
);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version text PRIMARY KEY,
    applied_at timestamp with time zone DEFAULT now()
);