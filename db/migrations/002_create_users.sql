
-- users
CREATE TABLE IF NOT EXISTS users (
    user_id serial PRIMARY KEY,
    user_name text NOT NULL,
    from_number text NOT NULL UNIQUE
)