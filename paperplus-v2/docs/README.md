# PaperPlus v2

Redevelopment of the PaperPlus OMR grading system as two independently deployable services.
See [/testing/redevelopment.md](../../testing/redevelopment.md) in the parent repo for the original guidelines driving this rewrite.

## Architecture

```
                    ┌───────────┐
 WhatsApp/Exotel ──▶│  nginx    │
                    └─────┬─────┘
                          │
                    ┌─────▼─────────┐        internal HTTP        ┌───────────────────┐
                    │  api-service   │ ───────────────────────────▶│  vision-service    │
                    │ (FastAPI)      │◀───────────────────────────│ (FastAPI, ML/CV)   │
                    │ - webhooks     │        JSON result           │ - AprilTag/dewarp  │
                    │ - worksheets   │                              │ - OCR (PaddleOCR)  │
                    │ - students     │                              │ - bubble inference │
                    │ - dashboard    │                              │ - preloaded models │
                    │ - SQLModel ORM │                              └────────┬───────────┘
                    └───────┬────────┘                                       │
                            │                                          shared volume
                       ┌────▼─────┐                                   (images, debug)
                       │ Postgres │
                       └──────────┘
```

- **api-service**: business logic, Postgres (SQLModel + Alembic), webhooks, worksheet/admin CLI, dashboard.
- **vision-service**: stateless image processing (AprilTag detection/dewarp, OCR, bubble inference). No DB access; called synchronously by api-service over internal HTTP. Kept as a separate service from day one so it can move to its own machine later without a rewrite.
- **shared**: pydantic contracts used by both services for the `/process` request/response schema.

## Status

Scaffolding in progress. See `docs/PROGRESS.md` for what's implemented vs pending.

## Local development quickstart

```bash
# api-service
cd api-service
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/paperplus_v2
alembic upgrade head   # once a Postgres instance is reachable
python3 scripts/seed_skills.py  # required once before inserting any worksheet (questions.skill_code FK)
uvicorn app.main:app --reload --port 8000

# vision-service
cd vision-service
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8100
```

## Worksheet insertion / admin scripts

Worksheet PDF/JSON *generation* is not ported to v2 — use the old repo's `worksheet_json_generator.py`/`batch_generate_worksheets.py` for that. Once JSON exists, insert it and manage students/answer-key variants with (from `api-service/`, venv activated):

```bash
python3 scripts/seed_skills.py                                             # once, before any worksheet insert
python3 scripts/insert_single_worksheet.py --json-file 8001_en.json --type homework
python3 scripts/bulk_insert_worksheets.py --json-dir ../../files/json      # or --dry-run first
python3 scripts/import_students_from_csv.py students.csv --school-code PSV
python3 scripts/insert_question_paper_variant.py --worksheet-id 9001 --code A --answer-key A,B,C,D,...
```

## Generating the first Alembic migration

The `alembic/versions/` folder is intentionally empty. Once a Postgres instance is reachable and the models in `app/models/` are finalized, generate the baseline migration with:

```bash
cd api-service
alembic revision --autogenerate -m "baseline schema"
alembic upgrade head
```
