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

## Worksheet generation / admin scripts

Worksheets are generated in v2 (`api-service/app/generation/`, plan and validation in
`docs/PHASE9_GENERATION.md`): one command composes the questions, inserts them, and writes the
printable PDFs, so a printed sheet always carries an id the database holds. From `api-service/`,
venv activated (step-by-step usage in `docs/ADMIN_HOWTO.md`):

```bash
python3 scripts/seed_skills.py                                                  # once, before the first batch
python3 scripts/generate_worksheets.py --type homework --level A --language mr --count 50 --merge
python3 scripts/generate_worksheets.py --type omr --start-id 5001 --count 100
python3 scripts/render_worksheet_pdf.py --id 4920                               # reprint from the DB
python3 scripts/import_students_from_csv.py students.csv --school-code PSV
python3 scripts/insert_code_answer_key.py --code D --answer-key A,B,C,D,...
```

The old repo's `batch_generate_worksheets.py`/`worksheet_json_generator.py`/`worksheet_pdf_generator.py`
are superseded. With the same seed, the port produces identical questions, and its PDFs are
pixel-identical to the old renderer's. The JSON-file insertion scripts (`insert_single_worksheet.py`,
`bulk_insert_worksheets.py`) remain for JSON files that already exist.

## Generating the first Alembic migration

The `alembic/versions/` folder is intentionally empty. Once a Postgres instance is reachable and the models in `app/models/` are finalized, generate the baseline migration with:

```bash
cd api-service
alembic revision --autogenerate -m "baseline schema"
alembic upgrade head
```
