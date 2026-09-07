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
uvicorn app.main:app --reload --port 8000

# vision-service
cd vision-service
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8100
```

## Generating the first Alembic migration

The `alembic/versions/` folder is intentionally empty. Once a Postgres instance is reachable and the models in `app/models/` are finalized, generate the baseline migration with:

```bash
cd api-service
alembic revision --autogenerate -m "baseline schema"
alembic upgrade head
```
