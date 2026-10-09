# PaperPlus

PaperPlus grades paper worksheets that students photograph and send over WhatsApp. A Community
Facilitator (CF) visits rural primary schools, runs foundational-numeracy practice sessions, and
hands out printed multiple-choice worksheets as homework. Each sheet carries AprilTag markers that
encode its worksheet ID. When a photo of a completed sheet arrives, PaperPlus finds the page,
reads the student's roll number and the marked bubbles, grades them against the worksheet's answer
key, updates the student's skill mastery and level, and replies on WhatsApp with an annotated
"checked" image. A web dashboard shows submissions, weekly metrics, system health, and failed
scans that need a person to fix them.

## Architecture

```
 WhatsApp (Exotel) ──▶ nginx ──▶ api-service ──── internal HTTP ────▶ vision-service
                                 (FastAPI)       photo in (base64),   (FastAPI, stateless)
                                 - webhook        marks + ROIs out    - AprilTag detect + dewarp
                                 - grading, mastery                    - row-tag ID/page decode
                                 - admin dashboard                     - OCR (PaddleOCR)
                                 - worksheet generation                - bubble classifier (TFLite)
                                 - SQLModel + Alembic                  - no DB, no disk writes
                                       │
                                   Postgres
```

- **api-service** owns everything with state: the webhook, the database, grading, mastery,
  the admin dashboard, and worksheet generation (questions, answer keys, printable PDFs).
- **vision-service** only turns a photo into raw marks. It has no database and writes nothing to
  disk, so it can move to its own machine (or a cloud GPU) without code changes. It doesn't know
  answer keys, so it never grades.
- **shared/** holds what both services must agree on: the `/process` request/response contract,
  the row-tag encoder/decoder (the printer and the scanner use the same code), per-template page
  sizes, and logging setup (one correlation ID traces a scan across both services).

Calls are synchronous: there is no task queue. Every received photo is stored with its full
vision result, so a failed scan can be corrected from the dashboard without rescanning.

## Repository layout

```
api-service/      FastAPI app, models, migrations (alembic/), scripts/, tests/
  app/generation/   worksheet question + PDF generation
  app/static/admin/ dashboard (static HTML/JS, no build step)
  assets/           fonts, print templates and tags for generated worksheets
vision-service/   FastAPI app, models/ (TFLite), tests/ (incl. real-scan accuracy suite)
shared/           code imported by both services
infra/            docker-compose.yml, nginx.conf, .env.example
docs/             ADMIN_HOWTO.md (day-to-day tasks), PROGRESS.md (history/decisions),
                  PHASE9_GENERATION.md (worksheet generation)
```

## Run it with Docker

```bash
cd infra
cp .env.example .env            # set secrets; LOCAL_MODE=true logs WhatsApp replies instead of sending
docker compose up -d --build
docker compose run --rm api-service alembic upgrade head
docker compose run --rm api-service python scripts/seed_skills.py
```

nginx serves on `http://localhost:8080` (`NGINX_HOST_PORT`). The dashboard is at
`/admin/<project>/` (`paperplus` or `navodaya`) and the WhatsApp webhook at `/webhook`.
vision-service is reachable only inside the compose network.

## Local development

Python 3.10+ and a local Postgres. Each service has its own venv.

```bash
# api-service
cd api-service
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # WeasyPrint also needs Pango (apt: libpango-1.0-0 libpangoft2-1.0-0)
echo 'DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/paperplus_v2' > .env
alembic upgrade head
python scripts/seed_skills.py
uvicorn app.main:app --reload --port 8000

# vision-service (separate shell)
cd vision-service
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # PaddleOCR models download on first start
uvicorn app.main:app --reload --port 8100
```

Settings come from environment variables or each service's `.env` (see `app/core/config.py`).
The two services' shared secret (`VISION_SERVICE_SHARED_SECRET` in api-service, `SHARED_SECRET` in
vision-service) must match.

### Copy production data to dev

`infra/sync-prod-to-dev.sh` replaces the dev database with a copy of prod (one way: prod is only
read), so you can try changes against real data. One-time setup: create `infra/.env.sync`
(git-ignored) containing `PROD_SSH=user@prod-host`. Your SSH user must be able to run `docker`
on the server; if it needs sudo, also set `PROD_DOCKER="sudo docker"`.

```bash
infra/sync-prod-to-dev.sh                 # dump prod, back up + replace the dev DB, run migrations
infra/sync-prod-to-dev.sh --with-files    # also copy prod's scan and checked images
infra/sync-prod-to-dev.sh --from-file db_snapshots/prod_<timestamp>.sql.gz   # reload a saved dump, no SSH
```

- **Safety checks:** it refuses unless the dev database is on this machine and is not the prod
  server (this catches an SSH tunnel to prod). It also requires `LOCAL_MODE=true` in
  `api-service/.env`, because prod data has real phone numbers.
- **Snapshots:** prod dumps and a backup of the replaced dev database go to `db_snapshots/`
  (git-ignored, newest 5 kept). These contain real student data, so keep them on this machine.
- **Tests:** the dev database's existing contents are replaced, and api-service tests then run
  against prod-shaped data.

## Everyday tasks

Full steps are in [docs/ADMIN_HOWTO.md](docs/ADMIN_HOWTO.md). Scripts run from `api-service/`:

```bash
python scripts/generate_worksheets.py --type homework --level A --language mr --count 50 --merge
python scripts/generate_worksheets.py --type omr --start-id 5001 --count 100
python scripts/render_worksheet_pdf.py --id 4920                 # reprint from the database
python scripts/import_students_from_csv.py students.csv --school-code KGP [--project navodaya]
python scripts/insert_code_answer_key.py --code D --answer-key A,B,C,D,...
curl --data-binary @scan.jpg "<host>/api/admin/projects/paperplus/scans?roll_number=0151"
```

## Tests

```bash
cd api-service && .venv/bin/python -m pytest      # runs against the dev database in .env
cd vision-service && .venv/bin/python -m pytest   # includes real-scan accuracy suite (~1 min)
```

The api-service tests use the real dev database, not an isolated one, and clean up after
themselves. Don't point `DATABASE_URL` at production when running them.

## Design rules

These come from the original redevelopment guidelines:

- Keep image processing (vision-service) separate from business logic and UI.
- Everything runs in Docker; migrations are Alembic; models are SQLModel.
- Store prediction confidences, vision results and processing events in the database.
- Submissions move through explicit processing states via transition functions
  (`app/domain/submission_state.py`), never by setting the state directly.
- Keep a golden set of real scans with ground truth (`vision-service/tests/fixtures/`) and keep
  the accuracy suite passing.
- The webhook is testable with mock payloads (`api-service/tests/test_webhook.py`).
- Keep commits small, and keep this README and `docs/` current.

## Status

The rewrite is feature-complete: scanning, grading, dashboard, projects, monitoring and worksheet
generation. `docs/PROGRESS.md` records what was built, the bugs found along the way, and known
limitations. Still open: production cutover from the pre-v2 database
(`scripts/migrate_legacy_data.py`), and authentication for the admin dashboard and API, which are
currently open to anyone with the URL.
