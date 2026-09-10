# Build context is paperplus-v2/ (the monorepo root), not api-service/ itself -- see
# infra/docker-compose.yml's `context: ..` -- so this Dockerfile can also COPY the sibling
# shared/ package (imported at runtime as `shared.contracts`/`shared.logging_config`), which
# would otherwise be physically outside a context scoped to api-service/ alone.
FROM python:3.11-slim

WORKDIR /app

COPY api-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api-service/ .
COPY shared/ ./shared

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
