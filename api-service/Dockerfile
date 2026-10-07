# Build context is the repo root, not api-service/ itself -- see
# infra/docker-compose.yml's `context: ..` -- so this Dockerfile can also COPY the sibling
# shared/ package (imported at runtime as `shared.contracts`/`shared.logging_config`), which
# would otherwise be physically outside a context scoped to api-service/ alone.
FROM python:3.11-slim

WORKDIR /app

# WeasyPrint (worksheet PDF generation) needs Pango. The printed sheets use DejaVu Serif/Sans Mono
# for Latin text and Noto Sans Devanagari for Marathi numerals (named in the templates' style.css);
# Noto Devanagari is bundled in the repo so the image doesn't need the ~100 MB fonts-noto-core.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpango-1.0-0 libpangoft2-1.0-0 fonts-dejavu-core fontconfig \
    && rm -rf /var/lib/apt/lists/*
COPY api-service/assets/worksheets/fonts/ /usr/share/fonts/truetype/paperplus/
RUN fc-cache -f

COPY api-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api-service/ .
COPY shared/ ./shared

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
