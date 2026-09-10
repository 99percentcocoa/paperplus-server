from fastapi import FastAPI

from shared.logging_config import configure_logging

configure_logging("api-service")

from app.routes import dashboard, files, webhook  # noqa: E402 - must follow configure_logging()

app = FastAPI(title="paperplus-api-service")

app.include_router(webhook.router)
app.include_router(dashboard.router)
app.include_router(files.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
