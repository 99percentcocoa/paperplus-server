from fastapi import FastAPI

from app.routes import dashboard, files, webhook

app = FastAPI(title="paperplus-api-service")

app.include_router(webhook.router)
app.include_router(dashboard.router)
app.include_router(files.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
