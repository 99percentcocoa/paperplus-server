from fastapi import FastAPI

from app.routes import dashboard, webhook

app = FastAPI(title="paperplus-api-service")

app.include_router(webhook.router)
app.include_router(dashboard.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
