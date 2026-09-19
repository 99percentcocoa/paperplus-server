from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from shared.logging_config import configure_logging

configure_logging("api-service")

from app.routes import admin, dashboard, files, webhook  # noqa: E402 - must follow configure_logging()

app = FastAPI(title="paperplus-api-service")

app.include_router(webhook.router)
app.include_router(dashboard.router)
app.include_router(admin.router)
app.include_router(files.router)

ADMIN_STATIC_DIR = Path(__file__).resolve().parent / "static" / "admin"


@app.middleware("http")
async def noindex_admin(request: Request, call_next):
    """The admin UI is unauthenticated (URL is simply not shared), so at least keep it out of
    search engines."""
    response = await call_next(request)
    if request.url.path.startswith(("/admin", "/api/admin", "/files")):
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/admin", include_in_schema=False)
def admin_redirect() -> RedirectResponse:
    return RedirectResponse(url="/admin/")


app.mount("/admin", StaticFiles(directory=ADMIN_STATIC_DIR, html=True), name="admin")
