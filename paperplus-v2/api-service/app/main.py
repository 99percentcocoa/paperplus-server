from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session

from shared.logging_config import configure_logging

configure_logging("api-service")

from app.db.session import get_session  # noqa: E402
from app.models import Project  # noqa: E402
from app.models.core import DEFAULT_PROJECT_CODE  # noqa: E402
from app.routes import admin, dashboard, files, webhook  # noqa: E402 - must follow configure_logging()

app = FastAPI(title="paperplus-api-service")

app.include_router(webhook.router)
app.include_router(dashboard.router)
app.include_router(admin.router)
app.include_router(admin.project_router)
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
    return RedirectResponse(url=f"/admin/{DEFAULT_PROJECT_CODE}/")


# One dashboard per project, all the same static app: /admin/<code>/ is that project's dashboard
# (app.js reads the code from the path), and that URL is what gets shared with the project's
# users -- there's deliberately no page listing the projects. Mounted before the
# /admin/{project_code}/ route so "assets" is never taken for a project code.
app.mount("/admin/assets", StaticFiles(directory=ADMIN_STATIC_DIR), name="admin-assets")


@app.get("/admin/", include_in_schema=False)
def admin_default_project() -> RedirectResponse:
    """Pre-projects bookmarks pointed at /admin/, which was the PaperPlus dashboard."""
    return RedirectResponse(url=f"/admin/{DEFAULT_PROJECT_CODE}/")


def _require_project(project_code: str, session: Session) -> None:
    if session.get(Project, project_code) is None:
        raise HTTPException(status_code=404, detail="project not found")


@app.get("/admin/{project_code}", include_in_schema=False)
def admin_project_redirect(project_code: str, session: Session = Depends(get_session)) -> RedirectResponse:
    _require_project(project_code, session)
    return RedirectResponse(url=f"/admin/{project_code}/")


@app.get("/admin/{project_code}/", include_in_schema=False)
def admin_project_dashboard(project_code: str, session: Session = Depends(get_session)) -> FileResponse:
    _require_project(project_code, session)
    return FileResponse(ADMIN_STATIC_DIR / "index.html")
