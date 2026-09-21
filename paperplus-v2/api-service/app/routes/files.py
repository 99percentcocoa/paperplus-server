"""Serves generated artifacts over HTTP, replacing the old system's Flask static route
(routes/file_routes.py: GET /checked/<filename>). Checked images need a fetchable URL because
Exotel's WhatsApp API and the Sheets logging webhook both require one, not a local path -- see
app.core.config.settings.public_base_url. uploads and checked images are served for the admin
dashboard, which shows the original scan next to its annotated version.
"""

import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings

router = APIRouter(prefix="/files")

# URL kind -> subdirectory under storage_root. The dewarped page is stored (checked images are
# drawn on it) but deliberately not served: nothing needs to display it.
SERVED_KINDS = {"uploads", "checked"}


@router.get("/{kind}/{filename}")
def get_artifact(kind: str, filename: str) -> FileResponse:
    if kind not in SERVED_KINDS:
        raise HTTPException(status_code=404, detail="not found")
    # Reject path traversal (e.g. "../../etc/passwd") -- filename must be a bare name.
    if "/" in filename or "\\" in filename or filename in (".", ".."):
        raise HTTPException(status_code=400, detail="invalid filename")

    path = Path(settings.storage_root) / kind / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not found")

    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)


def checked_image_url(filename: str) -> str:
    return f"{settings.public_base_url}/files/checked/{filename}"


def relative_artifact_url(kind: str, stored_path: str | None) -> str | None:
    """Same-origin URL for a stored artifact path (works behind nginx or on any host), or None."""
    if not stored_path:
        return None
    return f"/files/{kind}/{Path(stored_path).name}"
