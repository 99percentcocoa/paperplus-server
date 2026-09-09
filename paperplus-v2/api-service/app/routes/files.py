"""Serves generated artifacts (currently just checked images) over HTTP, replacing the old
system's Flask static route (routes/file_routes.py: GET /checked/<filename>). Needed because
Exotel's WhatsApp API and the Sheets logging webhook both require a fetchable URL, not a local
path -- see app.core.config.settings.public_base_url for how that URL is built.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings

router = APIRouter(prefix="/files")


@router.get("/checked/{filename}")
def get_checked_image(filename: str) -> FileResponse:
    # Reject path traversal (e.g. "../../etc/passwd") -- filename must be a bare name.
    if "/" in filename or "\\" in filename or filename in (".", ".."):
        raise HTTPException(status_code=400, detail="invalid filename")

    path = Path(settings.storage_root) / "checked" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not found")

    return FileResponse(path, media_type="image/jpeg")


def checked_image_url(filename: str) -> str:
    return f"{settings.public_base_url}/files/checked/{filename}"
