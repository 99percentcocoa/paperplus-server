#!/usr/bin/env python3
"""Run one worksheet image through the full pipeline (vision-service call -> grading ->
DB persistence -> mastery/level update) without WhatsApp/Exotel, for local testing.

Requires vision-service running (see vision-service/README) and LOCAL_MODE=true in
api-service/.env (send_message logs instead of calling Exotel).

Example:
    python3 scripts/test_local_image.py /path/to/scan.jpeg
    python3 scripts/test_local_image.py /path/to/scan.jpeg --from-number +919999999999
"""

import argparse
import logging
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlmodel import Session

from app.core.config import settings
from app.db.session import engine
from app.services.communication import ExotelCommunicationClient
from app.services.submission_service import handle_incoming_image
from app.services.vision_client import HTTPVisionClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one image through the full grading pipeline locally.")
    parser.add_argument("image_path", help="Path to a worksheet scan image.")
    parser.add_argument("--from-number", default="+910000000001", help="Fake WhatsApp sender number to attribute the submission to.")
    args = parser.parse_args()

    image_path = Path(args.image_path).resolve()
    if not image_path.exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    if not settings.local_mode:
        print("Warning: LOCAL_MODE is not true -- this will attempt a real Exotel send.", file=sys.stderr)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    correlation_id = str(uuid.uuid4())
    vision_client = HTTPVisionClient()
    comm_client = ExotelCommunicationClient()

    with Session(engine) as session:
        handle_incoming_image(session, vision_client, comm_client, args.from_number, str(image_path), correlation_id)

    print(f"\ncorrelation_id={correlation_id} -- query the DB for this image's submission (see docs/README.md).")


if __name__ == "__main__":
    main()
