"""
vision-service: stateless image processing (AprilTag/dewarp, OCR, bubble inference).
"""

import base64
import binascii
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, Header, HTTPException

from app.core.config import settings
from app.vision.bubble_inference import BubbleClassifier
from app.vision.errors import CornerTagDetectionError, RollNumberError, RowTagDetectionError
from app.vision.ocr import PaddleOCRProvider
from app.vision.pipeline import process_scan
from shared.contracts import ProcessingResult, ProcessRequest, QuestionMark
from shared.logging_config import configure_logging, correlation_id_var

configure_logging("vision-service")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # NOTE: the old system's active bubble-marking model is blur_model_optimized.tflite —
    # bubble_model_quantized.tflite is loaded but unused there (its call site is commented out).
    # Preloaded once here (not lazily per-request) to avoid cold-start latency.
    app.state.bubble_classifier = BubbleClassifier(settings.blur_model_path)
    app.state.ocr_provider = PaddleOCRProvider()
    yield


app = FastAPI(title="paperplus-vision-service", lifespan=lifespan)


def _require_shared_secret(x_service_secret: str | None) -> None:
    if x_service_secret != settings.shared_secret:
        raise HTTPException(status_code=401, detail="invalid service secret")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/process", response_model=ProcessingResult)
def process(request: ProcessRequest, x_service_secret: str | None = Header(default=None)) -> ProcessingResult:
    correlation_id_var.set(request.correlation_id)
    _require_shared_secret(x_service_secret)

    logger.info("Processing scan template_hint=%s", request.template_hint)

    image_array = _decode_image(request.image_b64)
    if image_array is None:
        logger.error("Request image_b64 is not a decodable image")
        raise HTTPException(status_code=400, detail="image_b64 is not a decodable image")

    try:
        result = process_scan(
            image_array,
            target_width=settings.target_width,
            target_height=settings.target_height,
            bubble_classifier=app.state.bubble_classifier,
            ocr_provider=app.state.ocr_provider,
            template_hint=request.template_hint,
        )
    except (CornerTagDetectionError, RowTagDetectionError, RollNumberError) as exc:
        logger.warning("Scan processing failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.info(
        "Scan processed: worksheet_id=%s page_no=%s template_name=%s roll_number=%s "
        "question_paper_code=%s question_marks_count=%s",
        result.worksheet_id, result.page_no, result.template_name, result.roll_number,
        result.question_paper_code, len(result.question_marks),
    )

    return ProcessingResult(
        worksheet_id=result.worksheet_id,
        page_no=result.page_no,
        first_question_index=result.first_question_index,
        template_name=result.template_name,
        roll_number=result.roll_number,
        roll_number_confidence=result.roll_number_confidence,
        question_paper_code=result.question_paper_code,
        question_marks=[
            QuestionMark(
                question_index=m.question_index,
                marked_option=m.marked_option,
                confidence=m.confidence,
                roi_x1=m.roi_x1,
                roi_y1=m.roi_y1,
                roi_x2=m.roi_x2,
                roi_y2=m.roi_y2,
            )
            for m in result.question_marks
        ],
        dewarped_image_b64=_encode_image(result.dewarped_image_array),
    )


def _decode_image(image_b64: str):
    """base64 photo bytes -> BGR array, or None if it isn't valid base64 / a decodable image."""
    try:
        data = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError):
        return None
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _encode_image(image_array) -> str | None:
    """BGR array -> base64 JPEG. vision-service writes nothing to disk; the dewarped page goes back
    in the response because the ROI boxes it returns are in that image's coordinates."""
    if image_array is None:
        return None
    ok, buffer = cv2.imencode(".jpg", image_array)
    if not ok:
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")
