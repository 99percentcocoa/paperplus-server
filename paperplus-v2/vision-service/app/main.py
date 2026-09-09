"""
vision-service: stateless image processing (AprilTag/dewarp, OCR, bubble inference).
"""

from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, Header, HTTPException

from app.core.config import settings
from app.vision.bubble_inference import BubbleClassifier
from app.vision.errors import CornerTagDetectionError, RollNumberError, RowTagDetectionError
from app.vision.ocr import PaddleOCRProvider
from app.vision.pipeline import process_scan
from shared.contracts import ProcessingResult, ProcessRequest, QuestionMark


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
    _require_shared_secret(x_service_secret)

    image_array = cv2.imread(request.image_path)
    if image_array is None:
        raise HTTPException(status_code=400, detail=f"could not read image at {request.image_path}")

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
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    dewarped_path = _save_artifact(result.dewarped_image_array, request.correlation_id, "dewarped")
    debug_path = _save_artifact(result.debug_image_array, request.correlation_id, "debug")

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
        dewarped_image_path=dewarped_path,
        debug_image_path=debug_path,
    )


def _save_artifact(image_array, correlation_id: str, role: str) -> str | None:
    if image_array is None:
        return None
    output_dir = Path(settings.storage_root) / role
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{correlation_id}_{role}.jpg"
    cv2.imwrite(str(output_path), image_array)
    return str(output_path)

