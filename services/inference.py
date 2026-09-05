
import logging
from pathlib import Path

from models import InputImageMeta

import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

logger = logging.getLogger(__name__)

interpreter = None
input_details = None
output_details = None

# def init_interpreter(model_path=Path(__file__).parent / "bubble_model_quantized.tflite"):
def init_interpreter(model_path=Path(__file__).parent / "blur_model_optimized.tflite"):
    global interpreter, input_details, output_details
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

def preprocess_image(input_image: InputImageMeta, target_size=128) -> InputImageMeta:
    """
    Preprocess the input image for bubble classification.

    Args:
        input_image (InputImageMeta): The input image metadata for the bubble image.
        target_size (int): The target size for the model input (default is 128).

    Returns:
        InputImageMeta: The preprocessed image metadata ready for model input.
    """

    img = input_image.image_array
    if img is None:
        raise ValueError("Input image is empty; cannot preprocess.")
    img = np.ascontiguousarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape

    # --- Resize while preserving aspect ratio ---
    scale = max(target_size / h, target_size / w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    img = cv2.resize(img, (new_w, new_h))

    # --- Center crop ---
    start_x = (new_w - target_size) // 2
    start_y = (new_h - target_size) // 2

    img = img[start_y:start_y+target_size, start_x:start_x+target_size]

    # Convert to float
    img = img.astype(np.float32, copy=False)

    # --- MobileNetV2 preprocessing ---
    # img = (img / 127.5) - 1

    img = np.expand_dims(img, axis=0)
    img_meta = InputImageMeta(image_array=img)

    return img_meta

def preprocess_image_PIL(input_image: InputImageMeta, target_size=128) -> InputImageMeta:
    img = Image.fromarray(input_image.image_array)
    img = img.resize((target_size, target_size))
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_meta = InputImageMeta(image_array=img_array)
    return img_meta

def predict_bubble(input_bubble: InputImageMeta):
    """
    Predict whether the bubble is marked or unmarked using the TFLite model.

    Args:
        input_bubble (InputImageMeta): The input image metadata for the bubble image.
    
    Returns:
        Tuple[float, str, float, float]: A tuple containing the probability, result label ("Marked" or "Unmarked"), and confidence percentage.
    """

    if interpreter is None or input_details is None or output_details is None:
        init_interpreter()
    # preprocessed_img = preprocess_image(input_bubble, target_size=128)
    preprocessed_img = preprocess_image_PIL(input_bubble, target_size=128)

    interpreter.set_tensor(input_details[0]['index'], preprocessed_img.image_array)
    interpreter.invoke()
    
    # 6. Get Result
    output = interpreter.get_tensor(output_details[0]['index'])
    logit = output[0][0]
    probability = 1.0 / (1.0 + np.exp(-logit))

    # print("Logit:", logit)
    # print("Probability:", probability)
    
    if probability < 0.5:
        result = "Marked"
        confidence = (1 - probability) * 100
    else:
        result = "Unmarked"
        confidence = probability * 100

    return probability.item(), result, confidence, probability

# -- OCR --

def init_ocr():
    global ocr
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv6_small_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )

def predict_ocr(input_image: InputImageMeta):
    """
    Perform OCR on the input image using PaddleOCR.

    Handwritten single-character fields can fail on the raw image while succeeding
    on a thresholded variant. We try the standard path first and then a small set
    of normalized fallback variants so that the ROI can still be recognized.
    """

    if 'ocr' not in globals():
        init_ocr()

    img = input_image.image_array
    if img is None:
        raise ValueError("Input image is empty; cannot perform OCR.")

    def _extract_best_text(result):
        try:
            res = result[0]
            rec_texts = res.get("rec_texts", [])
            rec_scores = res.get("rec_scores", [])
        except (IndexError, TypeError, AttributeError):
            return ""

        if rec_texts and rec_scores:
            max_score_index = int(np.argmax(rec_scores))
            return str(rec_texts[max_score_index]).strip()
        return ""

    def _prepare_variants(image):
        variants = []
        seen = set()

        if len(image.shape) == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            variants.append(rgb)
            variants.append(gray)
        else:
            gray = image
            variants.append(gray)

        if len(gray.shape) == 2:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                10,
            )
            variants.extend([
                cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB),
                cv2.cvtColor(cv2.bitwise_not(thresh), cv2.COLOR_GRAY2RGB),
            ])

        ordered = []
        for variant in variants:
            key = (variant.shape, variant.dtype, variant.tobytes()[:64])
            if key in seen:
                continue
            seen.add(key)
            ordered.append(variant)
        return ordered

    for variant in _prepare_variants(img):
        try:
            result = ocr.predict(variant)
        except RuntimeError as exc:
            logger.warning("OCR runtime failure on input image: %s", exc)
            continue
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("OCR prediction failed unexpectedly: %s", exc)
            continue

        text = _extract_best_text(result)
        if text:
            return text

    return ""