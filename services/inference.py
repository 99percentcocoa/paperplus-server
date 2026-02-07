
from pathlib import Path

from ker_ocr.models import InputImageMeta

import numpy as np
import cv2
from ai_edge_litert.interpreter import Interpreter

interpreter = None
input_details = None
output_details = None

def init_interpreter(model_path=Path(__file__).parent / "bubble_model_quantized.tflite"):
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

def predict_bubble(input_bubble: InputImageMeta):
    """
    Predict whether the bubble is marked or unmarked using the TFLite model.

    Args:
        input_bubble (InputImageMeta): The input image metadata for the bubble image.
    
    Returns:
        Tuple[float, str, float]: A tuple containing the probability, result label ("Marked" or "Unmarked"), and confidence percentage.
    """

    if interpreter is None or input_details is None or output_details is None:
        init_interpreter()
    preprocessed_img = preprocess_image(input_bubble, target_size=128)

    interpreter.set_tensor(input_details[0]['index'], preprocessed_img.image_array)
    interpreter.invoke()
    
    # 6. Get Result
    output = interpreter.get_tensor(output_details[0]['index'])
    logit = output[0][0]
    probability = 1 / (1 + np.exp(-logit))

    # print("Logit:", logit)
    # print("Probability:", probability)
    
    if probability < 0.5:
        result = "Marked"
        confidence = (1 - probability) * 100
    else:
        result = "Unmarked"
        confidence = probability * 100

    return probability.item(), result, confidence