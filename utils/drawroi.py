# draw rois on a preprocessed image

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import SETTINGS
import services.image_service as img_service

import cv2
import numpy as np

def get_anchor_points(image):

    tags_25h9 = img_service.detect_tags_25h9(image)
    if not tags_25h9:
        raise ValueError("No 25h9 tags detected in the image.")
    
    return [t.center for t in tags_25h9]

def draw_rois(output_image, anchor):

    rois = [SETTINGS.LEFT_QUESTION_ROI, SETTINGS.RIGHT_QUESTION_ROI]

    for roi in rois:
        (rx, ry, rw, rh) = roi
        (anchor_x, anchor_y) = anchor

        x1 = int(anchor_x + rx)
        y1 = int(anchor_y + ry)
        x2 = int(x1 + rw)
        y2 = int(y1 + rh)

        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    return output_image

if __name__ == "__main__":

    input_image_path = "images/preprocessed_cropped_thick1.jpg"
    input_image = cv2.imread(input_image_path)
    output_image = input_image.copy()

    output_image_path = "images/rois_preprocessed_cropped_thick1.jpg"

    anchor_points = get_anchor_points(input_image)
    if not anchor_points:
        raise ValueError("No anchor points found.")
    
    for anchor_point in anchor_points:
        output_image = draw_rois(output_image, anchor_point)
    
    cv2.imwrite(output_image_path, output_image)
    
    print(f"ROIs drawn and saved to {output_image_path}")