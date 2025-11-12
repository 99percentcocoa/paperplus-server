import cv2
from pupil_apriltags import Detector
import numpy as np
import logging

logger = logging.getLogger(__name__)

at_detector_36h11 = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.2,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

at_detector_25h9 = Detector(
    families="tag25h9",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.2,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

def detect_tags_36h11(image_input):
    # case 1: image input is a file path
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    
    # case 2: image input is an opencv image array
    elif isinstance(image_input, np.ndarray):
        img = image_input
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detection = at_detector_36h11.detect(gray_img)
    # detected_tags = list(map(lambda x:x.tag_id, detection))
    return detection

def detect_tags_25h9(image_input):

    # case 1: image input is a file path
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    
    # case 2: image input is an opencv image array
    elif isinstance(image_input, np.ndarray):
        img = image_input

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detection = at_detector_25h9.detect(gray_img)
    # detected_tags = list(map(lambda x:x.tag_id, detection))
    return detection

if __name__ == "__main__":
    print(detect_tags_36h11('testing/testaprilfull.jpg'))