import cv2
from pupil_apriltags import Detector

at_detector = Detector(
    families="tag36h11",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.2,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

def detect_tags(filepath):
    img = cv2.imread(filepath)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detection = at_detector.detect(gray_img)
    # detected_tags = list(map(lambda x:x.tag_id, detection))
    return detection

if __name__ == "__main__":
    print(detect_tags('testaprilfull.jpg'))