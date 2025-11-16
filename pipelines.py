import tags
import cv2
import apriltags
import image

def crop_image(fp):
    detection = apriltags.detect_tags_36h11(fp)
    print(f"tags detected: {[d.tag_id for d in detection]}")

    # arrange clockwise
    detection = tags.sort_detections_clockwise(detection)
    print(f"clockwise sorted: {[d.tag_id for d in detection]}")

    wid, detection = tags.detect_orientation_and_decode(detection)
    print(f"detected wid {wid}")

    cropped = image.dewarp_omr(fp, detection)

    return cropped