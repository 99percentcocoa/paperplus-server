import cv2
import numpy as np

TARGET_WIDTH = 1240
TARGET_HEIGHT = 1754

# dewarping using apriltags. Note that "tags" should be the corner tags (36h11)
def dewarp_omr(filepath, tags):
    image = cv2.imread(filepath)
    
    if len(tags) != 4:
        print("Less/more than 4 tags detected.")
        return []
    
    corner_tags = {d.tag_id: d.center for d in tags}

    tl = corner_tags[1]
    tr = corner_tags[2]
    bl = corner_tags[3]
    br = corner_tags[4]
    
    # Re-order the final source points: TL, TR, BR, BL
    # This is the essential input for cv2.getPerspectiveTransform
    src_pts_aligned = np.array([tl, tr, br, bl], dtype="float32")

    dst_pts = np.array([
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1]], dtype="float32")

    # Calculate the global perspective transform matrix (M)
    M = cv2.getPerspectiveTransform(src_pts_aligned, dst_pts)

    # Apply the dewarping
    dewarped = cv2.warpPerspective(image, M, (TARGET_WIDTH, TARGET_HEIGHT))
    return dewarped

# split image into halves
def split_img(image):
    # image = cv2.imread(filepath)
    height, width = image.shape[:2]

    mid_x = width // 2

    left_half = image[:, :mid_x]
    right_half = image[:, mid_x:]

    return [left_half, right_half]

    # cv2.imwrite(f"{filepath}_left.jpg", left_half)
    # cv2.imwrite(f"{filepath}_right.jpg", right_half)

# preprocessing pipeline designed for a scanned sheet
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35,35))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    diff = cv2.absdiff(gray, background)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    norm_inv = cv2.bitwise_not(norm)

    # convert back to BGR
    color_img = cv2.cvtColor(norm_inv, cv2.COLOR_GRAY2BGR)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # enhanced = clahe.apply(norm_inv)
    # cv2.imwrite('clean_enhanced.jpg', enhanced)

    # median = cv2.medianBlur(enhanced, 3)
    # cv2.imwrite('clean_median.jpg', median)
    # bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
    # cv2.imwrite('clean_bilateral.jpg', bilateral)

    # cv2.imwrite("clean_preprocessed.jpg", color_img)
    return color_img

if __name__ == "__main__":
    dewarp_omr("2col.jpg_dewarped.jpg")