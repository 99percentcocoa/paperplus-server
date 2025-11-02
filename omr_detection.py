import cv2
import numpy as np
import image
import apriltags

# roi format: (x_offset, y_offset, width, height)
LEFT_QUESTION_ROI = (75, -40, 475, 85)
RIGHT_QUESTION_ROI = (602, -40, 475, 85)

MIN_MARK_AREA = 400 # TUNE THIS if needed
MAX_MARK_AREA = 1000
FILL_THRESHOLD = 0.9

# aspect ratios of the bounding boxes of potential contour candidates
MIN_CNT_ASPECT_RATIO = 0.7
MAX_CNT_ASPECT_RATIO = 1.3

# uses globally defined LEFT_QUESTION_ROI and RIGHT_QUESTION_ROI
def show_roi_zones(image, points, debug_image):
    # preprocessed = image.preprocess(image)
    # cv2.imwrite('debug_preprocessed.jpg', preprocessed)

    for i, point in enumerate(points):
        (point_x, point_y) = point

        # draw left ROI
        (left_rx, left_ry, left_rw, left_rh) = LEFT_QUESTION_ROI
        left_x1 = point_x + left_rx
        left_y1 = point_y + left_ry
        left_x2 = left_x1 + left_rw
        left_y2 = left_y1 + left_rh

        # draw red rectangle on left ROI
        cv2.rectangle(debug_image, (left_x1, left_y1), (left_x2, left_y2), (255, 0, 0), 2)

        # draw right ROI
        (right_rx, right_ry, right_rw, right_rh) = RIGHT_QUESTION_ROI
        right_x1 = point_x + right_rx
        right_y1 = point_y + right_ry
        right_x2 = right_x1 + right_rw
        right_y2 = right_y1 + right_rh

        # draw red rectangle on right ROI
        cv2.rectangle(debug_image, (right_x1, right_y1), (right_x2, right_y2), (255, 0, 0), 2)
    
    # save image
    # cv2.imwrite('debug_roi.jpg', debug_image)

# function to detect filled bubble given anchor point and roi (either LEFT or RIGHT)
# will return: 'A', 'B', 'C', 'D' or '' (when none is selected or multiple are selected)
def detect_bubble(image, anchor, roi, debug_image):
    (rx, ry, rw, rh) = roi
    (anchor_x, anchor_y) = anchor

    x1 = anchor_x + rx
    y1 = anchor_y + ry
    x2 = x1 + rw
    y2 = y1 + rh

    # draw green rectangle around ROI in debug image
    cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # left_zone_roi = thresh[y1:y2, x1:x2]

    q_crop = image[y1:y2, x1:x2]
    # cv2.imwrite('q_crop.jpg', q_crop)

    gray_crop = cv2.cvtColor(q_crop, cv2.COLOR_BGR2GRAY)
    gray_norm = cv2.normalize(gray_crop, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray_norm)
    # cv2.imwrite('q_crop_enhanced.jpg', enhanced)

    blur = cv2.GaussianBlur(gray_crop, (5,5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # cv2.imwrite("q_thresh.jpg", thresh)

    # Contour Detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"{len(contours)} contours found.")

    bubble_candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_MARK_AREA < area < MAX_MARK_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if MIN_CNT_ASPECT_RATIO < aspect_ratio < MAX_CNT_ASPECT_RATIO:
                bubble_candidates.append(cnt)

    bubble_candidates = sorted(bubble_candidates, key=lambda c: cv2.boundingRect(c)[0])

    filled_index = []
    ratios = []
    debug_crop = q_crop.copy()

    for i, cnt in enumerate(bubble_candidates):
        mask = np.zeros(thresh.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        total_pixels = cv2.countNonZero(mask)
        filled_pixels = cv2.countNonZero(cv2.bitwise_and(mask, thresh))
        fill_ratio = filled_pixels / total_pixels if total_pixels > 0 else 0
        ratios.append(fill_ratio)
        # print(f"Bubble {chr(65+i)}: fill_ratio = {fill_ratio:.3f}, area = {cv2.contourArea(cnt)}")

        color = (0, 255, 0)
        if fill_ratio > FILL_THRESHOLD:
            color = (255, 0, 0)
            filled_index.append(i)

        x, y, w, h = cv2.boundingRect(cnt)

        cv2.drawContours(debug_crop, [cnt], -1, color, 2)
        cv2.putText(debug_crop, f"{chr(65+i)}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # global debug image
        cnt_global = cnt + np.array([[[x1, y1]]])
        cv2.drawContours(debug_image, [cnt_global], -1, color, 2)
        cv2.putText(debug_image, 
                    f"{chr(65+i)} {fill_ratio:.2f}",
                    (x1 + x, y1 + y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # cv2.imwrite('q_detected.jpg', debug_crop)

    if not filled_index:
        # print("No bubble detected as filled.")
        return ''
    elif len(filled_index) > 1:
        # print("Multiple bubbles detected.")
        return ''
    else:
        ans = chr(65+filled_index[0])
        # print(f"Detected bubble: {ans}")
        return ans

if __name__ == "__main__":
    print("In main function.")
    image = cv2.imread('dewarped_test25h9.jpg')
    preprocessed_image = image.preprocess(image)
    debug_image = preprocessed_image.copy()

    # detect 25h9 tags
    detection = apriltags.detect_tags_25h9(preprocessed_image)
    tag_points = list(map(lambda t: tuple(map(int, t.center.tolist())), detection))
    print(f"tag points: {tag_points}")

    show_roi_zones(preprocessed_image, tag_points, debug_image)
    # cv2.imwrite('debug_roi.jpg', debug_image)
    answers = []

    for i, point in enumerate(tag_points):
        # print(f"In point {i+1}.")
        q_left_ans = detect_bubble(preprocessed_image, point, LEFT_QUESTION_ROI, debug_image)
        q_right_ans = detect_bubble(preprocessed_image, point, RIGHT_QUESTION_ROI, debug_image)
        answers.extend([q_left_ans, q_right_ans])
        print(f"Q{i*2+1}: {q_left_ans}")
        print(f"Q{i*2+2}: {q_right_ans}")
    
    print(answers)

    # write final debug image
    cv2.imwrite('debug_final.jpg', debug_image)