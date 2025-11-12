import cv2
import numpy as np
import image
import apriltags
import math
import logging

logger = logging.getLogger(__name__)

# roi format: (x_offset, y_offset, width, height)
LEFT_QUESTION_ROI = (75, -40, 475, 85)
RIGHT_QUESTION_ROI = (602, -40, 475, 85)

MIN_MARK_AREA = 500 # TUNE THIS if needed
MAX_MARK_AREA = 800
FILL_THRESHOLD = 0.9

MIN_CNT_ASPECT_RATIO = 0.7
MAX_CNT_ASPECT_RATIO = 1.3

# circularity condition
MIN_CIRCULARITY = 0.5

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
def detect_bubble(image, anchor, roi, debug_image, checked_image, ans_key):
    (rx, ry, rw, rh) = roi
    (anchor_x, anchor_y) = anchor

    x1 = anchor_x + rx
    y1 = anchor_y + ry
    x2 = x1 + rw
    y2 = y1 + rh

    logger.info(f"ROI coordinates: {x1}, {y1} to {x2}, {y2}")

    # draw green rectangle around ROI in debug image
    cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    q_crop = image[y1:y2, x1:x2]
    # cv2.imwrite('q_crop.jpg', q_crop)

    gray_crop = cv2.cvtColor(q_crop, cv2.COLOR_BGR2GRAY)
    gray_norm = cv2.normalize(gray_crop, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray_norm)
    # cv2.imwrite('q_crop_enhanced.jpg', enhanced)

    # blur = cv2.GaussianBlur(gray_norm, (5,5), 0)
    blur = cv2.medianBlur(gray_norm, 3)
    # _, thresh = cv2.threshold(
    #     blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    # )
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        55, 30
    )
    # cv2.imwrite("q_thresh.jpg", thresh)

    # Contour Detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.info(f"{len(contours)} contours found in ROI.")

    # array of contours
    bubble_candidates = []

    for cnt in contours:
        # draw every contour in red in debug image
        cnt_global = cnt + np.array([[[x1, y1]]])
        cv2.drawContours(debug_image, [cnt_global], -1, (0, 0, 255), 1)

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * (area / (perimeter * perimeter))

        # contour checks: 1. area, 2. circularity
        # if there are still more than 4, check if they are evenly spaced and horizontal, and remove the y outlier (todo)

        # 1. area condition
        if MIN_MARK_AREA < area < MAX_MARK_AREA:
            
            # 2. circularity condition
            if circularity > float(MIN_CIRCULARITY):
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
        bubble_area = cv2.contourArea(cnt)
        bubble_perimeter = cv2.arcLength(cnt, True)
        bubble_circularity = 4 * math.pi * (bubble_area / (bubble_perimeter * bubble_perimeter))
        logger.info(f"Bubble {chr(65+i)}: fill_ratio = {fill_ratio:.3f}, area = {bubble_area}, circularity = {bubble_circularity}")

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

    if len(bubble_candidates) != 4:
        logger.debug(f"{len(bubble_candidates)} bubble candidates detected instead of 4.")

        # draw blue box in checked image and write "+0" near top-right
        cv2.rectangle(checked_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(checked_image, "+0", (x1 + rw - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)

        return ''
    else:
        if not filled_index:
            logger.debug("No bubble detected as filled.")

            # draw red box in checked image and write "+0" near top-right
            cv2.rectangle(checked_image, (x1, y1), (x2, y2), (86, 86, 255), 2)
            cv2.putText(checked_image, "+0", (x1 + rw - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)

            return ''
        elif len(filled_index) > 1:
            logger.debug("Multiple bubbles detected.")

            # draw red box in checked image and write "+0" near top-right
            cv2.rectangle(checked_image, (x1, y1), (x2, y2), (86, 86, 255), 2)
            cv2.putText(checked_image, "+0", (x1 + rw - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)

            return ''
        else:
            ans = chr(65+filled_index[0])
            logger.info(f"Detected bubble: {ans}, correct ans: {ans_key}")
            if ans.lower() == ans_key.lower():
                logger.info("Correct ans.")
                # correct ans
                # draw green box in checked image and write "+1" near top-right
                cv2.rectangle(checked_image, (x1, y1), (x2, y2), (0, 127, 0), 2)
                cv2.putText(checked_image, "+1", (x1 + rw - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 127, 0), 5, cv2.LINE_AA)
            else:
                # wrong ans
                cv2.rectangle(checked_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # debug
                logger.info("Wrong ans: drew rectangle.")
                cv2.putText(checked_image, "+0", (x1 + rw - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)
                logger.info("Wrong ans: wrote +0.")
            return ans

if __name__ == "__main__":
    logger.info("In main function.")
    image = cv2.imread('dewarped_test25h9.jpg')
    preprocessed_image = image.preprocess(image)
    debug_image = preprocessed_image.copy()

    # detect 25h9 tags
    detection = apriltags.detect_tags_25h9(preprocessed_image)
    tag_points = list(map(lambda t: tuple(map(int, t.center.tolist())), detection))
    tag_points.sort(key = lambda d: d[1])
    logger.info(f"tag points: {tag_points}")

    show_roi_zones(preprocessed_image, tag_points, debug_image)
    # cv2.imwrite('debug_roi.jpg', debug_image)
    answers = []

    for i, point in enumerate(tag_points):
        logger.debug(f"In point {i+1}.")
        q_left_ans = detect_bubble(preprocessed_image, point, LEFT_QUESTION_ROI, debug_image)
        q_right_ans = detect_bubble(preprocessed_image, point, RIGHT_QUESTION_ROI, debug_image)
        answers.extend([q_left_ans, q_right_ans])
        logger.debug(f"Q{i*2+1}: {q_left_ans}")
        logger.debug(f"Q{i*2+2}: {q_right_ans}")
    
    # code for debugging to check a specific question
    # q_no = 1
    # tag_point_index = (q_no + 1) // 2
    # q_roi = LEFT_QUESTION_ROI if q_no % 2 == 1 else RIGHT_QUESTION_ROI
    # q_ans = detect_bubble(preprocessed_image, tag_points[tag_point_index-1], q_roi, debug_image, checked_image, ans_key[q_no-1])
    # print(f"Q{q_no}: {q_ans}")
    
    logger.info(answers)

    # write final debug image
    cv2.imwrite('debug_final.jpg', debug_image)