# pylint: disable=no-member
import math
import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import image
import apriltags

logger = logging.getLogger(__name__)

# roi format: (x_offset, y_offset, width, height)
LEFT_QUESTION_ROI = (85, -40, 475, 85)
RIGHT_QUESTION_ROI = (620, -40, 475, 85)

MIN_MARK_AREA = 600 # TUNE THIS if needed
MAX_MARK_AREA = 950
FILL_THRESHOLD = 0.6

# circularity condition
MIN_CIRCULARITY = 0.75

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
# checked_image is a PIL image while the other images are cv2 images.
# will return: 'A', 'B', 'C', 'D' or '' (when none is selected or multiple are selected)
def detect_bubble(image, anchor, roi, debug_image, checked_image, ans_key):
    (rx, ry, rw, rh) = roi
    (anchor_x, anchor_y) = anchor

    x1 = anchor_x + rx
    y1 = anchor_y + ry
    x2 = x1 + rw
    y2 = y1 + rh

    logger.info("ROI coordinates: %s, %s to %s, %s.", x1, y1, x2, y2)

    # PIL setup for adding tick and cross marks
    font = ImageFont.truetype("NotoSansSymbols2-Regular.ttf", 60)
    pil_draw = ImageDraw.Draw(checked_image)

    # draw green rectangle around ROI in debug image
    # cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    q_crop = image[y1:y2, x1:x2]
    # cv2.imwrite('q_crop.jpg', q_crop)

    gray_crop = cv2.cvtColor(q_crop, cv2.COLOR_BGR2GRAY)
    gray_norm = cv2.normalize(gray_crop, None, 0, 255, cv2.NORM_MINMAX)
    
    thresh = cv2.adaptiveThreshold(
        gray_norm, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        55, 30
    )
    # cv2.imwrite("q_thresh.jpg", thresh)

    # Contour Detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.info("%s contours found in ROI.", len(contours))

    # array of contours
    bubble_candidates = []

    for idx, cnt in enumerate(contours):
        # draw every contour in red in debug image
        cnt_global = cnt + np.array([[[x1, y1]]])
        cv2.drawContours(debug_image, [cnt_global], -1, (0, 0, 255), 1)

        # label the contour for debugging
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"]) + x1
            cY = int(M["m01"] / M["m00"]) + y1
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cX = (x + w // 2) + x1
            cY = (y + h // 2) + y1
        
        cv2.putText(debug_image, f"{idx+1}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) # Red text

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * (area / (perimeter * perimeter))
        logger.debug("Contour %s: area = %s, perimiter = %s, circularity = %s.", idx+1, area, perimeter, circularity)

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

        # shrink the mask to account for the thickness of the bubble shape
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shrunken_mask = cv2.erode(mask, kernel, iterations=2)

        total_pixels = cv2.countNonZero(shrunken_mask)
        filled_pixels = cv2.countNonZero(cv2.bitwise_and(shrunken_mask, thresh))
        fill_ratio = filled_pixels / total_pixels if total_pixels > 0 else 0
        ratios.append(fill_ratio)
        bubble_area = cv2.contourArea(cnt)
        bubble_perimeter = cv2.arcLength(cnt, True)
        bubble_circularity = 4 * math.pi * (bubble_area / (bubble_perimeter * bubble_perimeter))
        logger.info("Bubble %s: fill_ratio = %s, area = %s, circularity = %s.", chr(65+i), fill_ratio, bubble_area, bubble_circularity)

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
        logger.debug("%s bubble candidates detected instead of 4.", len(bubble_candidates))

        # draw blue box in debug image
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # draw blue box in checked image and write "+0" near top-right
        # cv2.rectangle(checked_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # cv2.putText(checked_image, "+0", (x1 + rw - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)
        pil_draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=(0, 0, 200))
        pil_draw.text((x1 + rw - 5, y1 - 5), "?", fill=(0, 0, 200), font=font)

        return ''
    else:
        if not filled_index:
            logger.debug("No bubble detected as filled.")

            # draw red box in debug image
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (86, 86, 255), 2)

            # draw red box in checked image and write "+0" near top-right
            # cv2.rectangle(checked_image, (x1, y1), (x2, y2), (86, 86, 255), 2)
            # cv2.putText(checked_image, "+0", (x1 + rw - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)
            pil_draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=(255, 86, 86))
            pil_draw.text((x1 + rw - 5, y1 - 5), "?", fill=(255, 86, 86), font=font)

            return ''
        elif len(filled_index) > 1:
            logger.debug("Multiple bubbles detected.")

            # draw red box in debug image
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (86, 86, 255), 2)

            # draw red box in checked image and write "+0" near top-right
            # cv2.rectangle(checked_image, (x1, y1), (x2, y2), (86, 86, 255), 2)
            # cv2.putText(checked_image, "+0", (x1 + rw - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)
            pil_draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=(255, 86, 86))
            pil_draw.text((x1 + rw - 5, y1 - 5), "✘", fill=(255, 86, 86), font=font)

            return ''
        else:
            ans = chr(65+filled_index[0])
            logger.info("Detected bubble: %s, correct ans: %s.", ans, ans_key)
            if ans.lower() == ans_key.lower():
                logger.info("Correct ans.")
                # correct ans
                # draw green box in debug image
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # draw green box in checked image and write "+1" near top-right
                # cv2.rectangle(checked_image, (x1, y1), (x2, y2), (0, 127, 0), 2)
                # cv2.putText(checked_image, "+1", (x1 + rw - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 127, 0), 5, cv2.LINE_AA)
                pil_draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=(0, 127, 0))
                pil_draw.text((x1 + rw - 5, y1 - 5), "✔", fill=(0, 127, 0), font=font)

            else:
                # wrong ans
                # draw red box in debug image
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # draw red box in checked image
                # cv2.rectangle(checked_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # cv2.putText(checked_image, "+0", (x1 + rw - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, cv2.LINE_AA)
                pil_draw.rectangle([(x1, y1), (x2, y2)], fill=None, outline=(255, 86, 86))
                pil_draw.text((x1 + rw - 5, y1 - 5), "✘", fill=(255, 86, 86), font=font)
            return ans

# make a circle mark to show marks obtained
def make_circle_mark(obtained, total, diameter=150):
    # Canvas (RGBA so it supports transparency)
    img = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    DARK_BLUE = (10, 20, 120, 255)

    # Circle border
    draw.ellipse(
        [(5, 5), (diameter-5, diameter-5)],
        outline=DARK_BLUE,
        width=7
    )

    # Horizontal line through center
    center_y = diameter // 2
    draw.line(
        [(20, center_y), (diameter-20, center_y)],
        fill=DARK_BLUE,
        width=7
    )

    # Load font
    try:
        font = ImageFont.truetype("NotoSans-Bold.ttf", 50)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # --- TOP TEXT (obtained marks) ---
    top_text = str(obtained)
    bbox = draw.textbbox((0, 0), top_text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    draw.text(
        ((diameter - tw) // 2, center_y - th - 30),
        top_text,
        fill=DARK_BLUE,
        font=font
    )

    # --- BOTTOM TEXT (total marks) ---
    bottom_text = str(total)
    bbox2 = draw.textbbox((0, 0), bottom_text, font=font)
    tw2 = bbox2[2] - bbox2[0]
    th2 = bbox2[3] - bbox2[1]

    draw.text(
        ((diameter - tw2) // 2, center_y),
        bottom_text,
        fill=DARK_BLUE,
        font=font
    )

    return img

if __name__ == "__main__":
    logger.info("In main function.")
    image = cv2.imread('dewarped_test25h9.jpg')
    preprocessed_image = image.preprocess(image)
    debug_image = preprocessed_image.copy()

    # detect 25h9 tags
    detection = apriltags.detect_tags_25h9(preprocessed_image)
    tag_points = list(map(lambda t: tuple(map(int, t.center.tolist())), detection))
    tag_points.sort(key = lambda d: d[1])
    logger.info("tag points: %s.", tag_points)

    show_roi_zones(preprocessed_image, tag_points, debug_image)
    # cv2.imwrite('debug_roi.jpg', debug_image)
    answers = []

    for i, point in enumerate(tag_points):
        logger.debug("In point %s.", i+1)
        q_left_ans = detect_bubble(preprocessed_image, point, LEFT_QUESTION_ROI, debug_image)
        q_right_ans = detect_bubble(preprocessed_image, point, RIGHT_QUESTION_ROI, debug_image)
        answers.extend([q_left_ans, q_right_ans])
        logger.debug("Q%s: %s.", i*2+1, q_left_ans)
        logger.debug("Q%s: %s.", i*2+2, q_right_ans)
    
    # code for debugging to check a specific question
    # q_no = 1
    # tag_point_index = (q_no + 1) // 2
    # q_roi = LEFT_QUESTION_ROI if q_no % 2 == 1 else RIGHT_QUESTION_ROI
    # q_ans = detect_bubble(preprocessed_image, tag_points[tag_point_index-1], q_roi, debug_image, checked_image, ans_key[q_no-1])
    # print(f"Q{q_no}: {q_ans}")
    
    logger.info(answers)

    # write final debug image
    cv2.imwrite('debug_final.jpg', debug_image)