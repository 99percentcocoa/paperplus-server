import tags
import cv2
import apriltags
import image
from tinydb import TinyDB
import omr_detection
import os
from dotenv import load_dotenv
from pathlib import Path
import app
from PIL import Image
load_dotenv()

SAVE_DIR = "downloads"
DEWARPED_DIR = "dewarped"
SHEETS_LOGGING_URL = os.getenv("SHEETS_LOGGING_URL")

DOWNLOADS_PATH = os.getenv("DOWNLOADS_PATH")
DEWARPED_PATH = os.getenv("DEWARPED_PATH")
DEBUG_PATH = os.getenv("DEBUG_PATH")
CHECKED_PATH = os.getenv("CHECKED_PATH")
LOGS_PATH = os.getenv("LOGS_PATH")
SERVER_IP = os.getenv("SERVER_IP")
TESTING_PATH = "testing"

def crop_image(fp):
    detection = apriltags.detect_tags_36h11(fp)
    print(f"tags detected: {[d.tag_id for d in detection]}")

    if len(detection) == 4:
        # arrange clockwise
        detection = tags.sort_detections_clockwise(detection)
        print(f"clockwise sorted: {[d.tag_id for d in detection]}")

        wid, detection = tags.detect_orientation_and_decode(detection)
        print(f"detected wid {wid}")

        cropped = image.dewarp_omr(fp, detection)
        cropped_filename = f'cropped_{Path(fp).stem}.jpg'
        cropped_filepath = os.path.join(TESTING_PATH, cropped_filename)
        cv2.imwrite(cropped_filepath, cropped)
        return wid, cropped
    else:
        print("Less/more than 4 tags detected.")
        return None, None

def check_e2e(fp):
    worksheet_id, cropped = crop_image(fp)

    # returns color preprocessed image
    preprocessed = image.clean_document(cropped)
    # preprocessed = image.preprocess(cropped)

    debug_img = preprocessed.copy()
    checked_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

    db = TinyDB('worksheets.json')
    ans_key = db.get(doc_id=worksheet_id).get('answerKey')
    print(f"Answer key for worksheet {worksheet_id}: {ans_key}")

    answers = []

    # detect the 25h9 tags
    detection_25h9 = apriltags.detect_tags_25h9(preprocessed)
    detected_tags_25h9 = list(map(lambda x:x.tag_id, detection_25h9))
    print(f"Detected tags: {detected_tags_25h9}")

    # verify 25h9 tags detection
    required = set(range(1, 11))
    present = set(detected_tags_25h9)

    if not required.issubset(present):
        missing = required - present
        print(f"Missing 25h9 tags: {missing}")

        # 25h9 tags are missing, ask user to send image again.
        print("25h9 tags missing")

    else: # if not required.issubset(present):
        extra = present - required
        print(f"Extra tags detected: {extra}")
        if extra:
            # extra tags, remove them and continue
            detection_25h9[:] = [d for d in detection_25h9 if d.tag_id not in list(extra)]
            detected_tags_25h9 = list(map(lambda x:x.tag_id, detection_25h9))
            
            print(f"Extra tags removed, new list: {detected_tags_25h9}")

        else: # if extra:
            # all tags correct
            print("All 25h9 tags are correct.")

        tag_points = list(map(lambda t: tuple(map(int, t.center.tolist())), detection_25h9))
        for i, point in enumerate(tag_points):
            print(f"In point {i+1}.")
            q_left_ans_key = ans_key[i*2]
            print(f"In question {i*2+1}")
            q_left_ans = omr_detection.detect_bubble(preprocessed, point, omr_detection.LEFT_QUESTION_ROI, debug_img, checked_img, q_left_ans_key)

            q_right_ans_key = ans_key[i*2+1]
            print(f"In question {i*2+2}")
            q_right_ans = omr_detection.detect_bubble(preprocessed, point, omr_detection.RIGHT_QUESTION_ROI, debug_img, checked_img, q_right_ans_key)
            answers.extend([q_left_ans, q_right_ans])
            print(f"Q{i*2+1}: {q_left_ans}")
            print(f"Q{i*2+2}: {q_right_ans}")
            
        print("Finished checking.")
    
        print(answers)

        # save debug image
        debug_filename = f'debug_{Path(fp).stem}.jpg'
        debug_filepath = os.path.join(TESTING_PATH, debug_filename)
        cv2.imwrite(debug_filepath, debug_img)
        print(f"Saved debug image at {debug_filepath}")

        # calculate and send score
        score = app.check_results(answers, ans_key)

        # save checked image
        checked_filename = f'checked_{Path(fp).stem}.jpg'
        checked_filepath = os.path.join(TESTING_PATH, checked_filename)
        # checked_URL = f"http://{SERVER_IP}:3000/checked/{checked_filename}"
        # cv2.imwrite(checked_filepath, checked_img)
        check_circle = omr_detection.make_circle_mark(score, len(ans_key))
        checked_img.paste(check_circle, (100, 50), check_circle)
        checked_img.save(checked_filepath)
        print(f"Saved checked image at {checked_filepath}.")

        # debugURL = f"http://{SERVER_IP}:3000/debug/{debug_filename}"

        # send message with reply
        # sendmessage.sendMessage(fromNo, "Your answers:\n"+'\n '.join(f"{i}. {item}" for i, item in enumerate(answers, start=1)))
        print(score)