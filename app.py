from dotenv import load_dotenv
import os
from flask import Flask, request, send_from_directory, abort
import os, requests, time
import gemini
import sendmessage
import apriltags
import json
import threading
import image
import cv2
import omr_detection
from pathlib import Path

load_dotenv()

app = Flask(__name__)

SAVE_DIR = "downloads"
DEWARPED_DIR = "dewarped"
SHEETS_LOGGING_URL = os.getenv("SHEETS_LOGGING_URL")

DOWNLOADS_PATH = os.getenv("DOWNLOADS_PATH")
DEWARPED_PATH = os.getenv("DEWARPED_PATH")
DEBUG_PATH = os.getenv("DEBUG_PATH")
CHECKED_PATH = os.getenv("CHECKED_PATH")
SERVER_IP = os.getenv("SERVER_IP")

def log_to_sheet(sender, fileURL, debugURL, marked, score):
    payload = {
        "sender": sender,
        "fileURL": fileURL,
        "debugURL": debugURL,
        "marked": marked,
        "score": score
    }
    print(f"Payload: {payload}")
    requests.post(SHEETS_LOGGING_URL, json=payload, headers={"Content-Type": "application/json"})

def handle_message(data):
    try:
        print("Received:", data)

        messages = data.get("whatsapp", {}).get("messages", [])
        for message in messages:
            fromNo = message.get("from")
            callback_type = message.get("callback_type")

            print(f"Received message from {fromNo}")

            # filter out the non-incoming messages (delivery, status messages)
            if callback_type and callback_type != "incoming_message":
                print("Received delivery message. Continuing...")
                continue
            
            content = message.get("content", {})

            if not content:
                print("No content found. Skipping...")
                continue
            
            if content.get("type") == "image" and "image" in content:
                img = content["image"]
                url = img.get("url") or img.get("link")
                if not url:
                    continue

                r = requests.get(url, stream=True, timeout=30)
                ext = r.headers.get("Content-Type", "image/jpeg").split("/")[-1]
                filename = f"{message.get('from','unknown')[1:]}_{int(time.time())}.{ext}"
                fileURL = f"http://{SERVER_IP}:3000/files/{filename}"

                filepath = os.path.join(SAVE_DIR, filename)
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)

                print(f"Saved image: {filepath}")

                sendmessage.sendMessage(fromNo, "Processing...")

                corner_tags = apriltags.detect_tags_36h11(filepath)
                print(f"Detected tags: {list(map(lambda x:x.tag_id, corner_tags))}")

                # corner tags (36h11) should be 1, 2, 3, 4
                # question tags (25h9) should be 1, 2, 3, ..., 10
                if (len(corner_tags) == 4):
                    # dewarp the image and save the dewarped image. Also preprocess it.
                    dewarped_img = image.preprocess(image.dewarp_omr(filepath, corner_tags))
                    debug_img = dewarped_img.copy()
                    print("Dewarped image.")

                    # checked image
                    checked_img = dewarped_img.copy()

                    dewarped_filename = f"{Path(filepath).stem}_dewarped.jpg"
                    dewarped_filepath = os.path.join(DEWARPED_DIR, dewarped_filename)
                    # dewarpedURL = f"http://{SERVER_IP}:3000/dewarped/{dewarped_filename}"

                    cv2.imwrite(dewarped_filepath, dewarped_img)
                    print(f"Saved dewarped image to {dewarped_filepath}")

                    # # split image into left half and right half
                    # dewarped_left, dewarped_right = image.split_img(dewarped_img)

                    # # send the image to gemini, and send back the results
                    # results_left = gemini.scanImage(dewarped_left)['marked_answers']
                    # print(f"Left results: {results_left}")
                    # results_right = gemini.scanImage(dewarped_right)['marked_answers']
                    # print(f"Right results: {results_right}")
                    # results_combined = [val for pair in zip(results_left, results_right) for val in pair]
                    # print(f"Combined results: {results_combined}")

                    # # send message with reply
                    # sendmessage.sendMessage(fromNo, ', '.join(f"{i}. {item}" for i, item in enumerate(results_combined, start=1)))

                    # calculate and send score
                    # score = check_results(results_combined, ['C', 'A', 'D', 'C', 'C', 'A', 'D', 'C', 'D', 'A', 'B', 'C', 'A', 'D', 'C', 'C', 'A', 'C', 'A', 'B'])

                    # detect the 25h9 tags
                    detection_25h9 = apriltags.detect_tags_25h9(dewarped_img)
                    tag_points = list(map(lambda t: tuple(map(int, t.center.tolist())), detection_25h9))
                    print(f"Detected tags: {tag_points}")

                    ans_key = ['C', 'A', 'D', 'C', 'C', 'A', 'D', 'C', 'D', 'A', 'B', 'C', 'A', 'D', 'C', 'C', 'A', 'C', 'A', 'B']
                    answers = []

                    for i, point in enumerate(tag_points):
                        # print(f"In point {i+1}.")
                        q_left_ans_key = q_left_ans_key = ans_key[i*2]
                        q_left_ans = omr_detection.detect_bubble(dewarped_img, point, omr_detection.LEFT_QUESTION_ROI, debug_img, checked_img, q_left_ans_key)

                        q_right_ans_key = ans_key[i*2+1]
                        q_right_ans = omr_detection.detect_bubble(dewarped_img, point, omr_detection.RIGHT_QUESTION_ROI, debug_img, checked_img, q_right_ans_key)
                        answers.extend([q_left_ans, q_right_ans])
                        print(f"Q{i*2+1}: {q_left_ans}")
                        print(f"Q{i*2+2}: {q_right_ans}")
                        
                    print("Finished checking.")
                
                    print(answers)

                    # save debug image
                    debug_filename = f'debug_{Path(filepath).stem}.jpg'
                    debug_filepath = os.path.join(DEBUG_PATH, debug_filename)
                    cv2.imwrite(debug_filepath, debug_img)
                    print(f"Saved debug image at {debug_filepath}")

                    # save checked image
                    checked_filename = f'checked_{Path(filepath).stem}.jpg'
                    checked_filepath = os.path.join(CHECKED_PATH, checked_filename)
                    checked_URL = f"http://{SERVER_IP}:3000/checked/{checked_filename}"
                    cv2.imwrite(checked_filepath, checked_img)
                    print(f"Saved checked image at {checked_filepath}.")

                    debugURL = f"http://{SERVER_IP}:3000/debug/{debug_filename}"

                    # send message with reply
                    # sendmessage.sendMessage(fromNo, "Your answers:\n"+'\n '.join(f"{i}. {item}" for i, item in enumerate(answers, start=1)))
                    # calculate and send score
                    score = check_results(answers, ans_key)

                    sendmessage.sendMessage(fromNo, score)

                    # send visual checked paper
                    print("Sending checked image.")
                    sendmessage.sendImage(fromNo, checked_URL)

                    # log successful scan to google sheet
                    print(f"Logging {fromNo}, {fileURL}, {debugURL}, {json.dumps(answers)}, {score}")
                    log_to_sheet(fromNo, fileURL, debugURL, json.dumps(answers), score)
                else:
                    sendmessage.sendMessage(fromNo, "Please take a complete photo of the image. ⟳")

                    # log failed scan to google sheet
                    log_to_sheet(fromNo, fileURL, "", "failed", "")
            else:
                sendmessage.sendMessage(fromNo, "Please send an image of a scanned worksheet.")

                # log failed scan (user message does not contain image) to google sheet
                log_to_sheet(fromNo, "none", "", "failed", "")
    except Exception as e:
        print("Error in background thread: ", e)


# use threading for the webhook so that can return 200 ok to exotel to avoid receiving duplicates
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    threading.Thread(target=handle_message, args=(data,)).start()
    return "ok", 200

def check_results(results, ans_key):
    print("Checking results.")
    if (len(results) != len(ans_key)):
        return "An error occurred. Please try again. ⟳"
    else:
        marked_lowercase = [item.lower() for item in results]
        anskey_lowercase = [item.lower() for item in ans_key]

        marks = 0

        for i in range(len(marked_lowercase)):
            if marked_lowercase[i] == anskey_lowercase[i]:
                marks += 1
        
        return f"Congratulations! Total marks: {marks}/20"


# serve files from downloads
@app.route('/files/<path:filename>')
def serve_file(filename):
    try:
        return send_from_directory(DOWNLOADS_PATH, filename, as_attachment=False)
    except FileNotFoundError:
        abort(404)

# serve files from debug
@app.route('/debug/<path:filename>')
def serve_debug_file(filename):
    try:
        return send_from_directory(DEBUG_PATH, filename, as_attachment=False)
    except FileNotFoundError:
        abort(404)

# serve files from checked
@app.route('/checked/<path:filename>', methods=['GET', 'POST'])
def serve_checked_file(filename):
    try:
        return send_from_directory(CHECKED_PATH, filename, as_attachment=False)
    except FileNotFoundError:
        abort(404)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)