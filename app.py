from dotenv import load_dotenv
import os
from flask import Flask, request, send_from_directory
import os, requests, time
import gemini
import sendmessage
import apriltags
import json
import threading
import image
import cv2
from pathlib import Path

load_dotenv()

app = Flask(__name__)

SAVE_DIR = "downloads"
DEWARPED_DIR = "dewarped"
SHEETS_LOGGING_URL = os.getenv("SHEETS_LOGGING_URL")

def log_to_sheet(sender, filename, marked, score):
    payload = {
        "sender": sender,
        "filename": filename,
        "marked": marked,
        "score": score
    }
    requests.post(SHEETS_LOGGING_URL, json=payload)

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

                filepath = os.path.join(SAVE_DIR, filename)
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)

                print(f"Saved image: {filepath}")

                sendmessage.sendMessage(fromNo, "Processing...")

                scanned_tags = apriltags.detect_tags(filepath)
                print(f"Detected tags: {list(map(lambda x:x.tag_id, scanned_tags))}")
                if (len(scanned_tags) == 4):
                    # dewarp the image and save the dewarped image
                    dewarped_img = image.dewarp_omr(filepath, scanned_tags)
                    print("Dewarped image.")
                    dewarped_filename = f"{Path(filepath).stem}_dewarped.jpg"
                    dewarped_filepath = os.path.join(DEWARPED_DIR, dewarped_filename)

                    cv2.imwrite(dewarped_filepath, dewarped_img)
                    print(f"Saved dewarped image to {dewarped_filepath}")

                    # send the image to gemini, and send back the results
                    results = gemini.scanImage(dewarped_filepath)
                    # print(f"Results: {results}")

                    # send message with reply
                    sendmessage.sendMessage(fromNo, json.dumps(results))

                    # calculate and send score
                    score = check_results(results, ['C', 'A', 'D', 'C', 'D', 'A', 'B'])
                    sendmessage.sendMessage(fromNo, score)

                    # log successful scan to google sheet
                    log_to_sheet(fromNo, filename, json.dumps(results), score)
                else:
                    sendmessage.sendMessage(fromNo, "Please take a complete photo of the image. ⟳")

                    # log failed scan to google sheet
                    log_to_sheet(fromNo, filename, "failed", "")
            else:
                sendmessage.sendMessage(fromNo, "Please send an image of a scanned worksheet.")

                # log failed scan to google sheet
                log_to_sheet(fromNo, "none", "failed", "")
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
    if (len(results['marked_answers']) != len(ans_key)):
        return "An error occurred. Please try again. ⟳"
    else:
        marked_lowercase = [item.lower() for item in results['marked_answers']]
        anskey_lowercase = [item.lower() for item in ans_key]

        marks = 0

        for i in range(len(marked_lowercase)):
            if marked_lowercase[i] == anskey_lowercase[i]:
                marks += 1
        
        return f"Congratulations! Total marks: {marks}/7"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)