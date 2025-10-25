from flask import Flask, request
import os, requests, time
import gemini
import sendmessage

app = Flask(__name__)

SAVE_DIR = "downloads"

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    print("Received:", data)

    messages = data.get("whatsapp", {}).get("messages", [])
    for message in messages:
        fromNo = message.get("from")
        sendmessage.sendMessage(fromNo, "Hi.")

    return "ok", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)