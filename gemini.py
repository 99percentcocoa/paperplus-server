from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
import cv2
import numpy as np
import json

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def scanImage(image_input):

    # case 1: image input is a file path
    if isinstance(image_input, str):
        with open(image_input, 'rb') as f:
            image_bytes = f.read()
    
    # case 2: image input is an opencv image array
    elif isinstance(image_input, np.ndarray):
        success, buffer = cv2.imencode('.jpg', image_input)
        if not success:
            raise ValueError("Failed to encode OpenCV image.")
        image_bytes = buffer.tobytes()

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg'
            ),
            'Scan this worksheet and identify the marked answers.'
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "required": [
                    "marked_answers"
                ],
                "properties": {
                    "marked_answers": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "type": "object"
            },
            temperature=0.0,
            system_instruction="""You are a teacher's assistant. Your role is to identify marked student responses in multiple-choice OMR worksheets.
    There are 10 questions on each page, each question having 4 options A, B, C and D in a row from left to right, each with a bubble which students darken to mark their answer. Identify the marked option by checking which of the 4 bubbles is darkened.
    Identify the marked answers for all 10 questions in the worksheet."""
        )
    )
    print(response.text)
    return json.loads(response.text)

if __name__ == '__main__':
    result = scanImage('testaprilfull.jpg')
    print(result)
    print(type(result['marked_answers']))