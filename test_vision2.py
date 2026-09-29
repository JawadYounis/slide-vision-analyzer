import base64
import json
import requests
from pathlib import Path
from datetime import datetime


# -----------------------------
# Configuration
# -----------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:4b"

IMAGE_PATH = Path("slide2.png")

JSON_OUTPUT = Path("vision_result.json")
TEXT_OUTPUT = Path("vision_result.txt")


# -----------------------------
# Encode image
# -----------------------------

def encode_image(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


# -----------------------------
# Send image to Ollama
# -----------------------------

def analyze_image(image_path):

    image_base64 = encode_image(image_path)

    prompt = """
Analyze this image as if it were a presentation slide.

Provide a detailed analysis containing:

1. Title
2. All visible text
3. Main subject/topic
4. Important visual elements
5. Images and what they represent
6. Charts or graphs
7. Tables
8. Diagrams
9. Key points
10. Overall summary
11. Any important relationships between visual elements and text

Be accurate and only describe information that is actually visible in the slides.
Do not invent information.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64]
                }
            ],
            "stream": False
        },
        timeout=300
    )

    response.raise_for_status()

    return response.json()


# -----------------------------
# Main
# -----------------------------

print("Analyzing image...")
print(f"Model: {MODEL}")
print(f"Image: {IMAGE_PATH}")

result = analyze_image(IMAGE_PATH)

model_response = result["message"]["content"]


# -----------------------------
# Save JSON
# -----------------------------

output_data = {
    "timestamp": datetime.now().isoformat(),
    "model": MODEL,
    "image": str(IMAGE_PATH),
    "response": model_response
}

with open(JSON_OUTPUT, "w", encoding="utf-8") as file:
    json.dump(
        output_data,
        file,
        indent=4,
        ensure_ascii=False
    )


# -----------------------------
# Save readable TXT
# -----------------------------

with open(TEXT_OUTPUT, "w", encoding="utf-8") as file:

    file.write("OLLAMA VISION TEST\n")
    file.write("=" * 60 + "\n\n")

    file.write(f"Model: {MODEL}\n")
    file.write(f"Image: {IMAGE_PATH}\n")
    file.write(
        f"Timestamp: {output_data['timestamp']}\n\n"
    )

    file.write("MODEL RESPONSE\n")
    file.write("=" * 60 + "\n\n")

    file.write(model_response)


# -----------------------------
# Console output 
# -----------------------------

print("\nAnalysis complete.")

print(f"\nJSON saved to:")
print(JSON_OUTPUT.absolute())

print(f"\nText saved to:")
print(TEXT_OUTPUT.absolute())

print("\n" + "=" * 60)
print("MODEL RESPONSE")
print("=" * 60)
print(model_response)