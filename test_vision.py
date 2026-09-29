import base64
import requests


def encode_image(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


image = encode_image("slide.png")

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "gemma3:4b",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Analyze this presentation slide. "
                    "Describe the title, text, images, diagrams, "
                    "charts, and the main message of the slide."
                ),
                "images": [image]
            }
        ],
        "stream": False
    }
)

response.raise_for_status()

result = response.json()

print(result["message"]["content"])