from flask import Flask, request, jsonify, send_file, render_template
import requests
import os
from io import BytesIO

app = Flask(__name__)

# Load Hugging Face API token (set this in Render or locally)
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set!")

# Model endpoints
STT_MODEL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-large-960h"
LLM_MODEL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
TTS_MODEL = "https://api-inference.huggingface.co/models/neuphonic/neutts-air"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}


@app.route("/")
def home():
    """Futuristic UI"""
    return render_template("index.html")


@app.route("/stt", methods=["POST"])
def stt():
    """Speech-to-Text"""
    if "file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["file"]
    response = requests.post(STT_MODEL, headers=headers, files={"file": audio_file})
    return jsonify(response.json())


@app.route("/chat", methods=["POST"])
def chat():
    """Summarizer or chat using Mistral LLM"""
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "No prompt provided"}), 400

    # Add summarization logic
    prompt_text = f"Summarize this clearly and concisely:\n\n{data['prompt']}"
    payload = {"inputs": prompt_text}
    response = requests.post(LLM_MODEL, headers=headers, json=payload)

    try:
        result = response.json()
        summary = result[0]["generated_text"] if isinstance(result, list) else str(result)
    except Exception:
        summary = "Error generating summary."

    return jsonify({"summary": summary})


@app.route("/tts", methods=["POST"])
def tts():
    """Text-to-Speech"""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    payload = {"inputs": data["text"]}
    response = requests.post(TTS_MODEL, headers=headers, json=payload)

    if response.status_code != 200:
        return jsonify({"error": "TTS generation failed", "details": response.text}), 500

    audio = BytesIO(response.content)
    return send_file(audio, mimetype="audio/wav")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
