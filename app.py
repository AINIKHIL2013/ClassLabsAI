from flask import Flask, render_template, request, jsonify, send_file
import requests, os
from io import BytesIO

app = Flask(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set!")

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

STT_MODEL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-large-960h"
LLM_MODEL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
TTS_MODEL = "https://api-inference.huggingface.co/models/neuphonic/neutts-air"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio = request.files["file"]

    # Step 1 - Speech to Text
    stt_res = requests.post(STT_MODEL, headers=headers, files={"file": audio})
    stt_data = stt_res.json()
    if "text" in stt_data:
        transcript = stt_data["text"]
    else:
        transcript = stt_data[0].get("text", "")

    # Step 2 - Summarize
    prompt = f"Summarize this classroom lecture transcript:\n\n{transcript}"
    llm_res = requests.post(LLM_MODEL, headers=headers, json={"inputs": prompt})
    llm_data = llm_res.json()
    summary = llm_data[0]["generated_text"] if isinstance(llm_data, list) else llm_data.get("generated_text", "")

    # Step 3 - Text to Speech
    tts_res = requests.post(TTS_MODEL, headers=headers, json={"inputs": summary})
    if tts_res.status_code != 200:
        return jsonify({"error": "TTS failed", "details": tts_res.text}), 500

    audio_data = BytesIO(tts_res.content)
    return send_file(audio_data, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
