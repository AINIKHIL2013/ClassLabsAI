from flask import Flask, request, jsonify, send_file, render_template_string
import requests
import os
from io import BytesIO

app = Flask(__name__)

# Load Hugging Face API token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set!")

# Model endpoints
STT_MODEL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-large-960h"
LLM_MODEL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
TTS_MODEL = "https://api-inference.huggingface.co/models/neuphonic/neutts-air"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# --- ROUTES --- #

@app.route("/")
def home():
    # Simple futuristic UI
    return render_template_string("""
    <html>
    <head>
        <title>ClassLabs AI</title>
        <style>
            body { background-color: #0d0d0d; color: #00e0ff; font-family: 'Orbitron', sans-serif; text-align: center; }
            h1 { margin-top: 50px; font-size: 3em; text-shadow: 0px 0px 20px #00ffff; }
            input[type=file] { margin-top: 30px; color: #00ffff; }
            button { background: #00ffff; border: none; padding: 10px 20px; font-size: 1em; margin-top: 20px; cursor: pointer; border-radius: 10px; transition: 0.3s; }
            button:hover { background: #0099cc; color: white; }
            audio { margin-top: 30px; }
        </style>
    </head>
    <body>
        <h1>🎧 ClassLabs AI Summarizer</h1>
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" name="file" accept="audio/*" required><br>
            <button type="submit">Upload & Summarize</button>
        </form>
        <p id="status"></p>
        <audio id="audioPlayer" controls style="display:none;"></audio>

        <script>
        const form = document.getElementById('uploadForm');
        const statusText = document.getElementById('status');
        const audioPlayer = document.getElementById('audioPlayer');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            statusText.textContent = 'Processing... please wait ⚙️';
            const formData = new FormData(form);

            const response = await fetch('/process', { method: 'POST', body: formData });
            if (!response.ok) {
                statusText.textContent = 'Error during processing.';
                return;
            }

            const blob = await response.blob();
            const audioUrl = URL.createObjectURL(blob);
            audioPlayer.src = audioUrl;
            audioPlayer.style.display = 'block';
            statusText.textContent = '✅ Done! Summary below:';
        });
        </script>
    </body>
    </html>
    """)


@app.route("/process", methods=["POST"])
def process():
    """Full pipeline: Speech → Text → Summary → Speech"""
    if "file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    # Step 1 — Speech-to-Text
    audio_file = request.files["file"]
    stt_response = requests.post(STT_MODEL, headers=headers, files={"file": audio_file})
    stt_text = ""
    try:
        stt_text = stt_response.json().get("text", "")
    except:
        pass
    if not stt_text:
        return jsonify({"error": "Speech recognition failed"}), 500

    # Step 2 — Summarize with LLM
    summary_prompt = f"Summarize this classroom discussion clearly and briefly:\n\n{stt_text}"
    llm_response = requests.post(LLM_MODEL, headers=headers, json={"inputs": summary_prompt})
    try:
        summary_text = llm_response.json()[0]["generated_text"]
    except:
        summary_text = "Summary generation failed."

    # Step 3 — Text-to-Speech
    tts_response = requests.post(TTS_MODEL, headers=headers, json={"inputs": summary_text})
    if tts_response.status_code != 200:
        return jsonify({"error": "TTS generation failed", "details": tts_response.text}), 500

    # Return playable WAV file
    return send_file(BytesIO(tts_response.content), mimetype="audio/wav")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
