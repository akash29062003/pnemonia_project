from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import requests
import json

app = Flask(__name__)
CORS(app)

print("Loading MobileNetV2 model...")

model = tf.keras.models.load_model(
    "pneumonia_mobilenetv2.h5"
)

print("Model loaded successfully.")

def preprocess_image(image):
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Page routes ──────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/signup")
def signup():
    return render_template("signup.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

# ── API routes ───────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    try:
        img = Image.open(file.stream)
        processed = preprocess_image(img)
        raw_pred = float(model.predict(processed, verbose=0)[0][0])
        if raw_pred > 0.5:
            label = "PNEUMONIA"
            confidence = raw_pred
            severity = "High Risk" if confidence > 0.85 else "Moderate Risk"
        else:
            label = "NORMAL"
            confidence = 1.0 - raw_pred
            severity = "Low Risk"
        buf = io.BytesIO()
        img.convert("RGB").resize((400, 400)).save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return jsonify({
            "label": label,
            "confidence": round(confidence * 100, 2),
            "severity": severity,
            "raw_score": round(raw_pred, 4),
            "image": img_b64
        })
    except Exception as e:
        print("PREDICT ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    from flask import Response, stream_with_context
    data = request.get_json()
    messages = data.get("messages", [])
    diagnosis_context = data.get("diagnosis_context", "")

    system_content = """You are PneumoAI, an expert medical AI assistant specialising in pneumonia diagnosis, treatment, and respiratory health. You have deep knowledge in:
- Pneumonia pathophysiology (bacterial, viral, fungal)
- Chest X-ray interpretation and radiological findings
- Diagnosis criteria (clinical and imaging-based)
- Treatment protocols (antibiotics, antivirals, supportive care)
- Risk stratification (PSI/PORT score, CURB-65)
- Prevention strategies and vaccines
- Post-recovery care and rehabilitation

Guidelines:
- Be clinically accurate but accessible to students and patients
- Always recommend consulting a real doctor for medical decisions
- Use medical terminology with plain-language explanations
- Keep responses concise — max 3-4 sentences for simple questions
- For casual greetings, reply in 1 sentence only
- Never number responses unless the user asks for a list"""

    if diagnosis_context:
        system_content += f"\n\nPatient Context: The AI model analysed a chest X-ray and detected: {diagnosis_context}. Reference this when relevant."

    chat_messages = [{"role": "system", "content": system_content}]
    for msg in messages:
        chat_messages.append({"role": msg["role"], "content": msg["content"]})

    def generate():
        try:
            response = requests.post(
                "http://localhost:1234/v1/chat/completions",
                json={
                    "model": "phi-3-mini-4k-instruct",
                    "messages": chat_messages,
                    "temperature": 0.7,
                    "max_tokens": 300,
                    "stream": True
                },
                stream=True,
                timeout=300
            )
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk.strip() == "[DONE]":
                            yield "data: [DONE]\n\n"
                            break
                        try:
                            obj = json.loads(chunk)
                            delta = obj["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield f"data: {json.dumps({'token': delta})}\n\n"
                        except:
                            pass
        except requests.exceptions.ConnectionError:
            yield f"data: {json.dumps({'error': 'LM Studio server is not running. Open LM Studio and start the server.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
