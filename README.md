# 🫁 PneumoAI — Chest X-Ray Intelligence Platform

> AI-powered pneumonia detection with real-time diagnostic chatbot

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────┐
│              Browser (HTML/CSS/JS)           │
│  ┌──────────────┐   ┌──────────────────┐    │
│  │  X-Ray Upload│   │  PneumoAI Chat   │    │
│  │  + Results   │   │  (Claude LLM)    │    │
│  └──────┬───────┘   └────────┬─────────┘    │
└─────────┼────────────────────┼──────────────┘
          │ POST /predict      │ POST /chat
          ▼                    ▼
┌─────────────────────────────────────────────┐
│              Flask Backend (Python)          │
│                                             │
│  ┌──────────────────────┐                  │
│  │  MobileNetV2 Model   │  ← .h5 file      │
│  │  TensorFlow/Keras    │                  │
│  └──────────────────────┘                  │
│                                             │
│  ┌──────────────────────┐                  │
│  │  Claude Sonnet API   │  ← Anthropic     │
│  │  (Medical Q&A)       │                  │
│  └──────────────────────┘                  │
└─────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone / unzip the project
```bash
cd pneumonia_app/
```

### 2. Place your model
Ensure `pneumonia_mobilenetv2.h5` is in the root of `pneumonia_app/`

### 3. Set your Anthropic API key
```bash
# Linux/Mac
export ANTHROPIC_API_KEY=your_api_key_here

# Windows (CMD)
set ANTHROPIC_API_KEY=your_api_key_here
```

### 4. Install & Run
```bash
pip install -r requirements.txt
python app.py
```

### 5. Open browser
Navigate to `http://localhost:5000`

---

## 🔑 Get Your Anthropic API Key (Free)

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign up / log in
3. Go to **API Keys** → **Create Key**
4. Set it as environment variable (step 3 above)

> **Note:** The chatbot uses the Claude API. Without a key, the X-ray prediction still works — only the chatbot will fail.

---

## 📁 Project Structure

```
pneumonia_app/
├── app.py                   # Flask backend (routes, model, chat proxy)
├── pneumonia_mobilenetv2.h5 # Your trained model
├── requirements.txt         # Python dependencies
├── run.sh                   # Startup script
├── README.md
└── templates/
    └── index.html           # Full frontend (HTML + CSS + JS)
```

---

## 🧠 Technical Details

### Model
- **Architecture:** MobileNetV2 (Transfer Learning)
- **Input:** 224×224 RGB images, normalized to [0,1]
- **Output:** Sigmoid score (>0.5 = Pneumonia, ≤0.5 = Normal)
- **Dataset:** Kaggle Chest X-Ray Images (Normal/Pneumonia) — 5,863 images
- **Framework:** TensorFlow / Keras

### Backend (Flask)
- `POST /predict` — Accepts image file, returns label, confidence, severity, raw score
- `POST /chat` — Proxies to Claude Sonnet with medical system prompt + optional diagnosis context
- `GET /` — Serves the HTML frontend

### Frontend (Vanilla HTML/CSS/JS)
- No frameworks — pure HTML5, CSS3, JavaScript
- Drag-and-drop file upload with live preview
- Animated confidence meter
- Full chatbot with conversation history, typing indicator, quick prompts
- Responsive design, dark theme with glassmorphism elements

### Chatbot (PneumoAI)
- Powered by **Claude Sonnet** via Anthropic API
- System prompt specialized in: pneumonia pathophysiology, radiology, CURB-65, treatment protocols
- **Context-aware:** automatically receives the X-ray diagnosis result when the user asks "Ask AI about result"
- Maintains full conversation history per session

---

## 📊 API Endpoints

### `POST /predict`
```
Content-Type: multipart/form-data
Body: image (file)

Response:
{
  "label": "PNEUMONIA" | "NORMAL",
  "confidence": 94.21,
  "severity": "High Risk" | "Moderate Risk" | "Low Risk",
  "raw_score": 0.9421,
  "image": "<base64>"
}
```

### `POST /chat`
```json
{
  "messages": [
    { "role": "user", "content": "What are the symptoms of pneumonia?" }
  ],
  "diagnosis_context": "PNEUMONIA with 94.21% confidence"
}

Response:
{
  "reply": "Pneumonia symptoms include..."
}
```

---

## 🎓 For Your Professor / Conference

**What makes this conference-level:**

1. **Production-grade stack** — Flask REST API + model serving, not Streamlit toy
2. **Context-aware AI chatbot** — LLM receives model output and can explain findings
3. **Clean frontend architecture** — HTML/CSS/JS separated from backend logic
4. **REST API design** — `/predict` and `/chat` are proper endpoints, easy to scale
5. **Clinical grounding** — Risk stratification, recommendations, CURB-65 references
6. **Drag-and-drop UX** — Professional user experience
7. **No dependencies on notebooks** — Deployable on any Python server

---

## ⚠️ Disclaimer
This application is for **educational and academic purposes only**. It is not a medical device and should not be used for clinical diagnosis. Always consult a licensed physician.
