---
title: Hermes Voice Assistant Studio
emoji: 🎙️
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: mit
---

# 🎙️ Hermes AI Voice Assistant Studio

[![Gradio](https://img.shields.io/badge/UI-Gradio%20SDK-orange.svg)](https://gradio.app)
[![FastAPI](https://img.shields.io/badge/API-FastAPI%20REST-009688.svg)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Production-ready, unified Voice Assistant server with **Gradio Studio Web UI** and **FastAPI REST API**. Integrates **Whisper Speech-to-Text (STT)** with **Kokoro-82M, Edge-TTS, and ElevenLabs Text-to-Speech (TTS)**, accelerated on ZeroGPU (NVIDIA RTX Pro 6000 Blackwell) or CPU.

---

## 🌟 Key Highlights

- **Full Voice Cycle (TTS + STT):** Complete end-to-end speech recognition and speech synthesis pipeline in a single container.
- **Whisper Speech-to-Text (STT):** Multilingual speech transcription for Hindi, Hinglish, English, and 90+ languages. ZeroGPU Blackwell GPU accelerated.
- **OpenAI-Compatible STT API:** `/v1/audio/transcriptions` endpoint works as a direct drop-in replacement for OpenAI Whisper in LangChain, AutoGPT, and mobile apps.
- **Hinglish & English TTS Auto-Routing:** Romanized Hindi and Hinglish queries are routed to Indian neural voices (`hi-IN-SwaraNeural`), while pure English queries run locally on Kokoro-82M.
- **Ultra-Fast Chunk Streaming:** Sentence-level producer-consumer pipeline with Time-To-First-Audio (TTFA) < 400ms.
- **Developer Lexicon Engine:** Automatically expands technical terms (`API`, `JWT`, `SQL`, `SSH`, `DNS`), file paths (`src/auth/jwt.py`), terminal commands (`git commit -m`), logic operators (`&&`, `!=`), 24h times, and currency symbols (`$`, `₹`).
- **Smooth Audio Boundary Transitions:** 4ms cosine fade in/out prevents boundary clicks and pops during chunk stitching.
- **In-Memory LRU Audio Cache:** Sub-10ms response times on repeated phrases without disk I/O.
- **Multi-Format Conversion:** Supports `MP3` (Telegram bot audio), `OGG/Opus` (Voice notes), `WAV`, and `PCM`.

---

## 🏗️ Architecture Overview

```text
                           [ User / Client Request ]
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
       [ Web Browser ]                             [ API Consumers ]
       Gradio Studio UI (/)                     FastAPI REST (/v1/...)
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      ▼
                        [ Text Normalization & Prosody ]
                     (Dev Lexicon, Pacing, Pauses, Emojis)
                                      │
               ┌──────────────────────┼──────────────────────┐
               ▼                      ▼                      ▼
        [ Kokoro-82M ]           [ Edge-TTS ]         [ ElevenLabs ]
         (Local CPU)             (Neural Cloud)          (Studio API)
        Pure English          Hinglish / Indian         Ultra-Premium
               │                      │                      │
               └──────────────────────┼──────────────────────┘
                                      ▼
                      [ Cosine Fade Boundary Stitching ]
                                      │
                                      ▼
                          [ -16 LUFS Normalization ]
                                      │
                                      ▼
                      [ In-Memory Binary Buffer Output ]
```

---

## 🚀 Quick Start (Local Development)

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/hermes-voice.git
cd hermes-voice
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
*(Optionally add your `ELEVENLABS_API_KEY` in `.env` if you wish to use ElevenLabs).*

### 4. Start the Unified Studio Server
```bash
python app.py
```
- **Interactive Gradio Studio:** Open [http://localhost:8006](http://localhost:8006)
- **Interactive Swagger Docs:** Open [http://localhost:8006/docs](http://localhost:8006/docs)
- **Health Check:** Open [http://localhost:8006/health](http://localhost:8006/health)

---

## 🌐 Deploy to Hugging Face Spaces (Gradio SDK)

### Step 1: Create a New Space
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Set Space Name: e.g. `hermes-voice-studio`
3. Space SDK: **`Gradio`**
4. Hardware: **`Free (2 vCPU, 16 GB RAM)`**
5. Click **Create Space**.

### Step 2: Push Your Code
Add Hugging Face as a remote and push:
```bash
git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/hermes-voice-studio
git push space main
```

### Step 3: Add Secrets (Optional)
In Space **Settings** -> **Variables and secrets**:
- `ELEVENLABS_API_KEY`: `sk_...` (Optional)
- `KOKORO_VOICE`: `af_bella`
- `EDGE_DEFAULT_VOICE`: `hi-IN-SwaraNeural`

---

## 📚 REST API Reference

### 1. Synthesize Speech (`POST /v1/tts/synthesize`)
Generates in-memory audio binary buffer.
```bash
curl -X POST "https://<your-space>.hf.space/v1/tts/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Server port 8080 par running hai. Build passed successfully! 🚀",
    "engine": "auto",
    "persona": "conversational",
    "format": "mp3"
  }' --output voice.mp3
```

### 2. Sentence-Level Streaming (`POST /v1/tts/stream`)
Real-time chunked streaming with TTFA < 400ms.
```bash
curl -X POST "https://<your-space>.hf.space/v1/tts/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "First sentence streams immediately. Second sentence follows.",
    "format": "mp3"
  }' --no-buffer --output stream.mp3
```

### 3. Clean Text & Prosody Preview (`POST /v1/tts/clean-text`)
```bash
curl -X POST "https://<your-space>.hf.space/v1/tts/clean-text" \
  -H "Content-Type: application/json" \
  -d '{"text": "git commit -m \"fix\" and check 500 error"}'
```

### 4. Voices & Personas (`GET /v1/voices`)
```bash
curl -X GET "https://<your-space>.hf.space/v1/voices"
```

---

## ⚙️ Voice Personas & Profiles

| Persona Preset | Rate Adjustment | Pitch Adjustment | Best Used For |
| :--- | :--- | :--- | :--- |
| `conversational` | `+2%` | `+3Hz` | Natural daily voice assistant conversations |
| `code_review` | `+12%` | `+0Hz` | Fast-paced technical explanations |
| `alert` | `+15%` | `+4Hz` | System warnings and urgent notifications |
| `serious` | `+10%` | `-1Hz` | Professional reports and architecture updates |
| `empathetic` | `-4%` | `-1Hz` | Calm, helpful assistant guidance |
| `explain` | `-6%` | `+0Hz` | Step-by-step tutorials and numbered guides |
| `fast` | `+25%` | `+0Hz` | High-speed summaries |
| `slow` | `-15%` | `+0Hz` | Clear diction and accessibility |

---

## 🛡️ License
MIT License. Free for open-source and commercial voice integrations.
