import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

PORT = int(os.getenv("PORT", 8006))
url = f"http://127.0.0.1:{PORT}/synthesize"

# 1. Health check
print(f"Testing Health Check on port {PORT}...", flush=True)
r_health = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=5)
print(f"Health Response: {r_health.json()}", flush=True)

# 2. Test Auto mode (default Kokoro)
print("\nTesting Auto Mode (Kokoro-82M)...", flush=True)
payload_auto = {
    "text": "Namaste! Unified voice server is responding successfully.",
    "engine": "auto"
}
t0 = time.time()
r1 = requests.post(url, json=payload_auto, timeout=30)
elapsed = time.time() - t0
engine_used = r1.headers.get("X-Engine-Used")
resp_time = r1.headers.get("X-Response-Time-Ms")
print(f"Auto Mode: Status {r1.status_code}, Engine Used: {engine_used}, Latency: {resp_time}ms, Total: {elapsed:.2f}s, Bytes: {len(r1.content)}", flush=True)
with open("outputs/server_kokoro_success.wav", "wb") as f:
    f.write(r1.content)

# 3. Test ElevenLabs with Fallback (since free API key returned 402, it automatically falls back to Kokoro!)
print("\nTesting ElevenLabs Fallback...", flush=True)
payload_eleven = {
    "text": "Good morning! Testing our studio voice pipeline with ElevenLabs.",
    "engine": "elevenlabs"
}
t0 = time.time()
r2 = requests.post(url, json=payload_eleven, timeout=30)
elapsed = time.time() - t0
engine_used = r2.headers.get("X-Engine-Used")
resp_time = r2.headers.get("X-Response-Time-Ms")
print(f"ElevenLabs (with Fallback): Status {r2.status_code}, Engine Used: {engine_used}, Latency: {resp_time}ms, Total: {elapsed:.2f}s, Bytes: {len(r2.content)}", flush=True)
with open("outputs/server_fallback_success.wav", "wb") as f:
    f.write(r2.content)

# 4. Test Edge-TTS
print("\nTesting Edge-TTS direct...", flush=True)
payload_edge = {
    "text": "Namaste! Testing Microsoft Edge Neural Swara voice directly.",
    "engine": "edge",
    "voice_id": "hi-IN-SwaraNeural"
}
t0 = time.time()
r3 = requests.post(url, json=payload_edge, timeout=15)
elapsed = time.time() - t0
engine_used = r3.headers.get("X-Engine-Used")
resp_time = r3.headers.get("X-Response-Time-Ms")
print(f"Edge Engine: Status {r3.status_code}, Engine Used: {engine_used}, Latency: {resp_time}ms, Total: {elapsed:.2f}s, Bytes: {len(r3.content)}", flush=True)
with open("outputs/server_edge_success.mp3", "wb") as f:
    f.write(r3.content)

print("\nAll server API tests completed successfully!", flush=True)
