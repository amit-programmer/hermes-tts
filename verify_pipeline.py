# Save as verify_pipeline.py in edge/
import os
import requests

PORT = 8005
BASE_URL = f"http://localhost:{PORT}"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_tests():
    print(f"[1] Testing /health endpoint...")
    try:
        h = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Health Response: {h.json()}")
    except Exception as e:
        print(f"Server not responding on port {PORT}: {e}")
        return

    # Test Kokoro
    print("\n[2] Testing Kokoro Engine...")
    r1 = requests.post(
        f"{BASE_URL}/synthesize",
        json={"text": "Hello, testing Kokoro girl voice pipeline.", "engine": "kokoro"}
    )
    if r1.status_code == 200:
        with open(f"{OUTPUT_DIR}/kokoro_verified.wav", "wb") as f:
            f.write(r1.content)
        print(f"Kokoro test passed! Engine used: {r1.headers.get('X-Engine-Used')}")
    else:
        print(f"Kokoro test failed: {r1.text}")

    # Test ChatTTS
    print("\n[3] Testing ChatTTS Engine...")
    r2 = requests.post(
        f"{BASE_URL}/synthesize",
        json={"text": "Hey there! How is everything going today?", "engine": "chattts"}
    )
    if r2.status_code == 200:
        with open(f"{OUTPUT_DIR}/chattts_verified.wav", "wb") as f:
            f.write(r2.content)
        print(f"ChatTTS test passed! Engine used: {r2.headers.get('X-Engine-Used')}")
    else:
        print(f"ChatTTS test failed: {r2.text}")

    # Test Edge-TTS
    print("\n[4] Testing Edge-TTS Engine...")
    r3 = requests.post(
        f"{BASE_URL}/synthesize",
        json={"text": "Namaste! Edge TTS Hindi voice test.", "engine": "edge"}
    )
    if r3.status_code == 200:
        with open(f"{OUTPUT_DIR}/edge_verified.mp3", "wb") as f:
            f.write(r3.content)
        print(f"Edge-TTS test passed! Engine used: {r3.headers.get('X-Engine-Used')}")
    else:
        print(f"Edge-TTS test failed: {r3.text}")

if __name__ == "__main__":
    run_tests()
