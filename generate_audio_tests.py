import time
import httpx
import soundfile as sf

c = httpx.Client(base_url="http://127.0.0.1:8006", timeout=30.0)

print("="*60)
print("TEST 1: GENERATING test_hinglish.mp3 (Auto Mode -> SwaraNeural)")
print("="*60)
t0 = time.time()
res1 = c.post("/v1/tts/synthesize", json={
    "text": "Maine memory leak fix kar di hai. PR review kar lo.",
    "engine": "auto",
    "persona": "code_review",
    "format": "mp3"
})
assert res1.status_code == 200, f"Error: {res1.status_code} - {res1.text}"
with open("test_hinglish.mp3", "wb") as f:
    f.write(res1.content)

print(f"Hinglish OK! File: test_hinglish.mp3 | Bytes: {len(res1.content)} | Time: {(time.time() - t0)*1000:.1f}ms")
print("Engine Used:", res1.headers.get("x-engine-used"))
print("Voice Used:", res1.headers.get("x-voice-used"))

print("\n" + "="*60)
print("TEST 2: GENERATING test_english.wav (English Tech -> Kokoro / Edge WAV)")
print("="*60)
t0 = time.time()
res2 = c.post("/v1/tts/synthesize", json={
    "text": "The asynchronous queue is now processing sentences in real time.",
    "engine": "auto",
    "format": "wav",
    "persona": "casual_chat"
})
assert res2.status_code == 200, f"Error: {res2.status_code} - {res2.text}"
with open("test_english.wav", "wb") as f:
    f.write(res2.content)

# Verify WAV properties
info = sf.info("test_english.wav")
print(f"English WAV OK! File: test_english.wav | Bytes: {len(res2.content)} | Duration: {info.duration:.2f}s | Sample Rate: {info.samplerate}Hz | Channels: {info.channels}")
print("Engine Used:", res2.headers.get("x-engine-used"))
print("Voice Used:", res2.headers.get("x-voice-used"))

print("\n" + "="*60)
print("BOTH AUDIO FILES SUCCESSFULLY GENERATED AND VERIFIED!")
print("="*60)
