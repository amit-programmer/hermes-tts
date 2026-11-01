import time
import io
import httpx
import soundfile as sf

c = httpx.Client(base_url="http://127.0.0.1:8006", timeout=30.0)

print("="*65)
print("TEST 1: SINGLE-WORD / VERY SHORT REPLIES CRASH GUARD")
print("="*65)
short_words = ["Sure.", "Done!", "Wait.", "OK."]

for w in short_words:
    t0 = time.time()
    res = c.post("/v1/tts/synthesize", json={
        "text": w,
        "engine": "edge",
        "format": "mp3"
    })
    assert res.status_code == 200, f"Failed on '{w}': {res.status_code} - {res.text}"
    print(f"PASS: Short word '{w}' -> Status {res.status_code} | Bytes: {len(res.content)} | Time: {(time.time()-t0)*1000:.1f}ms")

print("\n" + "="*65)
print("TEST 2: MP3 STREAMING WITH CLEAN MPEG FRAMES (NO ID3 STALL)")
print("="*65)
stream_text = "Sentence one is starting. Sentence two continues smoothly. Sentence three completes the thought."
res_stream = c.post("/v1/tts/stream", json={
    "text": stream_text,
    "engine": "edge",
    "format": "mp3"
})
assert res_stream.status_code == 200, f"Stream failed: {res_stream.status_code}"
stream_bytes = res_stream.content
print(f"PASS: Multi-sentence MP3 stream received: {len(stream_bytes)} bytes")
# Verify it decodes cleanly in soundfile
stream_buf = io.BytesIO(stream_bytes)
data, sr = sf.read(stream_buf)
print(f"PASS: Decoded seamlessly! Sample Rate: {sr}Hz | Length: {len(data)} samples ({len(data)/sr:.2f}s)")

print("\n" + "="*65)
print("TEST 3: NATIVE PCM STREAMING (Zero-Container Live Stream)")
print("="*65)
res_pcm = c.post("/v1/tts/stream", json={
    "text": "Streaming raw PCM samples directly to client audio player.",
    "engine": "edge",
    "format": "pcm"
})
assert res_pcm.status_code == 200, f"PCM stream failed: {res_pcm.status_code}"
assert res_pcm.headers.get("content-type") == "audio/pcm", f"Unexpected content-type: {res_pcm.headers.get('content-type')}"
print(f"PASS: PCM Stream OK! Header Content-Type: {res_pcm.headers.get('content-type')} | Bytes: {len(res_pcm.content)}")

print("\n" + "="*65)
print("ALL 3 PRODUCTION SUBTLETY TESTS PASSED FLAWLESSLY!")
print("="*65)
