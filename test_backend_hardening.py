import time
import httpx

BASE_URL = "http://127.0.0.1:8006"
c = httpx.Client(base_url=BASE_URL, timeout=30.0)

print("="*65)
print("AUDIT TEST 1: HEALTH CHECK (GET /health)")
print("="*65)
res_health = c.get("/health")
assert res_health.status_code == 200, f"Health check failed: {res_health.status_code}"
print("Status:", res_health.status_code, "| Body:", res_health.json())

print("\n" + "="*65)
print("AUDIT TEST 2: TEXT CLEANING (POST /v1/tts/clean-text)")
print("="*65)
res_clean = c.post("/v1/tts/clean-text", json={"text": "Review this `calculateTotal()` and **fix** bug!"})
assert res_clean.status_code == 200, f"Clean text failed: {res_clean.status_code}"
print("Status:", res_clean.status_code, "| Cleaned:", res_clean.json().get("clean_for_speech"))

print("\n" + "="*65)
print("AUDIT TEST 3: SYNTHESIZE NORMAL & EDGE CASES")
print("="*65)
# A. Normal
res_syn_normal = c.post("/v1/tts/synthesize", json={
    "text": "Server hardening is complete.",
    "engine": "edge",
    "format": "mp3"
})
assert res_syn_normal.status_code == 200, f"Normal synthesize failed: {res_syn_normal.status_code}"
print(f"PASS: Normal Synthesize -> Status {res_syn_normal.status_code} | Bytes: {len(res_syn_normal.content)}")

# B. Empty text -> 422
res_syn_empty = c.post("/v1/tts/synthesize", json={"text": "   ", "engine": "edge"})
assert res_syn_empty.status_code == 422, f"Expected 422 for empty, got {res_syn_empty.status_code}"
print(f"PASS: Empty Payload Rejected -> Status {res_syn_empty.status_code} | Detail: {res_syn_empty.json()}")

# C. Oversized text (> 2500 chars) -> 422
huge_text = "Hermes test sentence. " * 150  # ~3300 chars
res_syn_huge = c.post("/v1/tts/synthesize", json={"text": huge_text, "engine": "edge"})
assert res_syn_huge.status_code == 422, f"Expected 422 for oversized, got {res_syn_huge.status_code}"
print(f"PASS: Oversized Payload Rejected -> Status {res_syn_huge.status_code} | Detail: {res_syn_huge.json()}")

print("\n" + "="*65)
print("AUDIT TEST 4: STREAMING NORMAL, EDGE CASES & RAPID DISCONNECT")
print("="*65)
# A. Normal Stream
res_stream_normal = c.post("/v1/tts/stream", json={
    "text": "Sentence A is validated. Sentence B is streaming.",
    "engine": "edge",
    "format": "mp3"
})
assert res_stream_normal.status_code == 200, f"Normal stream failed: {res_stream_normal.status_code}"
print(f"PASS: Normal Stream -> Status {res_stream_normal.status_code} | Bytes: {len(res_stream_normal.content)}")

# B. Stream Empty text -> 422
res_stream_empty = c.post("/v1/tts/stream", json={"text": "", "engine": "edge"})
assert res_stream_empty.status_code == 422, f"Expected 422 for stream empty, got {res_stream_empty.status_code}"
print(f"PASS: Stream Empty Rejected -> Status {res_stream_empty.status_code}")

# C. Stream Oversized text -> 422
res_stream_huge = c.post("/v1/tts/stream", json={"text": huge_text, "engine": "edge"})
assert res_stream_huge.status_code == 422, f"Expected 422 for stream huge, got {res_stream_huge.status_code}"
print(f"PASS: Stream Oversized Rejected -> Status {res_stream_huge.status_code}")

# D. Rapid Consumer Disconnect Simulation:
# Open stream of 5 sentences, read only first chunk, and close connection immediately
multi_sentence = "One first sentence. Two second sentence. Three third sentence. Four fourth sentence. Five fifth sentence."
with c.stream("POST", "/v1/tts/stream", json={"text": multi_sentence, "engine": "edge", "format": "mp3"}) as stream_res:
    assert stream_res.status_code == 200
    for chunk in stream_res.iter_bytes():
        print(f"Client received first chunk ({len(chunk)} bytes) -> Aborting connection now!")
        break  # Disconnect immediately

time.sleep(1.0)
# Verify server is still alive and responsive after mid-stream abort
health_after_abort = c.get("/health")
assert health_after_abort.status_code == 200
print(f"PASS: Server healthy and responsive after rapid client abort -> Status {health_after_abort.status_code}")

print("\n" + "="*65)
print("ALL BACKEND HARDENING AUDIT TESTS PASSED 100%!")
print("="*65)
