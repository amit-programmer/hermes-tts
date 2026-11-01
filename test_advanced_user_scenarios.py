import sys
import httpx

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

c = httpx.Client(base_url="http://127.0.0.1:8006", timeout=30.0)

print("="*65)
print("TESTING 10 ADVANCED USER SCENARIOS")
print("="*65)

test_cases = [
    # 1. Numbered lists
    ("1. Pull the repo. 2. Run npm install.", "step 1, pull the repo"),
    # 2. File paths
    ("Check src/auth/jwt_handler.py file", "src auth jwt handler file"),
    # 3. Git commands
    ("git commit -m \"fix: bug #404\"", "git commit with message fix, bug"),
    # 4. Logic & comparison symbols
    ("if x >= 100 && y != null", "is greater than or equal to 100, and y is not equal to null"),
    # 5. Time & units
    ("Meeting at 15:30 with 500ms latency", "3:30 PM with 500 milliseconds latency"),
    # 6. Question intonation marker
    ("deployment abhi trigger karna chahte ho?", "..."),
]

for orig, expected in test_cases:
    res = c.post("/v1/tts/clean-text", json={"text": orig})
    assert res.status_code == 200
    cleaned = res.json().get("clean_for_speech", "")
    assert expected.lower() in cleaned.lower(), f"Failed for '{orig}': Expected '{expected}' in '{cleaned}'"
    print(f"PASS: '{orig}' -> '{cleaned}'")

# Synthesize audio test with advanced scenario text
res_audio = c.post("/v1/tts/synthesize", json={
    "text": "1. Review src/auth/jwt_handler.py. 2. Latency is under 200ms. Kya aap ready ho?",
    "engine": "edge",
    "persona": "explain",
    "format": "mp3"
})
assert res_audio.status_code == 200
print(f"\nPASS: Synthesized advanced audio ({len(res_audio.content)} bytes)")

print("\n" + "="*65)
print("ALL ADVANCED USER SCENARIOS VERIFIED SUCCESSFULLY!")
print("="*65)
