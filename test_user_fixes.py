import sys
import httpx

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

c = httpx.Client(base_url="http://127.0.0.1:8006", timeout=30.0)

print("="*65)
print("TESTING 10 REAL-USER ISSUE FIXES")
print("="*65)

# 1. Number & Currency & Version & OTP Tests via clean-text
samples = [
    ("$500", "500 dollars"),
    ("₹1200", "1200 rupees"),
    ("v2.4.0", "version 2 point 4 point 0"),
    ("Your OTP is 98102", "9 8 1 0 2"),
    ("CI/CD pipeline and SQL database", "C I C D pipeline, and Sequel database"),
    ("GUI interface with OAuth2", "Gooey interface with O Auth 2"),
    ("Actually, server status is 99.9%", "Actually, server status is 99.9 percent"),
]

for orig, expected_sub in samples:
    res = c.post("/v1/tts/clean-text", json={"text": orig})
    assert res.status_code == 200
    cleaned = res.json().get("clean_for_speech", "")
    assert expected_sub.lower() in cleaned.lower(), f"Failed for '{orig}': Expected '{expected_sub}' in '{cleaned}'"
    print(f"PASS: '{orig}' -> '{cleaned}'")

# 2. Synthesize audio test with new expanded numbers & dev terms
res_audio = c.post("/v1/tts/synthesize", json={
    "text": "Your OTP is 98102. Paid $500 for v2.4.0 upgrade.",
    "engine": "edge",
    "persona": "alert",
    "format": "mp3"
})
assert res_audio.status_code == 200
print(f"\nPASS: Synthesized alert audio with numbers: {len(res_audio.content)} bytes")

# 3. Voices check includes new personas
res_voices = c.get("/v1/voices")
assert res_voices.status_code == 200
personas = res_voices.json().get("personas", [])
assert "alert" in personas and "empathetic" in personas and "explain" in personas
print("PASS: New personas active:", personas)

print("\n" + "="*65)
print("ALL 10 REAL-USER ISSUES VERIFIED FIXED!")
print("="*65)
