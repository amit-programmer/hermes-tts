import sys
import httpx

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

c = httpx.Client(base_url="http://127.0.0.1:8006", timeout=35.0)

print("="*65)
print("TESTING 10 NEW REAL-USER SCENARIOS (V2)")
print("="*65)

test_cases = [
    # 1. URLs, Localhost & Ports
    ("Server running on http://localhost:3000", "localhost on port 3000"),
    ("Backend at http://192.168.1.5:8000", "I P 192 dot 168 dot 1 dot 5, on port 8000"),
    # 2. Tech acronyms (SSH, DNS, NPM, PIP)
    ("Check SSH keys and DNS config with NPM", "S S H keys, and D N S config with N P M"),
    # 3. Status Codes & Hexadecimal
    ("HTTP 404 Not Found at 0x7FFE", "four oh four Not Found at Hex 0 x 7 F F E"),
    ("Received 500 error", "five hundred error"),
    # 4. Multi-line Key-Value status pairs
    ("CPU: 45%\nMemory: 3.2 GB\nStatus: Running", "C P U, 45 percent.\nMemory, 3.2 gigabytes.\nStatus, Running."),
    # 5. Parentheses side-notes
    ("Database drop ho jayega (production me mat run karna) samjhe.", "Database drop ho jayega, production me mat run karna, samjhe."),
    # 6. Dates
    ("Release date 2026-11-15 fix hui hai", "15th November 2026"),
    # 7. Negative numbers
    ("Loss -5.4% hua", "Loss minus 5.4 percent hua"),
    # 8. AI Verbose filler preamble
    ("Sure! I would be happy to help you with that. Build complete ho gaya.", "Build complete ho gaya."),
    # 9. Emojis and emotion markers
    ("Server start ho gaya 🚀 Warning ⚠️ Build pass ✅", "Warning Caution, Build pass Success,"),
]

for orig, expected in test_cases:
    res = c.post("/v1/tts/clean-text", json={"text": orig})
    assert res.status_code == 200, f"Error {res.status_code} for text: {orig}"
    cleaned = res.json().get("clean_for_speech", "")
    assert expected.lower() in cleaned.lower(), f"Failed for '{orig}': Expected '{expected}' in '{cleaned}'"
    print(f"PASS: '{orig}' -> '{cleaned}'")

# 10. Synthesize audio test combining several new real-user features
res_audio = c.post("/v1/tts/synthesize", json={
    "text": "Sure! I would be happy to help. Server http://localhost:8080 par 15th November 2026 ko deploy hoga. Error -2.5% hai. ✅",
    "engine": "edge",
    "persona": "conversational",
    "format": "mp3"
})
assert res_audio.status_code == 200, f"Failed audio synthesis: {res_audio.text}"
print(f"\nPASS: Synthesized real-user V2 audio ({len(res_audio.content)} bytes)")

print("\n" + "="*65)
print("ALL 10 NEW REAL-USER SCENARIOS PASSED 100%!")
print("="*65)
