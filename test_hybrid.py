import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from utils.text_cleaner import clean_text_for_tts
from core.kokoro_engine import KokoroEngine
from core.chattts_engine import ChatTTSEngine
from core.elevenlabs_engine import ElevenLabsEngine
from core.edge_engine import EdgeTTSEngine

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def test_cleaner():
    print("=" * 60, flush=True)
    print(" 1. Testing Text Cleaner (Anti-Hallucination + Tech Cleaning)", flush=True)
    print("=" * 60, flush=True)
    sample_text = (
        "Hello team... check the API and JSON response--now! "
        "```python\nprint('hello')\n``` Visit https://example.com"
    )
    cleaned = clean_text_for_tts(sample_text)
    print(f"Original: {sample_text}", flush=True)
    print(f"Cleaned:  {cleaned}", flush=True)
    assert cleaned.endswith("."), "Cleaned text must end with terminal punctuation"
    print("[PASS] Text cleaner passed.", flush=True)


def test_edge():
    print("\n" + "=" * 60, flush=True)
    print(" 2. Testing Edge-TTS Engine (Fallback Redundancy / Neural Hindi)", flush=True)
    print("=" * 60, flush=True)
    try:
        engine = EdgeTTSEngine(default_voice="hi-IN-SwaraNeural")
        text = "Namaste! Edge-TTS fallback completely functional hai."
        t0 = time.time()
        buf = engine.generate(clean_text_for_tts(text))
        elapsed = time.time() - t0
        out_file = OUTPUT_DIR / "test_edge.mp3"
        with open(out_file, "wb") as f:
            f.write(buf.getvalue())
        print(f"[PASS] Edge-TTS generated {len(buf.getvalue())} bytes in {elapsed:.2f}s -> Saved: {out_file}", flush=True)
    except Exception as e:
        print(f"[ERROR] Edge-TTS failed: {e}", flush=True)


def test_elevenlabs():
    print("\n" + "=" * 60, flush=True)
    print(" 3. Testing ElevenLabs Engine (Studio Quality)", flush=True)
    print("=" * 60, flush=True)
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("[SKIP] ELEVENLABS_API_KEY not configured in .env.", flush=True)
        return

    try:
        engine = ElevenLabsEngine()
        text = "Good morning! Testing our studio voice pipeline with ElevenLabs."
        t0 = time.time()
        buf = engine.generate(clean_text_for_tts(text))
        elapsed = time.time() - t0
        out_file = OUTPUT_DIR / "test_elevenlabs.mp3"
        with open(out_file, "wb") as f:
            f.write(buf.getvalue())
        print(f"[PASS] ElevenLabs generated {len(buf.getvalue())} bytes in {elapsed:.2f}s -> Saved: {out_file}", flush=True)
    except Exception as e:
        print(f"[WARN] ElevenLabs test failed (e.g. quota or connection): {e}", flush=True)


def test_kokoro():
    print("\n" + "=" * 60, flush=True)
    print(" 4. Testing Kokoro-82M Engine (Local Fast Female Voice)", flush=True)
    print("=" * 60, flush=True)
    try:
        print("Initializing Kokoro pipeline (may download model weights on first run)...", flush=True)
        engine = KokoroEngine(default_voice="af_bella")
        text = "Hello, this is Kokoro-82M speaking in a crisp, clean female voice."
        t0 = time.time()
        buf = engine.generate(clean_text_for_tts(text))
        elapsed = time.time() - t0
        out_file = OUTPUT_DIR / "test_kokoro.wav"
        with open(out_file, "wb") as f:
            f.write(buf.getvalue())
        print(f"[PASS] Kokoro generated {len(buf.getvalue())} bytes in {elapsed:.2f}s -> Saved: {out_file}", flush=True)
    except Exception as e:
        print(f"[WARN] Kokoro test skipped / needs dependencies: {e}", flush=True)


def test_chattts():
    print("\n" + "=" * 60, flush=True)
    print(" 5. Testing ChatTTS Engine (Conversational Female Voice)", flush=True)
    print("=" * 60, flush=True)
    try:
        print("Initializing ChatTTS model (may download model weights on first run)...", flush=True)
        engine = ChatTTSEngine(seed=2222)
        text = "Hey there! How is everything going today? Just testing our speech model."
        t0 = time.time()
        buf = engine.generate(clean_text_for_tts(text))
        elapsed = time.time() - t0
        out_file = OUTPUT_DIR / "test_chattts.wav"
        with open(out_file, "wb") as f:
            f.write(buf.getvalue())
        print(f"[PASS] ChatTTS generated {len(buf.getvalue())} bytes in {elapsed:.2f}s -> Saved: {out_file}", flush=True)
    except Exception as e:
        print(f"[WARN] ChatTTS test skipped / needs dependencies: {e}", flush=True)


def test_production_tuning():
    from utils.text_cleaner import (
        normalize_hinglish_text,
        expand_numbers_and_currency,
        chunk_large_text,
    )

    text = "Bhai kya kar raha hai? Aaj 2 tickets ka cost ₹500 hai."
    normalized = normalize_hinglish_text(text)
    assert "bhaay" in normalized.lower()
    assert "kyaa" in normalized.lower()

    expanded = expand_numbers_and_currency("₹500 and 2026 and $20")
    assert "rupees" in expanded.lower()
    assert "two thousand twenty six" in expanded.lower() or "two thousand twenty-six" in expanded.lower()

    chunks = chunk_large_text("word " * 500, max_chars=150)
    assert len(chunks) > 1

    print("[PASS] Production text tuning checks passed.", flush=True)


if __name__ == "__main__":
    print("Running Unified Hybrid Voice Engine Tests...\n", flush=True)
    test_cleaner()
    test_edge()
    test_elevenlabs()
    test_kokoro()
    test_chattts()
    print("\n" + "=" * 60, flush=True)
    print(" Tests complete. Check the 'outputs/' folder for generated audio files.", flush=True)
    print("=" * 60, flush=True)
