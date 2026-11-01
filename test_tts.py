import asyncio
from pathlib import Path
from tts_engine import TTSEngine, clean_text_for_tts

SAMPLE_HERMES_OUTPUT = """
Maine API endpoint check kar liya hai. 
Aapka JSON payload format sahi hai, par SQL query me index missing hai.

```sql
SELECT * FROM users WHERE email = 'test@example.com';
```

Is code ko add karo aur PR create kar do. More details at [Documentation](https://docs.hermes-ai.internal).
"""


async def run_test():
    print("=" * 65)
    print(" Hermes AI Output Text-Cleaner Test")
    print("=" * 65)
    print("RAW INPUT:")
    print(SAMPLE_HERMES_OUTPUT.strip())
    print("-" * 65)

    cleaned = clean_text_for_tts(SAMPLE_HERMES_OUTPUT)
    print("CLEANED SPEAKABLE TEXT:")
    print(cleaned)
    print("=" * 65)

    # Initialize Engine with Hinglish Female voice (Swara)
    engine = TTSEngine(profile_name="hinglish_female")

    # 1. Test In-Memory RAM BytesIO Generation
    print("\n[1/2] Generating in-memory RAM audio buffer (Zero disk lag)...")
    audio_buffer = await engine.generate_audio_bytes(SAMPLE_HERMES_OUTPUT)
    print(f"  Buffer size in RAM: {len(audio_buffer.getvalue())} bytes")

    # 2. Test Disk File Save
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = str(output_dir / "test_hermes_voice.mp3")

    print(f"\n[2/2] Saving speech to file: {output_file} ...")
    await engine.save_to_file(SAMPLE_HERMES_OUTPUT, output_file)
    print(f"  [SUCCESS] Audio file successfully created: {output_file}")


if __name__ == "__main__":
    asyncio.run(run_test())
