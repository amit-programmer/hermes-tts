import asyncio
import edge_tts
from pathlib import Path

# Sample Text (Hindi & Roman Hinglish compatible)
TEXT = "Namaste! Aaj hum Edge-TTS ka setup dekh rahe hain. Ye Hindi aur Hinglish dono achhe se bol leta hai."

# hi-IN-SwaraNeural is the best voice for Hindi & Hinglish
VOICE = "hi-IN-SwaraNeural"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = str(OUTPUT_DIR / "basic_output.mp3")


async def generate_speech():
    print(f"Synthesizing speech...")
    print(f"Voice: {VOICE}")
    print(f"Text: {TEXT}\n")

    # Communicate object initialize karein
    communicate = edge_tts.Communicate(TEXT, VOICE)

    # Audio generate karke disk par save karein
    await communicate.save(OUTPUT_FILE)
    print(f"[SUCCESS] Audio save ho gaya: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(generate_speech())
