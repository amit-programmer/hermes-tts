import asyncio
import edge_tts
from pathlib import Path

TEXT = "Namaste! Yeh advanced controls ka demo hai. Speed thodi badhayi gayi hai aur pitch adjust kiya gaya hai."
VOICE = "hi-IN-SwaraNeural"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = str(OUTPUT_DIR / "advanced_output.mp3")


async def generate_advanced_speech():
    print("Advanced Controls:")
    print("Rate: +10% (Thoda tez speed)")
    print("Pitch: +2Hz (Voice clarity/sharpness)")
    print("Volume: +5% (Thodi zyada loudness)")
    print(f"Voice: {VOICE}\n")

    communicate = edge_tts.Communicate(
        text=TEXT,
        voice=VOICE,
        rate="+10%",      # Speed control (+10%, -15%, etc.)
        pitch="+2Hz",     # Pitch control (+5Hz, -5Hz, etc.)
        volume="+5%",     # Loudness control (+10%, -10%, etc.)
    )

    await communicate.save(OUTPUT_FILE)
    print(f"[SUCCESS] Advanced audio save ho gaya: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(generate_advanced_speech())
