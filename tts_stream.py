import asyncio
from typing import AsyncGenerator
from pathlib import Path
import edge_tts

TEXT = "In-memory streaming demo. Audio chunks stream ho rahe hain bina pehle file me write kiye."
VOICE = "hi-IN-SwaraNeural"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
STREAM_DUMP_FILE = OUTPUT_DIR / "stream_dump.mp3"


async def stream_audio_chunks(text: str, voice: str) -> AsyncGenerator[bytes, None]:
    """
    In-memory generator: Yields raw audio bytes chunk by chunk.
    Backend framework jaise FastAPI me StreamingResponse ke sath directly use kar sakte hain.
    """
    communicate = edge_tts.Communicate(text, voice)
    async for chunk in communicate.stream():
        # Audio bytes filter karein (metadata events chhod kar)
        if chunk["type"] == "audio":
            yield chunk["data"]


async def main():
    print("Starting in-memory stream...")
    total_bytes = 0
    chunk_count = 0

    # Stream chunks collect karke verify karne ke liye file me write kar rahe hain
    with open(STREAM_DUMP_FILE, "wb") as f:
        async for chunk in stream_audio_chunks(TEXT, VOICE):
            chunk_count += 1
            total_bytes += len(chunk)
            f.write(chunk)
            print(f"Received Chunk #{chunk_count}: {len(chunk)} bytes")

    print(f"\n[SUCCESS] Stream complete! Received {chunk_count} chunks, Total: {total_bytes} bytes.")
    print(f"Saved stream test file at: {STREAM_DUMP_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
