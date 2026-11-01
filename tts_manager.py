import asyncio
from typing import AsyncGenerator, List, Dict, Any, Optional
from pathlib import Path
import edge_tts

from config import (
    DEFAULT_VOICE,
    DEFAULT_RATE,
    DEFAULT_PITCH,
    DEFAULT_VOLUME,
    OUTPUT_DIR,
)


class EdgeTTSManager:
    """
    Manager class to handle Edge-TTS text-to-speech generation,
    file saving, streaming chunks, and voice listing.
    """

    def __init__(
        self,
        default_voice: str = DEFAULT_VOICE,
        default_rate: str = DEFAULT_RATE,
        default_pitch: str = DEFAULT_PITCH,
        default_volume: str = DEFAULT_VOLUME,
    ):
        self.default_voice = default_voice
        self.default_rate = default_rate
        self.default_pitch = default_pitch
        self.default_volume = default_volume

    async def save_speech(
        self,
        text: str,
        output_file: Optional[Path | str] = None,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
        volume: Optional[str] = None,
    ) -> Path:
        """
        Synthesizes speech and saves directly to an MP3 file.
        """
        voice = voice or self.default_voice
        rate = rate or self.default_rate
        pitch = pitch or self.default_pitch
        volume = volume or self.default_volume

        if output_file is None:
            output_path = OUTPUT_DIR / "output.mp3"
        else:
            output_path = Path(output_file)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
            volume=volume,
        )

        await communicate.save(str(output_path))
        return output_path

    async def stream_audio_chunks(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
        volume: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        In-memory streaming generator. Yields raw MP3 audio chunks.
        Suitable for backend servers (FastAPI / Flask / WebSocket).
        """
        voice = voice or self.default_voice
        rate = rate or self.default_rate
        pitch = pitch or self.default_pitch
        volume = volume or self.default_volume

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
            volume=volume,
        )

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    @staticmethod
    async def get_available_voices(locale_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetches all available voices from Microsoft Edge endpoints,
        with optional filtering by locale (e.g., 'hi-IN', 'en-IN', 'en-US').
        """
        voices = await edge_tts.list_voices()
        if locale_filter:
            locale_lower = locale_filter.lower()
            return [v for v in voices if locale_lower in v.get("Locale", "").lower()]
        return voices
