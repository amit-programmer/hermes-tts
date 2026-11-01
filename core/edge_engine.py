import asyncio
import io
import os
from typing import Optional

import numpy as np
import edge_tts


def apply_safe_ending(audio_data: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
    """Add a brief silence envelope to prevent trailing clicks and abrupt endings."""
    if audio_data.size == 0:
        return audio_data
    silence_samples = int(sample_rate * 0.20)
    silence = np.zeros(silence_samples, dtype=audio_data.dtype)
    return np.concatenate([audio_data, silence])


class EdgeTTSEngine:
    """
    Microsoft Edge Neural TTS Engine:
    Zero-cost, cloud neural voices with native Hindi, Hinglish, and English support.
    Requires 0 local GPU/CPU weights.
    """

    def __init__(self, default_voice: Optional[str] = None):
        self.default_voice = default_voice or os.getenv("EDGE_DEFAULT_VOICE", "hi-IN-SwaraNeural")

    async def stream_audio_async(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
        max_retries: int = 2,
    ):
        """Yields audio bytes as they stream in real-time from Microsoft Edge-TTS with auto-retry."""
        import socket
        import aiohttp

        target_voice = voice or self.default_voice
        last_error = None

        for attempt in range(max_retries):
            try:
                connector = aiohttp.TCPConnector(family=socket.AF_INET)
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=target_voice,
                    rate=rate,
                    pitch=pitch,
                    volume=volume,
                    connector=connector,
                )
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        yield chunk["data"]
                return
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.25 * (attempt + 1))
                else:
                    raise last_error

    async def generate_async(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
        max_retries: int = 2,
    ) -> io.BytesIO:
        """
        Synthesizes text into an in-memory MP3 audio buffer asynchronously.
        SwaraNeural pronounces Hinglish and Hindi with a native Indian accent.
        Uses IPv4 connector to bypass Windows IPv6 resolution latency.
        Includes auto-reconnect retry for network resilience.
        """
        import socket
        import aiohttp

        target_voice = voice or self.default_voice
        last_error = None

        for attempt in range(max_retries):
            try:
                connector = aiohttp.TCPConnector(family=socket.AF_INET)
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=target_voice,
                    rate=rate,
                    pitch=pitch,
                    volume=volume,
                    connector=connector,
                )

                buffer = io.BytesIO()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        buffer.write(chunk["data"])

                buffer.seek(0)
                if buffer.getbuffer().nbytes > 0:
                    return buffer
                else:
                    raise RuntimeError("Edge-TTS returned empty audio stream.")
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.25 * (attempt + 1))
                else:
                    raise last_error

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
    ) -> io.BytesIO:
        """
        Synchronous wrapper for generate_async.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(self.generate_async(text, voice, rate, pitch, volume))
        else:
            return loop.run_until_complete(self.generate_async(text, voice, rate, pitch, volume))
