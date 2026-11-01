import io
import os
import shutil
import subprocess
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, Optional

import edge_tts

from .cleaner import clean_text_for_tts, chunk_text
from .voice_config import VOICE_PROFILES, DEFAULT_PROFILE


class TTSEngine:
    """
    Production-ready isolated Edge-TTS Micro-Engine.
    Cleans developer model output and produces ready-to-stream audio buffers
    or saved audio files.
    """

    def __init__(
        self,
        profile_name: str = DEFAULT_PROFILE,
        custom_voice: Optional[str] = None,
        custom_rate: Optional[str] = None,
        custom_pitch: Optional[str] = None,
        custom_volume: Optional[str] = None,
    ):
        base_config = VOICE_PROFILES.get(profile_name, VOICE_PROFILES[DEFAULT_PROFILE])
        self.profile_name = profile_name
        self.config: Dict[str, Any] = {
            "voice": custom_voice or base_config["voice"],
            "rate": custom_rate or base_config["rate"],
            "pitch": custom_pitch or base_config["pitch"],
            "volume": custom_volume or base_config["volume"],
        }

    async def generate_audio_bytes(self, text: str, apply_cleaning: bool = True) -> io.BytesIO:
        """
        Disk use kiye bina direct RAM me audio buffer banata hai (In-Memory Streaming).
        Ideal for Telegram bots, web sockets, or streaming REST APIs.
        """
        prompt = clean_text_for_tts(text) if apply_cleaning else text.strip()
        if not prompt:
            return io.BytesIO()

        communicate = edge_tts.Communicate(
            text=prompt,
            voice=self.config["voice"],
            rate=self.config["rate"],
            pitch=self.config["pitch"],
            volume=self.config["volume"],
        )

        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_buffer.seek(0)
        return audio_buffer

    async def save_to_file(self, text: str, output_path: str, apply_cleaning: bool = True) -> str:
        """
        Synthesizes speech and saves directly to disk (.mp3).
        """
        prompt = clean_text_for_tts(text) if apply_cleaning else text.strip()
        if not prompt:
            raise ValueError("Text is empty or became empty after cleaning.")

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        communicate = edge_tts.Communicate(
            text=prompt,
            voice=self.config["voice"],
            rate=self.config["rate"],
            pitch=self.config["pitch"],
            volume=self.config["volume"],
        )

        await communicate.save(str(out_file))
        return str(out_file)

    async def stream_audio_chunks(
        self, text: str, apply_cleaning: bool = True
    ) -> AsyncGenerator[bytes, None]:
        """
        Yields raw audio chunks in real-time.
        """
        prompt = clean_text_for_tts(text) if apply_cleaning else text.strip()
        if not prompt:
            return

        communicate = edge_tts.Communicate(
            text=prompt,
            voice=self.config["voice"],
            rate=self.config["rate"],
            pitch=self.config["pitch"],
            volume=self.config["volume"],
        )

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    @staticmethod
    def convert_to_ogg_opus(input_mp3_path: str, output_ogg_path: str) -> bool:
        """
        Helper for Telegram voice notes format:
        Converts MP3 to OGG with libopus codec using system ffmpeg.
        Returns True if successful, False if ffmpeg is missing or failed.
        """
        if not shutil.which("ffmpeg"):
            print("[WARNING] ffmpeg not found in PATH. Cannot convert to .ogg opus.")
            return False

        try:
            cmd = [
                "ffmpeg",
                "-y",  # overwrite output
                "-i",
                str(input_mp3_path),
                "-c:a",
                "libopus",
                "-b:a",
                "32k",
                "-vbr",
                "on",
                str(output_ogg_path),
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to convert with ffmpeg: {e}")
            return False
