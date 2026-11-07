import os
import time
import tempfile
import io
from pathlib import Path
from typing import Optional, Dict, Any, Union

import numpy as np


class WhisperEngine:
    """
    Hermes Speech-to-Text (STT) Engine:
    Powered by OpenAI Whisper via Hugging Face Transformers.
    Runs on ZeroGPU (NVIDIA RTX Pro 6000 Blackwell) or CPU fallback.
    Multilingual support: Hindi, Hinglish, English, and 90+ other languages.
    """

    def __init__(self, model_id: str = "openai/whisper-base"):
        self.model_id = os.getenv("WHISPER_MODEL", model_id)
        self._pipe = None
        self._device = None

    def _get_device(self) -> str:
        if self._device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    self._device = "cuda"
                else:
                    self._device = "cpu"
            except Exception:
                self._device = "cpu"
        return self._device

    def _get_pipeline(self):
        if self._pipe is None:
            try:
                import torch
                from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

                device = self._get_device()
                torch_dtype = torch.float16 if device == "cuda" else torch.float32

                print(f"[STT] Loading Whisper model '{self.model_id}' on {device} ({torch_dtype})...")
                self._pipe = pipeline(
                    task="automatic-speech-recognition",
                    model=self.model_id,
                    torch_dtype=torch_dtype,
                    device=device,
                )
                print(f"[STT] Whisper engine successfully loaded on {device}.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize Whisper STT engine '{self.model_id}': {e}. "
                    f"Ensure 'transformers', 'torch' and 'soundfile' are installed."
                )
        return self._pipe

    def transcribe(
        self,
        audio_input: Union[str, bytes, io.BytesIO],
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> Dict[str, Any]:
        """
        Transcribes speech from audio input into text.

        Args:
            audio_input: File path (str), raw audio bytes, or BytesIO buffer.
            language: Optional language code (e.g. 'hindi', 'english', 'hi', 'en').
            task: 'transcribe' or 'translate'.

        Returns:
            Dict containing 'text', 'language', 'duration_ms', and 'engine'.
        """
        t0 = time.time()
        pipe = self._get_pipeline()

        temp_path = None
        target_path = None

        try:
            # 1. Resolve audio input into a filepath if bytes were passed
            if isinstance(audio_input, (bytes, bytearray)):
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(audio_input)
                tmp.flush()
                tmp.close()
                temp_path = tmp.name
                target_path = temp_path
            elif isinstance(audio_input, io.BytesIO):
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(audio_input.getvalue())
                tmp.flush()
                tmp.close()
                temp_path = tmp.name
                target_path = temp_path
            elif isinstance(audio_input, str):
                target_path = audio_input
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

            # 2. Build generation kwargs
            generate_kwargs = {"task": task}
            if language and language.lower() not in ["auto", "none", ""]:
                lang_code = language.lower().strip()
                # Map common language names
                lang_map = {
                    "hindi": "hindi",
                    "hi": "hindi",
                    "english": "english",
                    "en": "english",
                    "hinglish": "hindi",
                }
                generate_kwargs["language"] = lang_map.get(lang_code, lang_code)

            # 3. Perform transcription
            result = pipe(
                target_path,
                chunk_length_s=30,
                generate_kwargs=generate_kwargs,
                return_timestamps=False,
            )

            transcribed_text = ""
            if isinstance(result, dict):
                transcribed_text = result.get("text", "").strip()
            elif isinstance(result, str):
                transcribed_text = result.strip()

            elapsed_ms = int((time.time() - t0) * 1000)

            return {
                "text": transcribed_text,
                "language": generate_kwargs.get("language", "auto-detected"),
                "duration_ms": elapsed_ms,
                "engine": f"whisper ({self.model_id})",
                "device": self._get_device(),
            }

        finally:
            # Clean up temporary file if created
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
