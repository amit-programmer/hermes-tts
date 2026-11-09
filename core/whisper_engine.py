import os
import time
import tempfile
import io
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple

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
            # In Hugging Face ZeroGPU, raw FastAPI endpoints run outside Gradio's @spaces.GPU context.
            # Calling device="cuda" outside @spaces.GPU triggers ZeroGPU's:
            # "Low-level CUDA init (`torch._C._cuda_init`) reached"
            # Therefore, we safely use device="cpu" on Hugging Face Spaces for 100% reliable Postman/REST calls.
            # Whisper-base on CPU is ultra-lightweight (~140MB) and transcribes in ~250ms without GPU overhead.
            if os.getenv("SPACE_ID") or os.getenv("SPACES_ZERO_GPU"):
                self._device = "cpu"
            else:
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
            device = self._get_device()
            try:
                import torch
                from transformers import pipeline

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
                err_str = str(e)
                # If CUDA initialization failed or was intercepted by ZeroGPU, gracefully fallback to CPU
                if "cuda" in err_str.lower() or "torch._c._cuda_init" in err_str.lower() or device != "cpu":
                    print(f"[STT] CUDA init intercepted or failed ({e}). Auto-falling back to CPU...")
                    try:
                        import torch
                        from transformers import pipeline
                        self._device = "cpu"
                        self._pipe = pipeline(
                            task="automatic-speech-recognition",
                            model=self.model_id,
                            torch_dtype=torch.float32,
                            device="cpu",
                        )
                        print(f"[STT] Whisper engine successfully loaded on CPU fallback.")
                    except Exception as fallback_e:
                        raise RuntimeError(f"Whisper CPU fallback also failed: {fallback_e}")
                else:
                    raise RuntimeError(
                        f"Failed to initialize Whisper STT engine '{self.model_id}': {e}. "
                        f"Ensure 'transformers', 'torch' and 'soundfile' are installed."
                    )
        return self._pipe

    @staticmethod
    def _load_audio_array(audio_input: Union[str, bytes, io.BytesIO]) -> np.ndarray:
        """
        Robustly loads any audio input (filepath, bytes, BytesIO) into a 1D float32 16kHz NumPy array.
        Uses transformers.pipelines.audio_utils.ffmpeg_read (natively decodes MP3, WAV, WEBM, OGG, M4A, AAC, FLAC).
        """
        if isinstance(audio_input, (bytes, bytearray)):
            audio_bytes = bytes(audio_input)
        elif isinstance(audio_input, io.BytesIO):
            audio_bytes = audio_input.getvalue()
        elif isinstance(audio_input, str):
            with open(audio_input, "rb") as f:
                audio_bytes = f.read()
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

        # 1. Native transformers decoder via ffmpeg_read
        try:
            from transformers.pipelines.audio_utils import ffmpeg_read
            audio_array = ffmpeg_read(audio_bytes, sampling_rate=16000)
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            return audio_array.astype(np.float32)
        except Exception as e:
            print(f"[STT] ffmpeg_read fallback notice: {e}")

        # 2. In-memory soundfile fallback
        try:
            import soundfile as sf
            data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            return data.astype(np.float32)
        except Exception as sf_e:
            raise RuntimeError(f"Failed to decode audio: {sf_e}")

    def transcribe(
        self,
        audio_input: Union[str, bytes, io.BytesIO],
        language: Optional[str] = None,
        task: str = "transcribe",
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribes speech from audio input into text.
        Supports all audio formats (MP3, WAV, WEBM, OGG, M4A, etc.) via direct NumPy payload.

        Args:
            audio_input: File path (str), raw audio bytes, or BytesIO buffer.
            language: Optional language code (e.g. 'hindi', 'english', 'hi', 'en').
            task: 'transcribe' or 'translate'.
            filename: Original file name if available.

        Returns:
            Dict containing 'text', 'language', 'duration_ms', and 'engine'.
        """
        t0 = time.time()
        pipe = self._get_pipeline()

        audio_array = self._load_audio_array(audio_input)
        payload = {"sampling_rate": 16000, "raw": audio_array}

        # 1. Build generation kwargs
        generate_kwargs = {"task": task}
        if language and language.lower() not in ["auto", "none", ""]:
            lang_code = language.lower().strip()
            lang_map = {
                "hindi": "hindi",
                "hi": "hindi",
                "english": "english",
                "en": "english",
                "hinglish": "hindi",
            }
            generate_kwargs["language"] = lang_map.get(lang_code, lang_code)

        # 2. Perform transcription
        try:
            result = pipe(
                payload,
                chunk_length_s=30,
                generate_kwargs=generate_kwargs,
                return_timestamps=False,
            )
        except Exception as pipe_err:
            err_msg = str(pipe_err).lower()
            if "cuda" in err_msg or "torch._c._cuda_init" in err_msg:
                print(f"[STT] CUDA error during forward pass: {pipe_err}. Retrying on CPU...")
                import torch
                from transformers import pipeline
                self._device = "cpu"
                self._pipe = pipeline(
                    task="automatic-speech-recognition",
                    model=self.model_id,
                    torch_dtype=torch.float32,
                    device="cpu",
                )
                result = self._pipe(
                    payload,
                    chunk_length_s=30,
                    generate_kwargs=generate_kwargs,
                    return_timestamps=False,
                )
            else:
                raise

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

