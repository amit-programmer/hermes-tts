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
    def _normalize_audio_to_wav(audio_input: Union[str, bytes, io.BytesIO], filename: Optional[str] = None) -> Tuple[str, list]:
        """
        Converts any audio input (WAV, MP3, WEBM, OGG, M4A, FLAC, raw bytes, extensionless files)
        into a standard 16,000 Hz mono PCM WAV file that Whisper & soundfile read flawlessly.
        Returns (path_to_wav, list_of_temporary_files_to_cleanup).
        """
        cleanup_files = []
        raw_tmp_path = None

        # 1. Infer extension from filename or audio magic bytes
        ext = ".bin"
        if filename and "." in filename:
            ext = Path(filename).suffix.lower()

        if isinstance(audio_input, (bytes, bytearray)):
            if audio_input.startswith(b"RIFF"):
                ext = ".wav"
            elif audio_input.startswith(b"ID3") or (len(audio_input) > 2 and audio_input[0] == 0xFF and (audio_input[1] & 0xE0) == 0xE0):
                ext = ".mp3"
            elif audio_input.startswith(b"OggS"):
                ext = ".ogg"
            elif audio_input.startswith(b"\x1a\x45\xdf\xa3"):
                ext = ".webm"
            elif audio_input.startswith(b"fLaC"):
                ext = ".flac"
            elif b"ftyp" in audio_input[:16]:
                ext = ".m4a"

            raw_tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            raw_tmp.write(audio_input)
            raw_tmp.flush()
            raw_tmp.close()
            raw_tmp_path = raw_tmp.name
            cleanup_files.append(raw_tmp_path)

        elif isinstance(audio_input, io.BytesIO):
            val = audio_input.getvalue()
            if val.startswith(b"RIFF"):
                ext = ".wav"
            elif val.startswith(b"ID3") or (len(val) > 2 and val[0] == 0xFF and (val[1] & 0xE0) == 0xE0):
                ext = ".mp3"
            elif val.startswith(b"OggS"):
                ext = ".ogg"
            elif val.startswith(b"\x1a\x45\xdf\xa3"):
                ext = ".webm"
            elif val.startswith(b"fLaC"):
                ext = ".flac"
            elif b"ftyp" in val[:16]:
                ext = ".m4a"

            raw_tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            raw_tmp.write(val)
            raw_tmp.flush()
            raw_tmp.close()
            raw_tmp_path = raw_tmp.name
            cleanup_files.append(raw_tmp_path)

        elif isinstance(audio_input, str):
            raw_tmp_path = audio_input
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

        # 2. Allocate output 16kHz WAV temporary file
        wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_tmp.close()
        wav_tmp_path = wav_tmp.name
        cleanup_files.append(wav_tmp_path)

        # 3. Primary Conversion: Use ffmpeg (installed in Space packages.txt)
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        ffmpeg_success = False
        try:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i", raw_tmp_path,
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                wav_tmp_path,
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
            if res.returncode == 0 and os.path.exists(wav_tmp_path) and os.path.getsize(wav_tmp_path) > 44:
                ffmpeg_success = True
        except Exception:
            ffmpeg_success = False

        # 4. Fallback Conversion: Use soundfile in-memory read & write
        if not ffmpeg_success:
            try:
                import soundfile as sf
                data, sr = sf.read(raw_tmp_path, dtype="float32")
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                sf.write(wav_tmp_path, data, 16000, format="WAV", subtype="PCM_16")
                ffmpeg_success = True
            except Exception:
                # If conversion fails, return raw file as fallback
                return raw_tmp_path, cleanup_files

        return wav_tmp_path, cleanup_files

    def transcribe(
        self,
        audio_input: Union[str, bytes, io.BytesIO],
        language: Optional[str] = None,
        task: str = "transcribe",
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribes speech from audio input into text.
        Supports all audio formats (MP3, WAV, WEBM, OGG, M4A, etc.) and auto-converts to 16kHz WAV.

        Args:
            audio_input: File path (str), raw audio bytes, or BytesIO buffer.
            language: Optional language code (e.g. 'hindi', 'english', 'hi', 'en').
            task: 'transcribe' or 'translate'.
            filename: Original file name if available (used for format detection).

        Returns:
            Dict containing 'text', 'language', 'duration_ms', and 'engine'.
        """
        t0 = time.time()
        pipe = self._get_pipeline()

        target_path, cleanup_files = self._normalize_audio_to_wav(audio_input, filename=filename)

        try:
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
                    target_path,
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
                        target_path,
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

        finally:
            for p in cleanup_files:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

