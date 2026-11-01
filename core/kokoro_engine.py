import io
import os
import numpy as np


class KokoroEngine:
    """
    Kokoro-82M Engine:
    Zero-cost, local CPU-friendly lightweight neural TTS.
    Produces clear young female voice (e.g. af_bella, af_sarah).
    """

    def __init__(self, lang_code: str = "a", default_voice: str = "af_bella"):
        self.lang_code = lang_code
        self.default_voice = default_voice or os.getenv("KOKORO_VOICE", "af_bella")
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                from kokoro import KPipeline
                self._pipeline = KPipeline(lang_code=self.lang_code)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize Kokoro pipeline (lang_code='{self.lang_code}'). "
                    f"Ensure 'kokoro>=0.3.4', 'soundfile' and espeak-ng are installed. Error: {e}"
                )
        return self._pipeline

    def generate(self, text: str, voice: str = None, speed: float = 1.0) -> io.BytesIO:
        """
        Synthesizes text into a WAV byte stream using Kokoro-82M.
        """
        try:
            import soundfile as sf
        except ImportError:
            raise RuntimeError("Module 'soundfile' is required for Kokoro. Please run: pip install soundfile")

        try:
            import torch
        except ImportError:
            torch = None

        pipeline = self._get_pipeline()
        target_voice = voice or self.default_voice
        generator = pipeline(text, voice=target_voice, speed=speed, split_pattern=r"\n+")

        all_audio = []
        sample_rate = 24000

        for _, _, audio in generator:
            if audio is not None and len(audio) > 0:
                if torch is not None and isinstance(audio, torch.Tensor):
                    audio_np = audio.detach().cpu().numpy()
                else:
                    audio_np = np.asarray(audio, dtype=np.float32)
                all_audio.append(audio_np)

        if not all_audio:
            raise RuntimeError("Kokoro audio synthesis failed: generator yielded no audio chunks.")

        final_audio = np.concatenate(all_audio, axis=0)

        buffer = io.BytesIO()
        sf.write(buffer, final_audio, sample_rate, format="WAV")
        buffer.seek(0)
        return buffer
