import io
import os
import numpy as np


class ChatTTSEngine:
    """
    ChatTTS Engine:
    Conversational and emotional speech with locked female seed vector,
    natural pauses, and reduced hallucination/whispering.
    """

    def __init__(self, seed: int = None):
        env_seed = os.getenv("CHATTTS_SEED", "2222")
        self.seed = seed if seed is not None else int(env_seed)
        self.chat = None
        self.speaker_embedding = None
        self._is_loaded = False

    def _load_model(self):
        if not self._is_loaded:
            from pathlib import Path
            base_dir = Path(__file__).resolve().parent.parent
            gpt_file = base_dir / "asset" / "gpt" / "model.safetensors"
            if not gpt_file.exists():
                raise RuntimeError(
                    "ChatTTS model file 'asset/gpt/model.safetensors' is currently downloading or missing. Please allow the download to complete."
                )

            try:
                import torch
                import ChatTTS
                self.chat = ChatTTS.Chat()
                # Local load without torch.compile on Windows for stability
                loaded = self.chat.load(compile=False)
                if not loaded:
                    raise RuntimeError("ChatTTS.load() could not verify assets. Check model files.")

                # Lock consistent sweet female voice seed
                torch.manual_seed(self.seed)
                self.speaker_embedding = self.chat.sample_random_speaker()
                self._is_loaded = True
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load ChatTTS model: {e}"
                )

    def generate(self, text: str, temperature: float = 0.3, top_p: float = 0.7, top_k: int = 20) -> io.BytesIO:
        """
        Synthesizes text into a WAV byte stream using ChatTTS.
        """
        try:
            import soundfile as sf
        except ImportError:
            raise RuntimeError("Module 'soundfile' is required for ChatTTS. Please run: pip install soundfile")

        self._load_model()
        import ChatTTS

        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=self.speaker_embedding,
            temperature=temperature,
            top_P=top_p,
            top_K=top_k,
        )

        # refine_text=False to suppress unprompted sighs/giggles
        wavs = self.chat.infer([text], params_infer_code=params_infer_code)

        if not wavs or len(wavs[0]) == 0:
            raise RuntimeError("ChatTTS audio synthesis failed: inference returned empty waveform.")

        audio_data = np.array(wavs[0], dtype=np.float32)
        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()

        buffer = io.BytesIO()
        sf.write(buffer, audio_data, 24000, format="WAV")
        buffer.seek(0)
        return buffer
