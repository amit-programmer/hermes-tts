import io
import os
from typing import Optional
import requests


class ElevenLabsEngine:
    """
    ElevenLabs Engine:
    Uses the official ElevenLabs REST API via requests (zero Windows path-length issues, ultra-reliable).
    Can also use the elevenlabs SDK if installed.
    """

    def __init__(self, api_key: Optional[str] = None, default_voice_id: Optional[str] = None):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.default_voice_id = default_voice_id or os.getenv("ELEVENLABS_VOICE_ID", "cgSgspJ2msm6clMCkdW9")
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=1)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def generate(
        self,
        text: str,
        voice_id: Optional[str] = None,
        model: str = "eleven_multilingual_v2",
    ) -> io.BytesIO:
        """
        Synthesizes text into an MP3 byte stream using ElevenLabs.
        Configured for hyper-realistic human speech (natural breathing, emotional nuance, vocal fry).
        Supports seamless failover: tries ELEVENLABS_API_KEY first, and if that fails, tries ELEVENLABS_API_KEY_2.
        """
        # Dynamically fetch keys so runtime .env edits take effect
        key1 = (os.getenv("ELEVENLABS_API_KEY") or self.api_key or "").strip()
        key2 = (os.getenv("ELEVENLABS_API_KEY_2") or "").strip()

        api_keys = []
        if key1:
            api_keys.append(("Primary Key (ELEVENLABS_API_KEY)", key1))
        if key2 and key2 != key1:
            api_keys.append(("Secondary Key (ELEVENLABS_API_KEY_2)", key2))

        if not api_keys:
            raise ValueError(
                "Neither ELEVENLABS_API_KEY nor ELEVENLABS_API_KEY_2 is configured in .env."
            )

        target_voice = voice_id or self.default_voice_id or "cgSgspJ2msm6clMCkdW9"
        if target_voice == "21m00Tcm4TlvDq8ikWAM":
            target_voice = "cgSgspJ2msm6clMCkdW9"

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{target_voice}"
        payload = {
            "text": text,
            "model_id": model,
            "voice_settings": {
                "stability": 0.40,
                "similarity_boost": 0.80,
                "style": 0.15,
                "use_speaker_boost": True,
            },
        }

        errors = []
        for key_label, key_val in api_keys:
            masked_key = f"...{key_val[-6:]}" if len(key_val) > 6 else "***"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": key_val,
            }
            try:
                print(f"[ELEVENLABS] Attempting synthesis with {key_label} [{masked_key}]...")
                response = self._session.post(url, json=payload, headers=headers, timeout=8)

                # Catch 402 library_voice restriction on tier
                if response.status_code == 402 and "library_voice" in response.text:
                    response.close()
                    print("[ELEVENLABS] Voice requires paid plan. Falling back to default Jessica voice...")
                    url_fallback = "https://api.elevenlabs.io/v1/text-to-speech/cgSgspJ2msm6clMCkdW9"
                    response = self._session.post(url_fallback, json=payload, headers=headers, timeout=8)

                if response.status_code == 200:
                    print(f"[SUCCESS] ElevenLabs audio generated successfully using {key_label} [{masked_key}].")
                    buffer = io.BytesIO(response.content)
                    buffer.seek(0)
                    return buffer
                else:
                    err_detail = f"{key_label} [{masked_key}] HTTP {response.status_code}: {response.text[:140]}"
                    print(f"[WARN] {err_detail}. Attempting failover if next key available...")
                    errors.append(err_detail)
            except Exception as e:
                err_detail = f"{key_label} [{masked_key}] Exception: {str(e)}"
                print(f"[WARN] {err_detail}. Attempting failover if next key available...")
                errors.append(err_detail)

        raise RuntimeError(
            f"All ElevenLabs API keys failed ({len(api_keys)} tried). Summary:\n" + "\n".join(f" - {e}" for e in errors)
        )

    def stream_audio(
        self,
        text: str,
        voice_id: Optional[str] = None,
        model: str = "eleven_multilingual_v2",
    ):
        """Streams audio chunks in real-time as they arrive from ElevenLabs."""
        key1 = (os.getenv("ELEVENLABS_API_KEY") or self.api_key or "").strip()
        key2 = (os.getenv("ELEVENLABS_API_KEY_2") or "").strip()
        api_keys = [k for k in [key1, key2] if k]
        target_voice = voice_id or self.default_voice_id or "cgSgspJ2msm6clMCkdW9"
        if target_voice == "21m00Tcm4TlvDq8ikWAM":
            target_voice = "cgSgspJ2msm6clMCkdW9"

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{target_voice}/stream"
        payload = {
            "text": text,
            "model_id": model,
            "voice_settings": {
                "stability": 0.40,
                "similarity_boost": 0.80,
                "style": 0.15,
                "use_speaker_boost": True,
            },
        }

        for key in api_keys:
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": key,
            }
            try:
                response = self._session.post(url, json=payload, headers=headers, timeout=25, stream=True)
                if response.status_code == 200:
                    for chunk in response.iter_content(chunk_size=4096):
                        if chunk:
                            yield chunk
                    return
            except Exception as e:
                print(f"[WARN] ElevenLabs stream key failed: {e}")
