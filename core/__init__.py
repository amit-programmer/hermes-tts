try:
    from .kokoro_engine import KokoroEngine
except Exception:
    KokoroEngine = None

try:
    from .chattts_engine import ChatTTSEngine
except Exception:
    ChatTTSEngine = None

try:
    from .elevenlabs_engine import ElevenLabsEngine
except Exception:
    ElevenLabsEngine = None

try:
    from .edge_engine import EdgeTTSEngine
except Exception:
    EdgeTTSEngine = None

try:
    from .whisper_engine import WhisperEngine
except Exception:
    WhisperEngine = None

__all__ = [
    "KokoroEngine",
    "ChatTTSEngine",
    "ElevenLabsEngine",
    "EdgeTTSEngine",
    "WhisperEngine",
]
